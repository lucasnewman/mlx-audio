"""Run a full-duplex STS session with a microphone and speakers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event

import numpy as np
import sounddevice as sd


class _AudioBuffer:
    """Fixed-capacity mono ring buffer; the caller owns synchronization."""

    def __init__(self, capacity):
        self.data = np.empty(capacity, dtype=np.float32)
        self.start = 0
        self.size = 0

    def write(self, samples):
        count = len(samples)
        if count > len(self.data) - self.size:
            raise BufferError("Audio buffer is full")
        end = (self.start + self.size) % len(self.data)
        first = min(count, len(self.data) - end)
        self.data[end : end + first] = samples[:first]
        self.data[: count - first] = samples[first:]
        self.size += count

    def read_into(self, destination):
        count = min(len(destination), self.size)
        first = min(count, len(self.data) - self.start)
        destination[:first] = self.data[self.start : self.start + first]
        destination[first:count] = self.data[: count - first]
        self.start = (self.start + count) % len(self.data)
        self.size -= count
        return count


class _LiveAudio:
    """Transfer PCM between the audio callback and the inference thread."""

    def __init__(self, capacity):
        self.input = _AudioBuffer(capacity)
        self.output = _AudioBuffer(capacity)
        self.condition = Condition()
        self.finished = Event()
        self.draining = False
        self.error = None
        self.underruns = 0

    def enqueue(self, audio):
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("Live session output must be mono [samples]")
        with self.condition:
            try:
                self.output.write(samples)
            except BufferError as error:
                raise RuntimeError("Speaker playback could not keep up") from error

    def callback(self, indata, outdata, frames, callback_time, status):
        # Only copy PCM here; inference, printing, and device calls stay outside.
        outdata.fill(0)
        with self.condition:
            if not self.draining:
                if status.input_overflow:
                    self.error = "Microphone overflow: audio input was lost"
                else:
                    try:
                        self.input.write(indata[:, 0])
                    except BufferError:
                        self.error = (
                            "Inference could not keep up with the microphone. "
                            "Try an 8-bit model or close other GPU workloads."
                        )
                if self.error is not None:
                    self.condition.notify_all()
                    raise sd.CallbackAbort
            count = self.output.read_into(outdata[:, 0])
            if status.output_underflow or (count < frames and not self.draining):
                self.underruns += 1
            self.condition.notify_all()
            if self.draining and self.output.size == 0:
                # CallbackStop plays the final filled buffer before stopping.
                raise sd.CallbackStop

    def receive(self, stream, *, timeout=0.1):
        with self.condition:
            if not self.input.size and self.error is None:
                if not stream.active:
                    raise RuntimeError("Audio stream stopped unexpectedly")
                self.condition.wait(timeout=timeout)
            if self.error is not None:
                raise RuntimeError(self.error)
            if not self.input.size:
                if not stream.active:
                    raise RuntimeError("Audio stream stopped unexpectedly")
                return None
            audio = np.empty(self.input.size, dtype=np.float32)
            self.input.read_into(audio)
            return audio

    def finish(self):
        with self.condition:
            self.draining = True


@dataclass
class LiveStats:
    frames: int
    underruns: int


def run_session(
    session,
    sample_rate,
    *,
    input_device=None,
    output_device=None,
    playback_buffer=0.1,
):
    """Play and capture audio until ``frame.done`` or Ctrl-C.

    The caller loads the model on this same thread. A session exposes
    ``start_async()``, ``feed(samples)``, ``frames_ready``, ``step_async()``,
    and ``synchronize_decoder()``. Resolve pending frames to get mono PCM,
    incremental text, and a completion flag. Microphone input is bounded to
    one second so overload cannot accumulate an ever-growing delay.
    """
    if not 0 <= playback_buffer <= 0.5:
        raise ValueError("playback_buffer must be between 0 and 0.5 seconds")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not all(
        callable(getattr(session, name, None))
        for name in ("start_async", "step_async", "synchronize_decoder")
    ):
        raise ValueError("This session does not support asynchronous decoding")
    bridge = _LiveAudio(int(sample_rate))
    sd.check_input_settings(
        device=input_device, channels=1, dtype="float32", samplerate=sample_rate
    )
    sd.check_output_settings(
        device=output_device, channels=1, dtype="float32", samplerate=sample_rate
    )
    stream = sd.Stream(
        samplerate=sample_rate,
        device=(input_device, output_device),
        channels=1,
        dtype="float32",
        blocksize=0,
        latency="low",
        callback=bridge.callback,
        finished_callback=bridge.finished.set,
    )
    frame_count = 0

    def emit(frame):
        nonlocal frame_count
        bridge.enqueue(frame.audio)
        frame_count += 1
        print(frame.text, end="", flush=True)

    try:
        print("Preparing the conversation…", file=sys.stderr, flush=True)
        warmup = getattr(session, "warmup", None)
        if callable(warmup):
            warmup()
        frame = session.start_async().resolve()
        bridge.enqueue(np.zeros(round(sample_rate * playback_buffer), dtype=np.float32))
        emit(frame)
        if frame.done:
            bridge.finish()
        print(
            "Listening. Press Ctrl-C to stop. Use headphones to avoid feedback.",
            file=sys.stderr,
            flush=True,
        )
        stream.start()
        while not frame.done:
            audio = bridge.receive(stream)
            if audio is not None:
                session.feed(audio)
            pending = None
            while session.frames_ready and not frame.done:
                frame = session.step_async()
                # At most two PCM frames are in flight. If input has backed
                # up, generate the next latent before resolving the previous
                # audio, allowing its decode to overlap the next backbone.
                if pending is not None:
                    emit(pending.resolve())
                pending = frame
                if not frame.done:
                    # Encode already captured microphone chunks while the
                    # decoder runs; never wait for a future mic callback.
                    audio = bridge.receive(stream, timeout=0)
                    if audio is not None:
                        session.feed(audio)
            if pending is not None:
                frame = pending.resolve()
                emit(frame)
        bridge.finish()
        # Wait for the callback's final buffer to reach the output device.
        latency = max(stream.latency)
        if not bridge.finished.wait(timeout=3.0 + latency):
            raise RuntimeError("Timed out while draining speaker audio")
        if bridge.error is not None:
            raise RuntimeError(bridge.error)
        return LiveStats(frames=frame_count, underruns=bridge.underruns)
    finally:
        try:
            stream.abort()
        finally:
            stream.close()
            session.synchronize_decoder()
            print(flush=True)


def run_live(
    model="higurashi-ai/Haru-4B",
    *,
    ref_audio=None,
    system_prompt=None,
    input_device=None,
    output_device=None,
    playback_buffer=0.1,
    session_kwargs=None,
):
    """Load a supported live STS model and connect its session to audio devices."""
    from mlx_audio.sts.utils import load_model

    if isinstance(model, (str, Path)):
        print(f"Loading {model}…", file=sys.stderr, flush=True)
        model = load_model(model, strict=True)
    if getattr(model, "model_type", None) != "haru":
        raise ValueError("The live STS CLI currently supports Haru models")

    from mlx_audio.sts.models.haru import DuplexConfig
    from mlx_audio.sts.models.haru.audio import load_audio, prepare_reference

    config = DuplexConfig(**(session_kwargs or {}))
    reference = None
    if ref_audio is not None:
        reference = prepare_reference(
            load_audio(ref_audio, target_rate=model.sample_rate),
            max_seconds=config.reference_seconds,
            sample_rate=model.sample_rate,
        )
        if not len(reference):
            raise ValueError("The reference clip contains no audible speech")
    session = model.create_session(
        system_prompt=system_prompt, reference=reference, config=config
    )
    return run_session(
        session,
        model.sample_rate,
        input_device=input_device,
        output_device=output_device,
        playback_buffer=playback_buffer,
    )


def _device(value):
    try:
        return int(value)
    except ValueError:
        return value


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Talk to a live full-duplex STS model (currently Haru)"
    )
    parser.add_argument(
        "--model", default="higurashi-ai/Haru-4B", help="Model ID or local checkpoint"
    )
    parser.add_argument("--ref-audio", help="Reference voice clip")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument(
        "--system-prompt",
        help='Override the model\'s default system prompt; pass "" for no prompt',
    )
    prompt.add_argument(
        "--system-prompt-file", type=Path, help="UTF-8 character card file"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit"
    )
    parser.add_argument("--input-device", type=_device, help="Microphone index or name")
    parser.add_argument("--output-device", type=_device, help="Speaker index or name")
    parser.add_argument(
        "--playback-buffer",
        type=float,
        default=0.1,
        help="Initial playback cushion in seconds (default: 0.1, maximum: 0.5)",
    )
    generation = parser.add_argument_group("Session options")
    generation.add_argument(
        "--speak-first", action="store_true", default=argparse.SUPPRESS
    )
    generation.add_argument(
        "--max-frames",
        type=int,
        default=argparse.SUPPRESS,
        help="Stop after this many output frames (default: 27000, one hour)",
    )
    generation.add_argument(
        "--steps",
        dest="meanflow_steps",
        metavar="STEPS",
        type=_positive_int,
        default=argparse.SUPPRESS,
        help="MeanFlow Euler steps per audio frame (positive integer; checkpoint default: 1 for Haru-4B)",
    )
    for name, kind in (
        ("temperature", float),
        ("top-k", int),
        ("top-p", float),
        ("guidance-scale", float),
        ("eop-bias", float),
        ("seed", int),
    ):
        generation.add_argument(f"--{name}", type=kind, default=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    options = vars(parse_args(argv))
    if options.pop("list_devices"):
        print(sd.query_devices())
        return
    prompt_file = options.pop("system_prompt_file")
    session_kwargs = {
        name: options.pop(name)
        for name in (
            "speak_first",
            "max_frames",
            "temperature",
            "top_k",
            "top_p",
            "guidance_scale",
            "meanflow_steps",
            "eop_bias",
            "seed",
        )
        if name in options
    }
    try:
        if prompt_file is not None:
            options["system_prompt"] = prompt_file.expanduser().read_text(
                encoding="utf-8"
            )
        stats = run_live(**options, session_kwargs=session_kwargs)
        print(f"Session finished after {stats.frames} frames.", file=sys.stderr)
        if stats.underruns:
            print(
                f"Playback underruns: {stats.underruns}. Try a larger --playback-buffer.",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        print("\nSession stopped.", file=sys.stderr)
    except (ValueError, RuntimeError, OSError, sd.PortAudioError) as error:
        print(f"Live STS error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
