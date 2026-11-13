"""Live audio transport tests without microphone or speaker access."""

from threading import Event, Thread, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_audio.sts import live


def status(**kwargs):
    return SimpleNamespace(input_overflow=False, output_underflow=False, **kwargs)


def test_ring_buffer_wraparound_and_capacity():
    ring = live._AudioBuffer(7)
    ring.write(np.arange(5))
    first = np.empty(3)
    assert ring.read_into(first) == 3
    np.testing.assert_array_equal(first, [0, 1, 2])
    ring.write(np.arange(5, 10))
    with pytest.raises(BufferError):
        ring.write([10])
    rest = np.full(9, -1.0)
    assert ring.read_into(rest) == 7
    np.testing.assert_array_equal(rest, [3, 4, 5, 6, 7, 8, 9, -1, -1])
    assert ring.size == 0


def test_callback_fills_silence_and_copies_input():
    bridge = live._LiveAudio(20)
    bridge.enqueue(np.array([0.1, 0.2]))
    audio = np.arange(4, dtype=np.float32)[:, None]
    output = np.empty_like(audio)
    bridge.callback(audio, output, 4, None, status())
    np.testing.assert_allclose(output[:, 0], [0.1, 0.2, 0, 0])
    assert bridge.underruns == 1
    audio.fill(99)
    np.testing.assert_array_equal(
        bridge.receive(SimpleNamespace(active=True)), np.arange(4)
    )


def test_callback_stops_only_after_final_output_is_filled():
    bridge = live._LiveAudio(20)
    bridge.enqueue(np.arange(5, dtype=np.float32))
    bridge.finish()
    output = np.empty((3, 1), dtype=np.float32)
    bridge.callback(np.zeros_like(output), output, 3, None, status())
    np.testing.assert_array_equal(output[:, 0], [0, 1, 2])
    with pytest.raises(live.sd.CallbackStop):
        bridge.callback(np.zeros_like(output), output, 3, None, status())
    np.testing.assert_array_equal(output[:, 0], [3, 4, 0])
    assert bridge.input.size == bridge.underruns == 0


def test_driver_underflow_is_counted_with_nonempty_playback_queue():
    bridge = live._LiveAudio(20)
    bridge.enqueue(np.ones(8, dtype=np.float32))
    output = np.empty((4, 1), dtype=np.float32)
    flags = SimpleNamespace(input_overflow=False, output_underflow=True)
    bridge.callback(np.zeros_like(output), output, 4, None, flags)
    np.testing.assert_array_equal(output, 1)
    assert bridge.underruns == 1


def test_full_playback_buffer_rejects_without_overwriting():
    bridge = live._LiveAudio(3)
    bridge.enqueue(np.array([1, 2], dtype=np.float32))
    with pytest.raises(RuntimeError, match="playback could not keep up"):
        bridge.enqueue(np.array([3, 4], dtype=np.float32))
    output = np.empty(2, dtype=np.float32)
    bridge.output.read_into(output)
    np.testing.assert_array_equal(output, [1, 2])


@pytest.mark.parametrize("hardware_overflow", [False, True])
def test_input_overload_reaches_the_inference_thread(hardware_overflow):
    bridge = live._LiveAudio(3)
    output = np.ones((4, 1), dtype=np.float32)
    flags = SimpleNamespace(input_overflow=hardware_overflow, output_underflow=False)
    with pytest.raises(live.sd.CallbackAbort):
        bridge.callback(output.copy(), output, 4, None, flags)
    np.testing.assert_array_equal(output, 0)
    with pytest.raises(
        RuntimeError,
        match="Microphone overflow" if hardware_overflow else "could not keep up",
    ):
        bridge.receive(SimpleNamespace(active=False))


def test_stopped_stream_does_not_leave_receiver_waiting():
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        live._LiveAudio(3).receive(SimpleNamespace(active=False))


class FakeSession:
    def __init__(self, frames=3, fail=None):
        self.limit = frames
        self.fail = fail
        self.index = 0
        self.pending = np.zeros(0, dtype=np.float32)
        self.received = []
        self.threads = []
        self.resolved = []
        self.synchronized = False
        self.warmed = False
        self.in_flight = self.max_in_flight = 0

    def warmup(self):
        self.threads.append(get_ident())
        self.warmed = True

    def frame(self):
        return SimpleNamespace(
            audio=np.full(4, (self.index + 1) / 10, dtype=np.float32),
            text=f"frame{self.index} ",
            done=self.index == self.limit - 1,
        )

    def defer(self, frame):
        index = self.index
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)

        def resolve():
            if self.fail == "decode" and index:
                raise RuntimeError("decoder failed")
            self.resolved.append(index)
            self.in_flight -= 1
            self.threads.append(get_ident())
            return frame

        return SimpleNamespace(done=frame.done, resolve=resolve)

    def start_async(self):
        self.threads.append(get_ident())
        if self.fail == "start":
            raise RuntimeError("prefill failed")
        return self.defer(self.frame())

    def feed(self, audio):
        self.threads.append(get_ident())
        self.pending = np.concatenate([self.pending, audio])

    @property
    def frames_ready(self):
        return len(self.pending) // 4

    def step_async(self):
        self.threads.append(get_ident())
        if self.fail == "step":
            raise RuntimeError("inference failed")
        if self.fail == "interrupt":
            raise KeyboardInterrupt
        self.received.append(self.pending[:4])
        self.pending = self.pending[4:]
        self.index += 1
        return self.defer(self.frame())

    def synchronize_decoder(self):
        self.synchronized = True
        self.threads.append(get_ident())


@pytest.fixture
def audio_driver(monkeypatch):
    streams = []
    settings = []

    class FakeStream:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.active = False
            self.closed = False
            self.latency = (0.01, 0.01)
            self.device = (2, 3)
            self.samplerate = kwargs["samplerate"]
            self.output = []
            self.stop_event = Event()
            self.thread = None
            self.error = None
            streams.append(self)

        def start(self):
            self.active = True
            self.thread = Thread(target=self.pump)
            self.thread.start()

        def pump(self):
            position = 0
            iteration = 0
            try:
                while not self.stop_event.is_set():
                    size = [3, 7, 2, 5][iteration % 4]
                    incoming = np.arange(position, position + size, dtype=np.float32)[
                        :, None
                    ]
                    outgoing = np.empty_like(incoming)
                    try:
                        self.options["callback"](
                            incoming, outgoing, size, None, status()
                        )
                    except live.sd.CallbackStop:
                        self.output.append(outgoing[:, 0].copy())
                        break
                    self.output.append(outgoing[:, 0].copy())
                    position += size
                    iteration += 1
                    self.stop_event.wait(0.002)
            except live.sd.CallbackAbort:
                pass
            except Exception as error:  # noqa: BLE001
                # Forward callback errors to the test's assertion.
                self.error = error
            finally:
                self.active = False
                self.options["finished_callback"]()

        def abort(self):
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=2)
                assert not self.thread.is_alive()
            self.active = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(live.sd, "Stream", FakeStream)
    monkeypatch.setattr(
        live.sd, "query_devices", lambda device, kind: {"name": f"Test {kind} {device}"}
    )
    monkeypatch.setattr(
        live.sd, "check_input_settings", lambda **kw: settings.append(("input", kw))
    )
    monkeypatch.setattr(
        live.sd, "check_output_settings", lambda **kw: settings.append(("output", kw))
    )
    return streams, settings


@pytest.mark.parametrize("frames", [1, 3, 10])
def test_live_loop_preserves_audio_order_and_drains(audio_driver, capsys, frames):
    session = FakeSession(frames)
    result = live.run_session(
        session,
        1000,
        input_device=2,
        output_device="Speakers",
    )
    streams, settings = audio_driver
    stream = streams[0]
    assert result.frames == frames
    assert result.underruns == 0
    assert stream.closed and not stream.active and stream.error is None
    assert stream.options["device"] == (2, "Speakers")
    assert stream.options["blocksize"] == 0
    assert settings[0][1]["samplerate"] == 1000
    assert set(session.threads) == {get_ident()}
    assert stream.thread.ident != get_ident()
    expected = np.concatenate(
        [np.zeros(100), *[np.full(4, (i + 1) / 10) for i in range(frames)]]
    )
    actual = np.concatenate(stream.output)
    np.testing.assert_allclose(actual[: len(expected)], expected)
    np.testing.assert_array_equal(actual[len(expected) :], 0)
    if frames > 1:
        np.testing.assert_array_equal(
            np.concatenate(session.received), np.arange(4 * (frames - 1))
        )
    assert session.resolved == list(range(frames))
    assert session.warmed and session.synchronized
    assert session.max_in_flight <= 2 and session.in_flight == 0
    logs = capsys.readouterr()
    assert logs.out == "".join(f"frame{i} " for i in range(frames)) + "\n"
    assert "[audio]" not in logs.err and "[mic]" not in logs.err


@pytest.mark.parametrize("failure", ["start", "step", "decode", "interrupt"])
def test_live_loop_closes_devices_on_failure(audio_driver, failure):
    session = FakeSession(fail=failure)
    expected = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    with pytest.raises(expected):
        live.run_session(session, 1000)
    stream = audio_driver[0][0]
    assert stream.closed and not stream.active
    assert session.synchronized


def test_cli_reads_prompt_and_passes_options(tmp_path, monkeypatch):
    card = tmp_path / "card.txt"
    card.write_text("Your name is Haru.\n", encoding="utf-8")
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return live.LiveStats(2, 0)

    monkeypatch.setattr(live, "run_live", run)
    live.main(
        [
            "--model",
            "local/model",
            "--ref-audio",
            "voice.wav",
            "--system-prompt-file",
            str(card),
            "--input-device",
            "2",
            "--output-device",
            "MacBook Pro Speakers",
            "--max-frames",
            "2",
            "--speak-first",
            "--steps",
            "4",
            "--seed",
            "42",
        ]
    )
    assert captured["system_prompt"] == "Your name is Haru.\n"
    assert captured["input_device"] == 2
    assert captured["output_device"] == "MacBook Pro Speakers"
    assert captured["session_kwargs"] == {
        "max_frames": 2,
        "speak_first": True,
        "seed": 42,
        "meanflow_steps": 4,
    }
    assert captured["ref_audio"] == "voice.wav"


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_steps_before_loading(value, monkeypatch):
    monkeypatch.setattr(
        live, "run_live", lambda **kwargs: pytest.fail("must not load a model")
    )
    with pytest.raises(SystemExit) as error:
        live.main(["--steps", value])
    assert error.value.code == 2


def test_list_devices_exits_without_loading(monkeypatch, capsys):
    monkeypatch.setattr(live.sd, "query_devices", lambda: "microphone / speakers")
    monkeypatch.setattr(
        live, "run_live", lambda **kwargs: pytest.fail("should not load a model")
    )
    live.main(["--list-devices"])
    assert capsys.readouterr().out == "microphone / speakers\n"


def test_cli_handles_interrupt_and_device_errors(monkeypatch, capsys):
    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(live, "run_live", interrupt)
    live.main([])
    assert "Session stopped" in capsys.readouterr().err

    def fail(**kwargs):
        raise live.sd.PortAudioError("Invalid sample rate")

    monkeypatch.setattr(live, "run_live", fail)
    with pytest.raises(SystemExit) as error:
        live.main([])
    assert error.value.code == 1
    assert "Invalid sample rate" in capsys.readouterr().err


def test_haru_session_options_and_reference(tmp_path, monkeypatch):
    from mlx_audio import audio_io
    from mlx_audio.sts import utils
    from mlx_audio.sts.models.haru import DuplexConfig

    reference = tmp_path / "voice.wav"
    audio_io.write(str(reference), np.ones(2400, dtype=np.float32) * 0.1, 24000)
    captured = {}

    def create_session(**kwargs):
        captured.update(kwargs)
        return "session"

    model = SimpleNamespace(
        model_type="haru", sample_rate=24000, create_session=create_session
    )

    def load(path, **kwargs):
        assert path == "test/checkpoint" and kwargs == {"strict": True}
        return model

    monkeypatch.setattr(utils, "load_model", load)
    monkeypatch.setattr(
        live, "run_session", lambda session, rate, **kwargs: (session, rate)
    )
    assert live.run_live(
        "test/checkpoint",
        ref_audio=reference,
        session_kwargs={"seed": 1, "meanflow_steps": 2},
    ) == ("session", 24000)
    assert isinstance(captured["config"], DuplexConfig)
    assert captured["config"].seed == 1
    assert captured["config"].meanflow_steps == 2
    assert captured["reference"].shape == (2400,)
    with pytest.raises(ValueError, match="supports Haru"):
        live.run_live(SimpleNamespace(model_type="other"))


@pytest.mark.parametrize("buffer", [-0.1, 1.0, float("nan")])
def test_invalid_playback_buffer_rejected_before_opening_audio(buffer, monkeypatch):
    monkeypatch.setattr(
        live.sd, "Stream", lambda **kwargs: pytest.fail("must not open devices")
    )
    with pytest.raises(ValueError, match="playback_buffer"):
        live.run_session(FakeSession(), 24000, playback_buffer=buffer)
