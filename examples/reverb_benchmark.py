"""Benchmark Reverb variants with warm generation timings and optional ASR.

python examples/reverb_benchmark.py \
    --model bf16=/path/to/Reverb-TTS-bf16 \
    --model 4bit=/path/to/Reverb-TTS-4bit \
    --output output/reverb-benchmark --asr-model mlx-community/parakeet-tdt-0.6b-v3

Each variant runs in a separate process, sequentially. Every workload gets one
discarded warmup followed by repeated measurements with a fixed random seed.
The first measured waveform is retained. ASR runs after all timing workers exit.
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROMPTS = {
    "en": "Good morning. The weather is beautiful today, and we are taking a short walk through the park.",
    "de": "Guten Morgen. Das Wetter ist heute wunderschön, und wir machen einen kurzen Spaziergang durch den Park.",
    "fr": "Bonjour. Il fait très beau aujourd'hui, et nous allons faire une petite promenade dans le parc.",
    "es": "Buenos días. Hoy hace muy buen tiempo y vamos a dar un pequeño paseo por el parque.",
    "it": "Buongiorno. Oggi il tempo è bellissimo e facciamo una breve passeggiata nel parco.",
    "pl": "Dzień dobry. Dzisiaj jest piękna pogoda, więc wybieramy się na krótki spacer po parku.",
    "pt": "Bom dia. O tempo está muito bonito hoje e vamos dar um pequeno passeio pelo parque.",
}


def workloads():
    cases = [
        dict(name=f"{lang}-16", text=text, lang_code=lang, num_steps=16, duration_s=7.5)
        for lang, text in PROMPTS.items()
    ]
    for steps in (4, 8, 32):
        cases.append(
            dict(
                name=f"en-{steps}",
                text=PROMPTS["en"],
                lang_code="en",
                num_steps=steps,
                duration_s=7.5,
            )
        )
    cases.append(
        dict(
            name="en-clone-16",
            text=PROMPTS["en"],
            lang_code="en",
            num_steps=16,
            duration_s=7.5,
            clone=True,
        )
    )
    return cases


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(*args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def worker(args):
    import mlx.core as mx
    import numpy as np

    from mlx_audio.audio_io import write
    from mlx_audio.tts.utils import load

    label, model_path = args.model[0].split("=", 1)
    output = Path(args.output)
    audio_dir = output / "audio" / label
    audio_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = load(model_path)
    load_seconds = time.perf_counter() - started
    # Decode a distributed voice prompt to a WAV for a reproducible encode path.
    reference, transcript = model._voice_prompt("standard_male")
    reference_audio = model.audio_tokenizer.decode(reference)
    mx.eval(reference_audio)
    reference_path = audio_dir / "reference.wav"
    write(str(reference_path), reference_audio, model.sample_rate)
    del reference_audio
    records = []
    for case in workloads():
        params = {
            key: value for key, value in case.items() if key not in ("name", "clone")
        }
        if case.get("clone"):
            params.update(ref_audio=str(reference_path), ref_text=transcript)
        else:
            params["voice"] = "standard_male"
        latencies, cpu_times, memory = [], [], []
        for repeat in range(args.repeats + 1):
            mx.random.seed(args.seed)
            mx.reset_peak_memory()
            cpu_start = time.process_time()
            start = time.perf_counter()
            result = next(model.generate(**params))
            mx.eval(result.audio)
            elapsed = time.perf_counter() - start
            cpu_elapsed = time.process_time() - cpu_start
            peak = mx.get_peak_memory()
            if repeat:
                latencies.append(elapsed)
                cpu_times.append(cpu_elapsed)
                memory.append(peak)
            if repeat == 1:
                audio = np.array(result.audio)
                if not np.all(np.isfinite(audio)):
                    raise RuntimeError(f"Non-finite output: {label}/{case['name']}")
                path = audio_dir / f"{case['name']}.wav"
                write(str(path), audio, result.sample_rate)
                waveform = dict(
                    audio_path=str(path.relative_to(output)),
                    sample_rate=result.sample_rate,
                    samples=int(audio.size),
                    duration_seconds=float(audio.size / result.sample_rate),
                    rms=float(np.sqrt(np.mean(audio**2))),
                    peak_amplitude=float(np.max(np.abs(audio))),
                )
            del result
        median = statistics.median(latencies)
        record = dict(
            variant=label,
            **case,
            **waveform,
            runs_seconds=latencies,
            median_seconds=median,
            min_seconds=min(latencies),
            max_seconds=max(latencies),
            real_time_factor=median / waveform["duration_seconds"],
            realtime_speedup=waveform["duration_seconds"] / median,
            peak_memory_gb=max(memory) / 1e9,
            median_process_cpu_percent=statistics.median(
                cpu / wall * 100 for cpu, wall in zip(cpu_times, latencies)
            ),
        )
        records.append(record)
        print(
            f"{label:5s} {case['name']:12s} {median:.3f}s {record['realtime_speedup']:.1f}x peak={record['peak_memory_gb']:.2f} GB",
            flush=True,
        )
    root = Path(model_path)
    weight_files = list(root.glob("*.safetensors"))
    codec_files = list((root / "audio_tokenizer").glob("*.safetensors"))
    report = dict(
        variant=label,
        model_path=model_path,
        model_weights_bytes=sum(p.stat().st_size for p in weight_files),
        codec_weights_bytes=sum(p.stat().st_size for p in codec_files),
        model_sha256={p.name: sha256(p) for p in weight_files},
        load_seconds=load_seconds,
        records=records,
    )
    (output / f"{label}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )


def normalized_words(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, start=1):
        current = [i]
        for j, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def transcribe(reports, output, model_path):
    from mlx_audio.stt import load

    asr = load(model_path, model_type="parakeet")
    for report in reports:
        for record in report["records"]:
            hypothesis = asr.generate(str(output / record["audio_path"])).text
            reference = normalized_words(record["text"])
            errors = edit_distance(reference, normalized_words(hypothesis))
            record.update(
                transcript=hypothesis,
                word_errors=errors,
                reference_words=len(reference),
                wer=errors / len(reference),
            )
            print(
                f"ASR {report['variant']}/{record['name']}: {errors}/{len(reference)} word errors: {hypothesis}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", required=True, help="label=local-model-directory"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--asr-model", help="Parakeet checkpoint, run after generation benchmarks"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    labels = [value.partition("=")[0] for value in args.model]
    if len(set(labels)) != len(labels) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels
    ):
        parser.error("model labels must be unique simple filenames")
    if any(
        "=" not in value or not Path(value.split("=", 1)[1]).is_dir()
        for value in args.model
    ):
        parser.error("--model must specify label=existing-local-directory")
    if args.worker:
        worker(args)
        return
    import mlx.core as mx

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for entry, label in zip(args.model, labels):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--model",
                entry,
                "--output",
                str(output),
                "--repeats",
                str(args.repeats),
                "--seed",
                str(args.seed),
            ],
            check=True,
        )
        reports.append(json.loads((output / f"{label}.json").read_text()))
    if args.asr_model:
        transcribe(reports, output, args.asr_model)
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        machine=command_output("sysctl", "-n", "machdep.cpu.brand_string"),
        ram_bytes=command_output("sysctl", "-n", "hw.memsize"),
        macos=platform.mac_ver()[0],
        python=platform.python_version(),
        mlx=mx.__version__,
        git_base=command_output("git", "rev-parse", "HEAD"),
        git_dirty=bool(command_output("git", "status", "--porcelain")),
        repeats=args.repeats,
        seed=args.seed,
        warmups_per_case=1,
        asr_model=args.asr_model,
        timing="Wall time around generate() plus mx.eval(audio); excludes loading, file writes and ASR. Includes reference encoding for clone workload. Fixed seed for every repetition.",
        memory="Maximum MLX active allocation peak across measured runs, including loaded TTS model and FP32 Higgs codec; not process RSS.",
        quality="One saved utterance per workload/variant; ASR WER is a smoke check, not a standard evaluation corpus, listening score, or speaker-similarity metric.",
    )
    result = dict(metadata=metadata, variants=reports)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    fields = [
        "variant",
        "name",
        "lang_code",
        "num_steps",
        "duration_seconds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "real_time_factor",
        "realtime_speedup",
        "peak_memory_gb",
        "wer",
        "transcript",
        "audio_path",
    ]
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for report in reports:
            writer.writerows(report["records"])
    print(f"Results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
