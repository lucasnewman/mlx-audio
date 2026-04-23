#!/usr/bin/env python3
"""Exercise server-side continuous batching with Qwen3 TTS 1.7B Base.

Start the API server in one terminal:

    mlx_audio.server --host 127.0.0.1 --port 8000 \
        --tts-max-batch-size 8

Then run this script in another terminal:

    python examples/qwen3_tts_continuous_batching_server.py \
        --requests 50 --concurrency 8 --stagger-ms 300

Watch the server logs for lines like:

    [tts-continuous] queued request for in-flight runner active=...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from mlx_audio.audio_io import read as audio_read


DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
DEFAULT_TEXTS = [
    "Continuous batching should group nearby requests into one model step loop.",
    "This sentence is intentionally short so the example finishes quickly.",
    "Each HTTP request still receives its own audio file from the server.",
    "Later arrivals should join the runner while earlier requests are in flight.",
]
DEFAULT_VOICES = ["Chelsie", "Ethan"]


@dataclass
class SpeechResult:
    index: int
    status: int | None
    elapsed_s: float
    bytes_received: int
    started_at_s: float
    completed_at_s: float
    output_path: Path | None = None
    error: str | None = None


@dataclass
class AudioStats:
    sample_rate: int
    samples: int
    duration_s: float
    peak: float
    rms: float
    error: str | None = None


def _post_json(url: str, payload: dict, timeout_s: float) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        return response.status, response.read()


def _preload_model(server_url: str, model: str, timeout_s: float) -> None:
    query = urlencode({"model_name": model})
    request = Request(f"{server_url}/v1/models?{query}", method="POST")
    with urlopen(request, timeout=timeout_s) as response:
        response.read()


def _request_speech(
    *,
    index: int,
    server_url: str,
    payload: dict,
    output_dir: Path,
    response_format: str,
    timeout_s: float,
) -> SpeechResult:
    started = time.perf_counter()
    try:
        status, body = _post_json(
            f"{server_url}/v1/audio/speech",
            payload,
            timeout_s,
        )
    except HTTPError as exc:
        completed = time.perf_counter()
        detail = exc.read().decode("utf-8", errors="replace")
        return SpeechResult(
            index=index,
            status=exc.code,
            elapsed_s=completed - started,
            bytes_received=0,
            started_at_s=started,
            completed_at_s=completed,
            error=detail or str(exc),
        )
    except URLError as exc:
        completed = time.perf_counter()
        return SpeechResult(
            index=index,
            status=None,
            elapsed_s=completed - started,
            bytes_received=0,
            started_at_s=started,
            completed_at_s=completed,
            error=str(exc),
        )

    output_path = output_dir / f"qwen3_continuous_request_{index:02d}.{response_format}"
    output_path.write_bytes(body)
    completed = time.perf_counter()
    return SpeechResult(
        index=index,
        status=status,
        elapsed_s=completed - started,
        bytes_received=len(body),
        started_at_s=started,
        completed_at_s=completed,
        output_path=output_path,
    )


def _build_payloads(args: argparse.Namespace) -> list[dict]:
    texts = args.text or DEFAULT_TEXTS
    voices = args.voice or DEFAULT_VOICES
    payloads = []

    for index in range(args.requests):
        text = texts[index % len(texts)]
        if args.requests > len(texts):
            text = f"{text} Request {index + 1}."

        payloads.append(
            {
                "model": args.model,
                "input": text,
                "voice": voices[index % len(voices)],
                "lang_code": args.lang_code,
                "response_format": args.response_format,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
        )

    return payloads


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _format_bytes_per_second(value: float) -> str:
    units = ["B/s", "KiB/s", "MiB/s", "GiB/s"]
    unit_index = 0
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def _read_audio_stats(path: Path) -> AudioStats:
    try:
        audio, sample_rate = audio_read(path, always_2d=False, dtype="float32")
        audio_array = np.asarray(audio)
        samples = int(audio_array.shape[0])
        duration_s = samples / sample_rate if sample_rate > 0 else 0.0
        peak = float(np.max(np.abs(audio_array))) if samples else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio_array)))) if samples else 0.0
        return AudioStats(
            sample_rate=sample_rate,
            samples=samples,
            duration_s=duration_s,
            peak=peak,
            rms=rms,
        )
    except Exception as exc:
        return AudioStats(
            sample_rate=0,
            samples=0,
            duration_s=0.0,
            peak=0.0,
            rms=0.0,
            error=str(exc),
        )


def _collect_audio_stats(results: list[SpeechResult]) -> dict[int, AudioStats]:
    stats = {}
    for result in results:
        if result.error or result.output_path is None:
            continue
        stats[result.index] = _read_audio_stats(result.output_path)
    return stats


def _build_summary(
    results: list[SpeechResult],
    audio_stats: dict[int, AudioStats],
    *,
    total_elapsed_s: float,
    total_input_chars: int,
) -> dict:
    successes = [result for result in results if result.error is None]
    failures = [result for result in results if result.error is not None]
    latencies = [result.elapsed_s for result in successes]
    client_window_s = total_elapsed_s
    if successes:
        client_window_s = max(result.completed_at_s for result in successes) - min(
            result.started_at_s for result in successes
        )

    total_audio_s = sum(
        stats.duration_s for stats in audio_stats.values() if stats.error is None
    )
    total_bytes = sum(result.bytes_received for result in successes)
    total_samples = sum(
        stats.samples for stats in audio_stats.values() if stats.error is None
    )
    sample_rates = sorted(
        {
            stats.sample_rate
            for stats in audio_stats.values()
            if stats.error is None and stats.sample_rate
        }
    )
    per_request = []
    for result in sorted(results, key=lambda item: item.index):
        stats = audio_stats.get(result.index)
        request_summary = {
            "index": result.index,
            "status": result.status,
            "elapsed_s": result.elapsed_s,
            "bytes_received": result.bytes_received,
            "output_path": str(result.output_path) if result.output_path else None,
            "error": result.error,
        }
        if stats:
            request_summary["audio"] = {
                "sample_rate": stats.sample_rate,
                "samples": stats.samples,
                "duration_s": stats.duration_s,
                "peak": stats.peak,
                "rms": stats.rms,
                "error": stats.error,
            }
            request_summary["rtf"] = (
                result.elapsed_s / stats.duration_s
                if stats.duration_s > 0
                else 0.0
            )
        per_request.append(request_summary)

    return {
        "requests": {
            "total": len(results),
            "succeeded": len(successes),
            "failed": len(failures),
        },
        "timing": {
            "client_wall_s": total_elapsed_s,
            "service_window_s": client_window_s,
            "aggregate_request_s": sum(latencies),
        },
        "latency_s": {
            "min": min(latencies) if latencies else 0.0,
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
        },
        "throughput": {
            "requests_per_s": len(successes) / client_window_s
            if client_window_s > 0
            else 0.0,
            "input_chars_per_s": total_input_chars / client_window_s
            if client_window_s > 0
            else 0.0,
            "bytes_per_s": total_bytes / client_window_s
            if client_window_s > 0
            else 0.0,
            "audio_s_per_wall_s": total_audio_s / client_window_s
            if client_window_s > 0
            else 0.0,
            "end_to_end_rtf": client_window_s / total_audio_s
            if total_audio_s > 0
            else 0.0,
            "parallelism_factor": sum(latencies) / client_window_s
            if client_window_s > 0
            else 0.0,
        },
        "audio": {
            "total_duration_s": total_audio_s,
            "total_samples": total_samples,
            "sample_rates": sample_rates,
            "decoded_files": sum(
                1 for stats in audio_stats.values() if stats.error is None
            ),
        },
        "bytes_received": total_bytes,
        "per_request": per_request,
    }


def _print_summary(summary: dict) -> None:
    requests = summary["requests"]
    timing = summary["timing"]
    latency = summary["latency_s"]
    throughput = summary["throughput"]
    audio = summary["audio"]

    print("\nThroughput")
    print(f"  Requests: {requests['succeeded']}/{requests['total']} succeeded")
    print(f"  Client wall time: {timing['client_wall_s']:.2f}s")
    print(f"  Service window: {timing['service_window_s']:.2f}s")
    print(f"  Requests/sec: {throughput['requests_per_s']:.2f}")
    print(f"  Input chars/sec: {throughput['input_chars_per_s']:.1f}")
    print(
        "  Response bytes/sec: "
        f"{_format_bytes_per_second(throughput['bytes_per_s'])}"
    )
    print(
        "  Audio throughput: "
        f"{throughput['audio_s_per_wall_s']:.2f}x realtime "
        f"({audio['total_duration_s']:.2f}s audio)"
    )
    print(f"  End-to-end RTF: {throughput['end_to_end_rtf']:.2f}")
    print(f"  Parallelism factor: {throughput['parallelism_factor']:.2f}x")

    print("\nLatency")
    print(
        "  min/mean/median/p95/max: "
        f"{latency['min']:.2f}s / {latency['mean']:.2f}s / "
        f"{latency['median']:.2f}s / {latency['p95']:.2f}s / "
        f"{latency['max']:.2f}s"
    )

    if audio["decoded_files"]:
        sample_rates = ", ".join(str(rate) for rate in audio["sample_rates"])
        print("\nAudio")
        print(f"  Decoded files: {audio['decoded_files']}")
        print(f"  Sample rates: {sample_rates or 'n/a'}")
        print(f"  Total samples: {audio['total_samples']}")


async def _run(args: argparse.Namespace) -> list[SpeechResult]:
    server_url = args.server_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_preload:
        print(f"Preloading {args.model} on {server_url} ...")
        await asyncio.to_thread(
            _preload_model,
            server_url,
            args.model,
            args.timeout_s,
        )

    payloads = _build_payloads(args)
    total_input_chars = sum(len(payload["input"]) for payload in payloads)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(index: int, payload: dict) -> SpeechResult:
        await asyncio.sleep(index * args.stagger_ms / 1000.0)
        async with semaphore:
            return await asyncio.to_thread(
                _request_speech,
                index=index,
                server_url=server_url,
                payload=payload,
                output_dir=output_dir,
                response_format=args.response_format,
                timeout_s=args.timeout_s,
            )

    started = time.perf_counter()
    results = await asyncio.gather(
        *(one(index, payload) for index, payload in enumerate(payloads))
    )
    total_elapsed = time.perf_counter() - started

    succeeded = sum(1 for result in results if result.error is None)
    print(
        f"Completed {succeeded}/{len(results)} requests in {total_elapsed:.2f}s "
        f"with concurrency={args.concurrency}, stagger={args.stagger_ms:g}ms."
    )

    audio_stats = _collect_audio_stats(results)
    for result in results:
        if result.error:
            print(
                f"  [{result.index:02d}] status={result.status} "
                f"elapsed={result.elapsed_s:.2f}s error={result.error}"
            )
        else:
            stats = audio_stats.get(result.index)
            audio_fields = ""
            if stats and stats.error is None:
                request_rtf = (
                    result.elapsed_s / stats.duration_s
                    if stats.duration_s > 0
                    else 0.0
                )
                audio_fields = (
                    f" audio={stats.duration_s:.2f}s"
                    f" rtf={request_rtf:.2f}"
                    f" peak={stats.peak:.3f}"
                    f" rms={stats.rms:.3f}"
                )
            elif stats and stats.error:
                audio_fields = f" audio_stats_error={stats.error}"
            print(
                f"  [{result.index:02d}] status={result.status} "
                f"elapsed={result.elapsed_s:.2f}s bytes={result.bytes_received} "
                f"file={result.output_path}{audio_fields}"
            )

    summary = _build_summary(
        results,
        audio_stats,
        total_elapsed_s=total_elapsed,
        total_input_chars=total_input_chars,
    )
    _print_summary(summary)

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote stats JSON to {stats_path}")

    print("Check the server log for '[tts-continuous]' in-flight admission lines.")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise Qwen3 TTS server-side continuous batching."
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--stagger-ms", type=float, default=300.0)
    parser.add_argument("--output-dir", default="outputs/qwen3-continuous-batching")
    parser.add_argument("--response-format", default="wav", choices=["wav", "mp3"])
    parser.add_argument("--voice", action="append", help="Voice to cycle through.")
    parser.add_argument("--text", action="append", help="Text prompt to cycle through.")
    parser.add_argument("--lang-code", default="English")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--stats-json",
        help="Optional path to write aggregate throughput stats as JSON.",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip POST /v1/models before sending speech requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests < 1:
        raise SystemExit("--requests must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
