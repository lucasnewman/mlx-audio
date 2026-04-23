#!/usr/bin/env python3
"""Exercise server-side continuous batching with Qwen3 TTS 1.7B Base.

Start the API server in one terminal:

    mlx_audio.server --host 127.0.0.1 --port 8000 \
        --tts-max-batch-size 4 \
        --inference-batch-wait-ms 25

Then run this script in another terminal:

    python examples/qwen3_tts_continuous_batching_server.py \
        --requests 4 --concurrency 4 --stagger-ms 5

Watch the server logs for lines like:

    [tts-batch] running 4 Qwen3 TTS requests for ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
DEFAULT_TEXTS = [
    "Continuous batching should group nearby requests into one model step loop.",
    "This sentence is intentionally short so the example finishes quickly.",
    "Each HTTP request still receives its own audio file from the server.",
    "The server log is the easiest way to confirm the batch size used.",
]
DEFAULT_VOICES = ["Chelsie", "Ethan"]


@dataclass
class SpeechResult:
    index: int
    status: int | None
    elapsed_s: float
    bytes_received: int
    output_path: Path | None = None
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
        detail = exc.read().decode("utf-8", errors="replace")
        return SpeechResult(
            index=index,
            status=exc.code,
            elapsed_s=time.perf_counter() - started,
            bytes_received=0,
            error=detail or str(exc),
        )
    except URLError as exc:
        return SpeechResult(
            index=index,
            status=None,
            elapsed_s=time.perf_counter() - started,
            bytes_received=0,
            error=str(exc),
        )

    output_path = output_dir / f"qwen3_batch_request_{index:02d}.{response_format}"
    output_path.write_bytes(body)
    return SpeechResult(
        index=index,
        status=status,
        elapsed_s=time.perf_counter() - started,
        bytes_received=len(body),
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
                "stream": args.stream,
                "streaming_interval": args.streaming_interval,
            }
        )

    return payloads


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

    for result in results:
        if result.error:
            print(
                f"  [{result.index:02d}] status={result.status} "
                f"elapsed={result.elapsed_s:.2f}s error={result.error}"
            )
        else:
            print(
                f"  [{result.index:02d}] status={result.status} "
                f"elapsed={result.elapsed_s:.2f}s bytes={result.bytes_received} "
                f"file={result.output_path}"
            )

    print("Check the server log for '[tts-batch]' to confirm batch size.")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise Qwen3 TTS server-side continuous batching."
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--stagger-ms", type=float, default=5.0)
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
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--streaming-interval", type=float, default=0.32)
    parser.add_argument("--timeout-s", type=float, default=900.0)
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
