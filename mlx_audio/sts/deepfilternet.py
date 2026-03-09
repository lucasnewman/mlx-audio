"""CLI entrypoint for DeepFilterNet speech enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Denoise audio with DeepFilterNet (MLX)")
    parser.add_argument("input", help="Input noisy audio file (48 kHz)")
    parser.add_argument("-o", "--output", help="Output denoised audio path")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=(
            "Model name or path. Can be a Hugging Face repo id or local model directory."
        ),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Run chunked streaming enhancement (stateful process_chunk/flush path)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=480.0,
        help="Chunk size for --stream mode in milliseconds (default: 480)",
    )
    parser.add_argument(
        "--pad-end-frames",
        type=int,
        default=3,
        help="Extra zero frames to flush streaming tail (default: 3)",
    )
    parser.add_argument(
        "--no-delay-compensation",
        action="store_true",
        help="Disable initial delay compensation crop in streaming output",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else in_path.with_stem(in_path.stem + "_enhanced_mlx")
    )

    model = DeepFilterNetModel.from_pretrained(model_name_or_path=args.model)
    if args.stream:
        sr = model.config.sample_rate
        chunk_samples = max(model.config.hop_size, int(sr * args.chunk_ms / 1000.0))
        try:
            model.enhance_file_streaming(
                in_path,
                out_path,
                chunk_samples=chunk_samples,
                pad_end_frames=max(0, args.pad_end_frames),
                compensate_delay=(not args.no_delay_compensation),
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Streaming mode is unavailable for {model.model_version}: {exc}"
            ) from exc
    else:
        model.enhance_file(in_path, out_path)

    print(f"Input : {in_path}")
    print(f"Model : {model.model_version}")
    print(f"Mode  : {'streaming' if args.stream else 'offline'}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
