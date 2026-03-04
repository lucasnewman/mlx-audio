"""CLI for DeepFilterNet MLX denoising."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import DeepFilterNetModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Denoise audio with DeepFilterNet (MLX)")
    p.add_argument("input", help="Input noisy audio file (48kHz)")
    p.add_argument("-o", "--output", help="Output denoised audio path")
    p.add_argument(
        "-m",
        "--model",
        default="3",
        choices=["1", "2", "3"],
        help="DeepFilterNet version to use (1, 2, or 3)",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Optional local model directory override",
    )
    return p


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

    version = int(args.model)
    model = DeepFilterNetModel.from_pretrained(version=version, model_dir=args.model_dir)
    model.enhance_file(in_path, out_path)

    print(f"Input : {in_path}")
    print(f"Model : DeepFilterNet{version}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
