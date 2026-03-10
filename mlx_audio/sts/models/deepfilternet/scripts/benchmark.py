#!/usr/bin/env python3
"""
Benchmark DeepFilterNet inference: PyTorch vs MLX.

Compares three inference paths (PyTorch reference script, installed deepFilter
CLI, and MLX CLI) on the same audio file, averaging over multiple runs.
Outputs CSV/JSON results and a bar chart of elapsed time and real-time factor.

Usage:
  python benchmark.py input.wav
  python benchmark.py input.wav -n 5 -o results/
  python benchmark.py input.wav --skip-pytorch --skip-deepfilter
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Resolve paths relative to this script's location.
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent  # mlx_audio/sts/models/deepfilternet/
REPO_ROOT = SCRIPT_DIR.parents[
    4
]  # mlx_audio -> sts -> models -> deepfilternet -> scripts

DEFAULT_INPUT = REPO_ROOT / "examples" / "denoise" / "noisey_audio_10s.wav"
PYTORCH_SCRIPT = SCRIPT_DIR / "deep_filter_pytorch.py"
MLX_EXAMPLE = REPO_ROOT / "examples" / "deepfilternet.py"


def run_cmd(cmd: list[str], cwd: Path) -> float:
    """Run a command and return elapsed wall-clock seconds."""
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=cwd, capture_output=True, text=True)
    return time.perf_counter() - t0


def prepare_input(src: Path, out_dir: Path) -> tuple[Path, float]:
    """Resample input to 48 kHz mono and return (prepared_path, duration_s)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = out_dir / "bench_input_48k_mono.wav"
    y, _ = librosa.load(str(src), sr=48000, mono=True)
    sf.write(str(prep), y, 48000)
    return prep, len(y) / 48000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark DeepFilterNet inference: PyTorch vs MLX"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Input audio file (default: examples/denoise/noisey_audio_10s.wav)",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=3,
        help="Number of runs per method (default: 3)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for results (default: outputs/benchmarks/ in repo root)",
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip the PyTorch reference script benchmark",
    )
    parser.add_argument(
        "--skip-deepfilter",
        action="store_true",
        help="Skip the installed deepFilter CLI benchmark",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        sys.exit(1)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "outputs" / "benchmarks"
    )
    runs = args.runs
    python = sys.executable

    prep, duration_s = prepare_input(src, out_dir)
    print(f"Input: {src} ({duration_s:.1f}s @ 48 kHz)")
    print(f"Runs per method: {runs}\n")

    configs: list[tuple[str, list[str]]] = []

    # PyTorch reference script
    if not args.skip_pytorch:
        if PYTORCH_SCRIPT.exists():
            configs.append(
                (
                    "PyTorch Script",
                    [python, str(PYTORCH_SCRIPT), str(prep), "-o", ""],
                )
            )
        else:
            print(f"Warning: PyTorch script not found at {PYTORCH_SCRIPT}, skipping")

    # Installed deepFilter CLI
    if not args.skip_deepfilter:
        deepfilter_cli = shutil.which("deepFilter")
        if deepfilter_cli:
            configs.append(
                (
                    "Installed deepFilter CLI",
                    [
                        deepfilter_cli,
                        str(prep),
                        "-o",
                        "",
                        "--no-suffix",
                        "--log-level",
                        "error",
                        "--model-base-dir",
                        "DeepFilterNet3",
                    ],
                )
            )
        else:
            print("Warning: deepFilter CLI not found in PATH, skipping")

    # MLX CLI
    if MLX_EXAMPLE.exists():
        configs.append(
            (
                "MLX CLI",
                [python, str(MLX_EXAMPLE), str(prep), "-o", ""],
            )
        )
    else:
        print(f"Warning: MLX example not found at {MLX_EXAMPLE}, skipping")

    if not configs:
        print("Error: no benchmark methods available", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for name, base in configs:
        for i in range(1, runs + 1):
            out = (
                out_dir
                / f"{name.lower().replace(' ', '_').replace('/', '_')}_run{i}.wav"
            )
            cmd = base.copy()
            out_idx = cmd.index("-o") + 1
            cmd[out_idx] = str(out)
            elapsed = run_cmd(cmd, cwd=REPO_ROOT)
            rows.append(
                {
                    "method": name,
                    "run": i,
                    "elapsed_s": elapsed,
                    "rtf": elapsed / duration_s,
                    "output": str(out),
                }
            )
            print(f"{name} run {i}: {elapsed:.3f}s (RTF {elapsed / duration_s:.4f})")

    # Save results
    csv_path = out_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["method", "run", "elapsed_s", "rtf", "output"]
        )
        w.writeheader()
        w.writerows(rows)

    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Summary
    methods = sorted({r["method"] for r in rows})
    print(f"\n{'Method':<30} {'Mean (s)':>10} {'Std (s)':>10} {'Mean RTF':>10}")
    print("-" * 65)
    for m in methods:
        times = [r["elapsed_s"] for r in rows if r["method"] == m]
        rtfs = [r["rtf"] for r in rows if r["method"] == m]
        print(
            f"{m:<30} {np.mean(times):>10.3f} {np.std(times):>10.3f} {np.mean(rtfs):>10.4f}"
        )

    # Chart
    means_elapsed = [
        np.mean([r["elapsed_s"] for r in rows if r["method"] == m]) for m in methods
    ]
    std_elapsed = [
        np.std([r["elapsed_s"] for r in rows if r["method"] == m]) for m in methods
    ]
    means_rtf = [np.mean([r["rtf"] for r in rows if r["method"] == m]) for m in methods]
    std_rtf = [np.std([r["rtf"] for r in rows if r["method"] == m]) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(methods))

    axes[0].bar(x, means_elapsed, yerr=std_elapsed, capsize=6)
    axes[0].set_xticks(x, methods, rotation=15, ha="right")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title(f"Elapsed Time (n={runs}, input={duration_s:.1f}s)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, means_rtf, yerr=std_rtf, capsize=6, color="#2f7ed8")
    axes[1].set_xticks(x, methods, rotation=15, ha="right")
    axes[1].set_ylabel("RTF (elapsed / audio duration)")
    axes[1].set_title("Real-Time Factor")
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    chart_path = out_dir / "benchmark_chart.png"
    plt.savefig(chart_path, dpi=180)
    plt.close()

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
