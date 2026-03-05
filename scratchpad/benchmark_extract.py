#!/usr/bin/env python3
"""
Benchmark DeepFilterNet inference paths on Extract.wav and render a chart.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


ROOT = Path("/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex")
PY = ROOT / ".venv" / "bin" / "python"
DEEPFILTER_CLI = ROOT / ".venv" / "bin" / "deepFilter"

SRC = ROOT / "examples" / "denoise" / "Extract.wav"
PREP = ROOT / "outputs" / "bench_extract_48k_mono.wav"
OUT_DIR = ROOT / "outputs" / "benchmarks_extract"
RUNS = 3


def run_cmd(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
    return time.perf_counter() - t0


def prepare_input() -> float:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y, _ = librosa.load(str(SRC), sr=48000, mono=True)
    sf.write(str(PREP), y, 48000)
    return len(y) / 48000.0


def main() -> None:
    duration_s = prepare_input()

    configs = [
        (
            "PyTorch Script",
            [
                str(PY),
                "scratchpad/deep_filter_pytorch.py",
                str(PREP),
                "-o",
                "",  # replaced per run
            ],
        ),
        (
            "Installed deepFilter CLI",
            [
                str(DEEPFILTER_CLI),
                str(PREP),
                "-o",
                "",  # replaced per run
                "--no-suffix",
                "--log-level",
                "error",
                "--model-base-dir",
                "DeepFilterNet3",
            ],
        ),
        (
            "MLX CLI",
            [
                str(PY),
                "examples/deepfilternet.py",
                str(PREP),
                "-o",
                "",  # replaced per run
                "-m",
                "3",
            ],
        ),
    ]

    rows: list[dict] = []
    for name, base in configs:
        for i in range(1, RUNS + 1):
            out = OUT_DIR / f"{name.lower().replace(' ', '_').replace('/', '_')}_run{i}.wav"
            cmd = base.copy()
            out_idx = cmd.index("-o") + 1
            cmd[out_idx] = str(out)
            elapsed = run_cmd(cmd)
            rows.append(
                {
                    "method": name,
                    "run": i,
                    "elapsed_s": elapsed,
                    "rtf": elapsed / duration_s,
                    "output": str(out),
                }
            )
            print(f"{name} run {i}: {elapsed:.3f}s (RTF {elapsed/duration_s:.4f})")

    csv_path = OUT_DIR / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "run", "elapsed_s", "rtf", "output"])
        w.writeheader()
        w.writerows(rows)

    json_path = OUT_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    methods = sorted({r["method"] for r in rows})
    means_elapsed = [np.mean([r["elapsed_s"] for r in rows if r["method"] == m]) for m in methods]
    std_elapsed = [np.std([r["elapsed_s"] for r in rows if r["method"] == m]) for m in methods]
    means_rtf = [np.mean([r["rtf"] for r in rows if r["method"] == m]) for m in methods]
    std_rtf = [np.std([r["rtf"] for r in rows if r["method"] == m]) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(methods))

    axes[0].bar(x, means_elapsed, yerr=std_elapsed, capsize=6)
    axes[0].set_xticks(x, methods, rotation=15, ha="right")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title(f"Elapsed Time (n={RUNS}, input={duration_s:.1f}s)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, means_rtf, yerr=std_rtf, capsize=6, color="#2f7ed8")
    axes[1].set_xticks(x, methods, rotation=15, ha="right")
    axes[1].set_ylabel("RTF (elapsed / audio duration)")
    axes[1].set_title("Real-Time Factor")
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    chart_path = OUT_DIR / "benchmark_chart.png"
    plt.savefig(chart_path, dpi=180)
    plt.close()

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
