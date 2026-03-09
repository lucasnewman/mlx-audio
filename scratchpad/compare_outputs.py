#!/usr/bin/env python3
"""
Compare PyTorch and MLX DeepFilterNet outputs.

Generates multiple metrics and visualizations:
1. Waveform comparison (time domain)
2. Spectrogram comparison (frequency domain)
3. RMS energy comparison
4. Correlation analysis
5. Spectral difference visualization
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Patch
from scipy import signal


def load_audio(path: str, sr: int = 48000) -> np.ndarray:
    """Load audio file at target sample rate."""
    audio, _ = librosa.load(path, sr=sr)
    return audio


def compute_stft(audio: np.ndarray, sr: int = 48000, n_fft: int = 960, hop: int = 480):
    """Compute STFT."""
    window = np.hanning(n_fft)
    f, t, spec = signal.stft(
        audio, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop
    )
    return f, t, spec


def compute_spectrogram_db(spec: np.ndarray) -> np.ndarray:
    """Convert STFT to dB scale."""
    mag = np.abs(spec)
    mag_db = 20 * np.log10(mag + 1e-10)
    return mag_db


def compute_rms_energy(
    audio: np.ndarray, window_size: int = 4800, hop: int = 2400
) -> np.ndarray:
    """Compute RMS energy over time."""
    n_windows = (len(audio) - window_size) // hop + 1
    rms = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * hop
        window = audio[start : start + window_size]
        rms[i] = np.sqrt(np.mean(window**2))
    return rms


def compute_metrics(
    orig: np.ndarray, pytorch: np.ndarray, mlx: np.ndarray, sr: int = 48000
):
    """Compute comparison metrics between original, PyTorch, and MLX outputs."""
    metrics = {}

    # 1. Overall RMS
    metrics["rms_original"] = np.sqrt(np.mean(orig**2))
    metrics["rms_pytorch"] = np.sqrt(np.mean(pytorch**2))
    metrics["rms_mlx"] = np.sqrt(np.mean(mlx**2))

    # 2. Noise reduction (estimated)
    noise_pytorch = orig[: len(pytorch)] - pytorch
    noise_mlx = orig[: len(mlx)] - mlx
    metrics["noise_rms_pytorch"] = np.sqrt(np.mean(noise_pytorch**2))
    metrics["noise_rms_mlx"] = np.sqrt(np.mean(noise_mlx**2))

    # 3. Correlation between PyTorch and MLX
    min_len = min(len(pytorch), len(mlx))
    corr = np.corrcoef(pytorch[:min_len], mlx[:min_len])[0, 1]
    metrics["correlation"] = corr

    # 4. Mean Absolute Error
    metrics["mae"] = np.mean(np.abs(pytorch[:min_len] - mlx[:min_len]))

    # 5. Mean Squared Error
    metrics["mse"] = np.mean((pytorch[:min_len] - mlx[:min_len]) ** 2)

    # 6. Max absolute difference
    metrics["max_abs_diff"] = np.max(np.abs(pytorch[:min_len] - mlx[:min_len]))

    # 7. Signal-to-error ratio
    signal_power = np.mean(pytorch[:min_len] ** 2)
    error_power = np.mean((pytorch[:min_len] - mlx[:min_len]) ** 2)
    metrics["ser_db"] = 10 * np.log10(signal_power / (error_power + 1e-10))

    # 8. Spectral convergence
    _, _, spec_pytorch = compute_stft(pytorch[:min_len])
    _, _, spec_mlx = compute_stft(mlx[:min_len])
    spec_diff = np.abs(spec_pytorch - spec_mlx[:, :min_len])
    metrics["spectral_mae"] = np.mean(spec_diff)
    metrics["spectral_max_diff"] = np.max(spec_diff)

    return metrics


def plot_waveform_comparison(
    orig: np.ndarray, pytorch: np.ndarray, mlx: np.ndarray, sr: int, output_path: str
):
    """Plot waveform comparison."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))

    t = np.arange(len(orig)) / sr

    # Plot first 1 second
    samples = sr
    axes[0].plot(t[:samples], orig[:samples], "b-", alpha=0.7, label="Original")
    axes[0].plot(t[:samples], pytorch[:samples], "r-", alpha=0.5, label="PyTorch")
    axes[0].plot(t[:samples], mlx[:samples], "g-", alpha=0.5, label="MLX")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform Comparison (First 1 second)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Full audio
    axes[1].plot(t, orig, "b-", alpha=0.7, label="Original")
    axes[1].plot(t, pytorch, "r-", alpha=0.5, label="PyTorch")
    axes[1].plot(t, mlx, "g-", alpha=0.5, label="MLX")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Waveform Comparison (Full Audio)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Difference signal
    min_len = min(len(pytorch), len(mlx))
    diff = pytorch[:min_len] - mlx[:min_len]
    axes[2].plot(t[:min_len], diff, "purple", alpha=0.8)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Difference")
    axes[2].set_title("PyTorch - MLX Difference")
    axes[2].grid(True, alpha=0.3)

    # Add zero line
    axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved waveform comparison: {output_path}")


def plot_spectrogram_comparison(
    orig: np.ndarray, pytorch: np.ndarray, mlx: np.ndarray, sr: int, output_path: str
):
    """Plot spectrogram comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    vmin, vmax = -80, 0

    # Original
    f, t, spec = compute_stft(orig)
    spec_db = compute_spectrogram_db(spec)
    im0 = axes[0, 0].pcolormesh(
        t, f, spec_db, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[0, 0].set_title("Original")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Frequency (Hz)")
    plt.colorbar(im0, ax=axes[0, 0], label="dB")

    # PyTorch
    _, _, spec_pt = compute_stft(pytorch)
    spec_db_pt = compute_spectrogram_db(spec_pt)
    im1 = axes[0, 1].pcolormesh(
        t, f, spec_db_pt, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[0, 1].set_title("PyTorch Enhanced")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im1, ax=axes[0, 1], label="dB")

    # MLX
    _, _, spec_mlx = compute_stft(mlx)
    spec_db_mlx = compute_spectrogram_db(spec_mlx)
    im2 = axes[0, 2].pcolormesh(
        t, f, spec_db_mlx, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis"
    )
    axes[0, 2].set_title("MLX Enhanced")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Frequency (Hz)")
    plt.colorbar(im2, ax=axes[0, 2], label="dB")

    # Difference PyTorch - MLX
    min_len = min(spec_db_pt.shape[1], spec_db_mlx.shape[1])
    spec_diff = spec_db_pt[:, :min_len] - spec_db_mlx[:, :min_len]
    im3 = axes[1, 0].pcolormesh(
        t[:min_len], f, spec_diff, shading="gouraud", vmin=-20, vmax=20, cmap="RdBu_r"
    )
    axes[1, 0].set_title("PyTorch - MLX Difference (dB)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    plt.colorbar(im3, ax=axes[1, 0], label="dB diff")

    # Reduction (Original - Enhanced) - PyTorch
    reduction_pt = spec_db[:, :min_len] - spec_db_pt[:, :min_len]
    im4 = axes[1, 1].pcolormesh(
        t[:min_len],
        f,
        reduction_pt,
        shading="gouraud",
        vmin=-40,
        vmax=40,
        cmap="coolwarm",
    )
    axes[1, 1].set_title("Noise Reduction (PyTorch)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im4, ax=axes[1, 1], label="dB reduction")

    # Reduction (Original - Enhanced) - MLX
    reduction_mlx = spec_db[:, :min_len] - spec_db_mlx[:, :min_len]
    im5 = axes[1, 2].pcolormesh(
        t[:min_len],
        f,
        reduction_mlx,
        shading="gouraud",
        vmin=-40,
        vmax=40,
        cmap="coolwarm",
    )
    axes[1, 2].set_title("Noise Reduction (MLX)")
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Frequency (Hz)")
    plt.colorbar(im5, ax=axes[1, 2], label="dB reduction")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved spectrogram comparison: {output_path}")


def plot_rms_comparison(
    orig: np.ndarray, pytorch: np.ndarray, mlx: np.ndarray, sr: int, output_path: str
):
    """Plot RMS energy comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))

    # Compute RMS
    rms_orig = compute_rms_energy(orig)
    rms_pytorch = compute_rms_energy(pytorch)
    rms_mlx = compute_rms_energy(mlx)

    # Time axis for RMS
    hop = 2400
    t_rms = np.arange(len(rms_orig)) * hop / sr

    axes[0].plot(t_rms, rms_orig, "b-", label="Original", alpha=0.8)
    axes[0].plot(t_rms, rms_pytorch[: len(rms_orig)], "r-", label="PyTorch", alpha=0.8)
    axes[0].plot(t_rms, rms_mlx[: len(rms_orig)], "g-", label="MLX", alpha=0.8)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("RMS Energy")
    axes[0].set_title("RMS Energy Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMS Difference
    min_len = min(len(rms_pytorch), len(rms_mlx), len(rms_orig))
    rms_diff = rms_pytorch[:min_len] - rms_mlx[:min_len]
    axes[1].plot(t_rms[:min_len], rms_diff, "purple", label="PyTorch - MLX")
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RMS Difference")
    axes[1].set_title("RMS Difference (PyTorch - MLX)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved RMS comparison: {output_path}")


def plot_correlation_scatter(pytorch: np.ndarray, mlx: np.ndarray, output_path: str):
    """Plot scatter plot showing correlation between PyTorch and MLX."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    min_len = min(len(pytorch), len(mlx))
    pt_samples = pytorch[:min_len]
    mlx_samples = mlx[:min_len]

    # Downsample for plotting
    step = max(1, min_len // 10000)
    pt_plot = pt_samples[::step]
    mlx_plot = mlx_samples[::step]

    # Scatter plot
    axes[0].scatter(pt_plot, mlx_plot, alpha=0.1, s=1)
    axes[0].plot([-1, 1], [-1, 1], "r--", label="y=x")
    axes[0].set_xlabel("PyTorch Output")
    axes[0].set_ylabel("MLX Output")
    axes[0].set_title("Sample-by-Sample Correlation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")

    # Histogram of differences
    diff = pt_samples - mlx_samples
    axes[1].hist(diff, bins=100, density=True, alpha=0.7, color="purple")
    axes[1].axvline(x=0, color="r", linestyle="--", label="Zero")
    axes[1].axvline(
        x=np.mean(diff), color="g", linestyle="-", label=f"Mean: {np.mean(diff):.6f}"
    )
    axes[1].set_xlabel("Difference (PyTorch - MLX)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution of Differences")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved correlation scatter: {output_path}")


def plot_noise_reduction_overlay(
    orig: np.ndarray, pytorch: np.ndarray, mlx: np.ndarray, sr: int, output_path: str
):
    """Plot overlay of PyTorch and MLX noise-reduction maps plus their difference."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # STFT and dB spectrograms
    f, t, spec_orig = compute_stft(orig, sr=sr)
    _, _, spec_pt = compute_stft(pytorch, sr=sr)
    _, _, spec_mlx = compute_stft(mlx, sr=sr)
    spec_db_orig = compute_spectrogram_db(spec_orig)
    spec_db_pt = compute_spectrogram_db(spec_pt)
    spec_db_mlx = compute_spectrogram_db(spec_mlx)

    min_len = min(spec_db_orig.shape[1], spec_db_pt.shape[1], spec_db_mlx.shape[1])
    t_plot = t[:min_len]
    reduction_pt = spec_db_orig[:, :min_len] - spec_db_pt[:, :min_len]
    reduction_mlx = spec_db_orig[:, :min_len] - spec_db_mlx[:, :min_len]

    # Overlay map: red=PyTorch, blue=MLX
    vmin, vmax = -40, 40
    axes[0].pcolormesh(
        t_plot,
        f,
        reduction_pt,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap="Reds",
        alpha=0.45,
    )
    axes[0].pcolormesh(
        t_plot,
        f,
        reduction_mlx,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        cmap="Blues",
        alpha=0.45,
    )
    axes[0].set_title("Noise Reduction Overlay (PyTorch vs MLX)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].legend(
        handles=[
            Patch(facecolor="red", alpha=0.45, label="PyTorch"),
            Patch(facecolor="blue", alpha=0.45, label="MLX"),
        ],
        loc="upper right",
    )

    # Difference map in dB reduction space
    reduction_diff = reduction_pt - reduction_mlx
    im = axes[1].pcolormesh(
        t_plot, f, reduction_diff, shading="gouraud", vmin=-20, vmax=20, cmap="RdBu_r"
    )
    axes[1].set_title("Noise Reduction Difference (PyTorch - MLX)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=axes[1], label="dB reduction diff")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved noise reduction overlay: {output_path}")


def plot_noise_reduction_absdiff(
    orig: np.ndarray, pytorch: np.ndarray, mlx: np.ndarray, sr: int, output_path: str
):
    """Plot absolute difference of noise-reduction maps (black means perfect match)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    f, t, spec_orig = compute_stft(orig, sr=sr)
    _, _, spec_pt = compute_stft(pytorch, sr=sr)
    _, _, spec_mlx = compute_stft(mlx, sr=sr)
    spec_db_orig = compute_spectrogram_db(spec_orig)
    spec_db_pt = compute_spectrogram_db(spec_pt)
    spec_db_mlx = compute_spectrogram_db(spec_mlx)

    min_len = min(spec_db_orig.shape[1], spec_db_pt.shape[1], spec_db_mlx.shape[1])
    t_plot = t[:min_len]
    reduction_pt = spec_db_orig[:, :min_len] - spec_db_pt[:, :min_len]
    reduction_mlx = spec_db_orig[:, :min_len] - spec_db_mlx[:, :min_len]

    # Absolute diff: 0 dB -> black, larger differences -> brighter.
    abs_diff = np.abs(reduction_pt - reduction_mlx)
    vmax = np.percentile(abs_diff, 99.5)
    if vmax <= 0:
        vmax = 1.0

    im = ax.pcolormesh(
        t_plot, f, abs_diff, shading="gouraud", cmap="gray", vmin=0.0, vmax=vmax
    )
    ax.set_title("Noise Reduction Absolute Difference (Black = Match)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax, label="|PyTorch - MLX| dB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved noise reduction abs diff: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and MLX DeepFilterNet outputs"
    )
    parser.add_argument("original", help="Original audio file")
    parser.add_argument("pytorch", help="PyTorch processed audio file")
    parser.add_argument("mlx", help="MLX processed audio file")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="comparison_output",
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DeepFilterNet Comparison: PyTorch vs MLX")
    print("=" * 60)

    # Load audio
    print("\nLoading audio files...")
    orig = load_audio(args.original)
    pytorch = load_audio(args.pytorch)
    mlx = load_audio(args.mlx)

    print(f"Original: {len(orig)/48000:.2f}s")
    print(f"PyTorch: {len(pytorch)/48000:.2f}s")
    print(f"MLX:     {len(mlx)/48000:.2f}s")

    # Compute metrics
    print("\n" + "-" * 40)
    print("Computing Metrics...")
    print("-" * 40)
    metrics = compute_metrics(orig, pytorch, mlx)

    # Print results
    print("\n" + "=" * 40)
    print("METRICS SUMMARY")
    print("=" * 40)

    print("\n1. RMS Energy:")
    print(f"   Original:     {metrics['rms_original']:.6f}")
    print(f"   PyTorch:      {metrics['rms_pytorch']:.6f}")
    print(f"   MLX:          {metrics['rms_mlx']:.6f}")

    print("\n2. Noise RMS (Original - Enhanced):")
    print(f"   PyTorch:      {metrics['noise_rms_pytorch']:.6f}")
    print(f"   MLX:          {metrics['noise_rms_mlx']:.6f}")

    print("\n3. PyTorch vs MLX Correlation:")
    print(f"   Correlation:  {metrics['correlation']:.6f}")

    print("\n4. Error Metrics (PyTorch vs MLX):")
    print(f"   MAE:          {metrics['mae']:.6e}")
    print(f"   MSE:          {metrics['mse']:.6e}")
    print(f"   RMSE:         {np.sqrt(metrics['mse']):.6e}")
    print(f"   Max Abs Diff: {metrics['max_abs_diff']:.6f}")

    print("\n5. Signal-to-Error Ratio:")
    print(f"   SER (dB):     {metrics['ser_db']:.2f} dB")

    print("\n6. Spectral Metrics:")
    print(f"   Spectral MAE: {metrics['spectral_mae']:.6e}")
    print(f"   Max Diff:     {metrics['spectral_max_diff']:.6e}")

    # Interpretation
    print("\n" + "=" * 40)
    print("INTERPRETATION")
    print("=" * 40)

    corr = metrics["correlation"]
    mae = metrics["mae"]

    if corr > 0.99:
        quality = "EXCELLENT"
    elif corr > 0.95:
        quality = "VERY GOOD"
    elif corr > 0.90:
        quality = "GOOD"
    elif corr > 0.80:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS INVESTIGATION"

    print(f"\nCorrelation Quality: {quality}")
    print(f"The PyTorch and MLX outputs are {100*corr:.2f}% similar.")

    if metrics["ser_db"] > 30:
        print(
            f"Signal-to-Error Ratio ({metrics['ser_db']:.1f} dB) indicates high fidelity."
        )
    elif metrics["ser_db"] > 20:
        print(
            f"Signal-to-Error Ratio ({metrics['ser_db']:.1f} dB) indicates good fidelity."
        )
    else:
        print(
            f"Signal-to-Error Ratio ({metrics['ser_db']:.1f} dB) indicates some differences."
        )

    # Generate plots
    print("\n" + "-" * 40)
    print("Generating Visualizations...")
    print("-" * 40)

    plot_waveform_comparison(
        orig, pytorch, mlx, 48000, output_dir / "waveform_comparison.png"
    )
    plot_spectrogram_comparison(
        orig, pytorch, mlx, 48000, output_dir / "spectrogram_comparison.png"
    )
    plot_rms_comparison(orig, pytorch, mlx, 48000, output_dir / "rms_comparison.png")
    plot_correlation_scatter(pytorch, mlx, output_dir / "correlation_scatter.png")
    plot_noise_reduction_overlay(
        orig, pytorch, mlx, 48000, output_dir / "noise_reduction_overlay.png"
    )
    plot_noise_reduction_absdiff(
        orig, pytorch, mlx, 48000, output_dir / "noise_reduction_absdiff.png"
    )

    print("\n" + "=" * 60)
    print(f"Done! Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
