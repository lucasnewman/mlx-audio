#!/usr/bin/env python3
"""CLI to denoise audio using the official DeepFilterNet PyTorch stack.

Usage:
  python deep_filter_pytorch.py input.wav -o output.wav
  python deep_filter_pytorch.py input.wav --stream -o output_stream.wav
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from df.enhance import df_features, init_df
from df.utils import as_complex


def _enhance_array_offline(model, df_state, audio: np.ndarray) -> np.ndarray:
    """Match official offline inference behavior used in this repo."""
    import torch

    audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).float()
    orig_len = audio_tensor.shape[-1]
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()
    audio_padded = torch.nn.functional.pad(audio_tensor, (0, n_fft))

    # Get features exactly like official path.
    from df.model import ModelParams

    p = ModelParams()
    spec, erb_feat, spec_feat = df_features(
        audio_padded, df_state, p.nb_df, device="cpu"
    )

    with torch.no_grad():
        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=audio_tensor.shape[0], device="cpu")
        enhanced, _, _, _ = model(spec, erb_feat, spec_feat)

    enhanced_complex = as_complex(enhanced.squeeze(1))
    enhanced_audio = df_state.synthesis(enhanced_complex.numpy())
    enhanced_audio = np.asarray(enhanced_audio, dtype=np.float32)

    d = n_fft - hop
    enhanced_audio = enhanced_audio[:, d : orig_len + d]
    return enhanced_audio[0].astype(np.float32)


def _default_stable_tail_samples(model, df_state) -> int:
    hop = int(df_state.hop_size())
    n_fft = int(df_state.fft_size())
    df_lookahead = int(getattr(model, "df_lookahead", 0))
    df_order = int(getattr(model, "df_order", 5))
    conv_lookahead = 0
    pad_feat = getattr(model, "pad_feat", None)
    if pad_feat is not None and hasattr(pad_feat, "padding"):
        try:
            conv_lookahead = max(0, int(pad_feat.padding[-1]))
        except Exception:
            conv_lookahead = 0
    return int(n_fft + (max(conv_lookahead, df_lookahead) + df_order) * hop)


def _enhance_array_streaming_reference(
    model,
    df_state,
    audio: np.ndarray,
    chunk_samples: int,
    stable_tail_samples: int,
) -> np.ndarray:
    """Streaming-reference mode: chunked incremental emission with stable tail holdback."""
    hop = int(df_state.hop_size())
    x = np.asarray(audio, dtype=np.float32)
    buffered = np.zeros((0,), dtype=np.float32)
    emitted = 0
    outs = []

    for start in range(0, x.shape[0], chunk_samples):
        buffered = np.concatenate([buffered, x[start : start + chunk_samples]], axis=0)
        enh = _enhance_array_offline(model, df_state, buffered)
        stable_end = max(0, enh.shape[0] - stable_tail_samples)
        stable_end -= stable_end % hop
        stable_end = max(stable_end, emitted)
        if stable_end > emitted:
            outs.append(enh[emitted:stable_end])
            emitted = stable_end

    # Final flush.
    enh = _enhance_array_offline(model, df_state, buffered)
    if enh.shape[0] > emitted:
        outs.append(enh[emitted:])

    y = np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)
    if y.shape[0] < x.shape[0]:
        y = np.pad(y, (0, x.shape[0] - y.shape[0]))
    elif y.shape[0] > x.shape[0]:
        y = y[: x.shape[0]]
    return y.astype(np.float32)


def denoise(
    input_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    stream: bool = False,
    chunk_ms: float = 480.0,
    stable_tail_ms: Optional[float] = None,
):
    """Denoise audio file using DeepFilterNet PyTorch."""
    audio, sr = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"Loaded: {input_path} (sr={sr}, duration={len(audio)/sr:.2f}s)")

    if model_path:
        model, df_state, _ = init_df(model_path, log_level="ERROR")
    else:
        model, df_state, _ = init_df(log_level="ERROR")
    model.eval()

    if stream:
        chunk_samples = max(int(sr * (chunk_ms / 1000.0)), int(df_state.hop_size()))
        stable_tail_samples = (
            _default_stable_tail_samples(model, df_state)
            if stable_tail_ms is None
            else max(0, int(sr * (stable_tail_ms / 1000.0)))
        )
        print(
            f"Streaming reference mode: chunk={chunk_samples} samples, stable_tail={stable_tail_samples} samples"
        )
        enhanced = _enhance_array_streaming_reference(
            model=model,
            df_state=df_state,
            audio=audio.astype(np.float32),
            chunk_samples=chunk_samples,
            stable_tail_samples=stable_tail_samples,
        )
    else:
        enhanced = _enhance_array_offline(model, df_state, audio.astype(np.float32))

    sf.write(output_path, enhanced, sr)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Denoise audio with DeepFilterNet (PyTorch)"
    )
    parser.add_argument("input", help="Input audio file")
    parser.add_argument(
        "-o", "--output", help="Output audio file (default: input_enhanced.wav)"
    )
    parser.add_argument("-m", "--model", help="Path to DeepFilterNet model (optional)")
    parser.add_argument(
        "--stream", action="store_true", help="Run streaming-reference mode"
    )
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=480.0,
        help="Chunk size for --stream mode in milliseconds (default: 480)",
    )
    parser.add_argument(
        "--stable-tail-ms",
        type=float,
        default=None,
        help="Stable tail holdback for --stream mode in milliseconds (default: auto)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_stem(input_path.stem + "_enhanced"))

    denoise(
        str(input_path),
        output_path,
        args.model,
        stream=args.stream,
        chunk_ms=args.chunk_ms,
        stable_tail_ms=args.stable_tail_ms,
    )


if __name__ == "__main__":
    main()
