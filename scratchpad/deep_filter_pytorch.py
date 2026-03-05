#!/usr/bin/env python3
"""
Simple CLI to denoise audio using DeepFilterNet PyTorch library.

Usage:
  python deep_filter_pytorch.py input.wav -o output.wav
  python deep_filter_pytorch.py input.wav  # saves to input_enhanced.wav
"""

import argparse
import soundfile as sf
import numpy as np
from pathlib import Path

from df.enhance import init_df, df_features
from df.utils import as_complex


def denoise(input_path: str, output_path: str, model_path: str = None):
    """Denoise audio file using DeepFilterNet PyTorch."""
    # Load audio
    audio, sr = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"Loaded: {input_path} (sr={sr}, duration={len(audio)/sr:.2f}s)")
    
    # Initialize model
    if model_path:
        model, df_state, _ = init_df(model_path, log_level='ERROR')
    else:
        model, df_state, _ = init_df(log_level='ERROR')
    model.eval()
    
    # Prepare audio
    import torch
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    orig_len = audio_tensor.shape[-1]
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()
    audio_padded = torch.nn.functional.pad(audio_tensor, (0, n_fft))
    
    # Get features
    from df.model import ModelParams
    p = ModelParams()
    spec, erb_feat, spec_feat = df_features(audio_padded, df_state, p.nb_df, device='cpu')
    
    # Run model
    with torch.no_grad():
        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=audio_tensor.shape[0], device="cpu")
        enhanced, _, _, _ = model(spec, erb_feat, spec_feat)
    
    # Convert to audio
    enhanced_complex = as_complex(enhanced.squeeze(1))
    enhanced_audio = df_state.synthesis(enhanced_complex.numpy())
    enhanced_audio = torch.as_tensor(enhanced_audio)
    
    # Remove padding
    d = n_fft - hop
    enhanced_audio = enhanced_audio[:, d:orig_len + d]
    
    # Save
    sf.write(output_path, enhanced_audio[0].numpy(), sr)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Denoise audio with DeepFilterNet (PyTorch)")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", help="Output audio file (default: input_enhanced.wav)")
    parser.add_argument("-m", "--model", help="Path to DeepFilterNet model (optional)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_stem(input_path.stem + "_enhanced"))
    
    denoise(str(input_path), output_path, args.model)


if __name__ == "__main__":
    main()
