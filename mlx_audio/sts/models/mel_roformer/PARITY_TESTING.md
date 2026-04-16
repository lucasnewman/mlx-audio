# Parity Testing Setup (M5 Pro 24GB)

This guide walks through setting up the parity verification test for the MLX
Mel-Band-RoFormer port on a second machine (M5 Pro 24GB Apple Silicon).

**Why a second machine?** The MLX 228M-parameter model fits comfortably on the
M5 Pro 24GB but running the PyTorch reference in parallel (for SDR comparison)
doubles memory. Doing it on a dedicated machine keeps your primary workstation
free and avoids memory contention with other work.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11 or 3.12 (the repo currently tests on 3.12)
- ~5 GB disk for the model weights + reference audio
- ~15 minutes for first-time setup, ~2 minutes per parity run afterward

## Step 1: Clone the fork

```bash
cd ~/src  # or wherever you keep checkouts
git clone https://github.com/<your-github-handle>/mlx-audio.git
cd mlx-audio
git checkout feat/mel-band-roformer
```

If you need to pull latest changes from your main machine:

```bash
git pull origin feat/mel-band-roformer
```

## Step 2: Create a venv and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

# MLX and mlx-audio (editable install)
pip install -e ".[dev]"

# PyTorch CPU build (we only need it for the reference inference, not training)
pip install torch torchaudio

# Audio loading + resampling
pip install soundfile librosa

# ZFTurbo's training framework — required for the PyTorch reference inference
pip install omegaconf einops rotary-embedding-torch
```

## Step 3: Download a checkpoint

Choose based on what you want to verify. For parity testing, any checkpoint
works — quality doesn't matter, we're testing implementation correctness.

**Option A — ZFTurbo release asset (MIT, cleanest license):**
```bash
mkdir -p ~/mel_roformer_test
cd ~/mel_roformer_test

# Browse releases at:
#   https://github.com/ZFTurbo/Music-Source-Separation-Training/releases
# Download a Mel-Band-RoFormer .ckpt and place in this dir.

# Example (replace URL with a real release asset):
# curl -LO "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/..."
```

**Option B — Kim Vocal 2 (GPL-3.0; for local testing only):**
```bash
cd ~/mel_roformer_test
# Download from HuggingFace via the hub CLI or manual click:
#   https://huggingface.co/KimberleyJSN/melbandroformer
# Save as MelBandRoformer.ckpt in this directory.
```

## Step 4: Convert to MLX

Use the preset matching your checkpoint family:

```bash
cd ~/src/mlx-audio

python -m mlx_audio.sts.models.mel_roformer.convert \
    --input ~/mel_roformer_test/MelBandRoformer.ckpt \
    --output ~/mel_roformer_test/ \
    --preset kim_vocal_2
```

The script will print a license warning if it detects a known source, hash
the input, and write:
- `~/mel_roformer_test/MelBandRoformer.<hash>.safetensors`
- `~/mel_roformer_test/MelBandRoformer.<hash>.config.json`

Re-running the same command is free (idempotent — skips if the output exists).

## Step 5: Get a reference audio file

Any stereo 44.1kHz WAV, 30-60 seconds. Mixed music works best for vocal
separation tests.

```bash
# If you have one:
cp ~/Music/reference.wav ~/mel_roformer_test/ref.wav

# Or convert from whatever:
ffmpeg -i some_song.mp3 -ar 44100 -ac 2 ~/mel_roformer_test/ref.wav
```

## Step 6: Wire up the PyTorch reference

The parity test in `tests/sts/test_mel_roformer_parity.py` currently
skips `test_sdr_parity` with a message about needing the reference inference
to be wired up. This is the one piece that requires the ZFTurbo repo or a
minimal inference script.

**Minimal option — just write an inline script:**

Create `~/mel_roformer_test/torch_infer.py`:

```python
# torch_infer.py — run PyTorch Mel-Band-RoFormer inference
import torch
import soundfile as sf
import numpy as np
import yaml
import sys
from pathlib import Path

sys.path.insert(0, "/path/to/Music-Source-Separation-Training")  # clone ZFTurbo

from models.mel_band_roformer import MelBandRoformer

def run(ckpt_path, config_yaml, audio_path, out_path):
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)
    model = MelBandRoformer(**cfg["model"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    audio, sr = sf.read(audio_path, always_2d=True)
    audio = audio.T.astype("float32")  # [2, samples]
    audio_t = torch.from_numpy(audio[None, :, :])
    with torch.no_grad():
        vocals = model(audio_t).squeeze(0).numpy()
    sf.write(out_path, vocals.T, sr)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    run(*sys.argv[1:])
```

Clone ZFTurbo repo:
```bash
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git ~/src/MSS-Training
```

Run the reference:
```bash
python ~/mel_roformer_test/torch_infer.py \
    ~/mel_roformer_test/MelBandRoformer.ckpt \
    ~/src/MSS-Training/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml \
    ~/mel_roformer_test/ref.wav \
    ~/mel_roformer_test/ref_torch_vocals.wav
```

## Step 7: Run the parity test

```bash
export MEL_ROFORMER_TORCH_CKPT=~/mel_roformer_test/MelBandRoformer.ckpt
export MEL_ROFORMER_MLX_WEIGHTS=~/mel_roformer_test/MelBandRoformer.<hash>.safetensors
export MEL_ROFORMER_REF_AUDIO=~/mel_roformer_test/ref.wav
export MEL_ROFORMER_PRESET=kim_vocal_2
export MEL_ROFORMER_SDR_TARGET=40.0

cd ~/src/mlx-audio
pytest tests/sts/test_mel_roformer_parity.py -v -m requires_torch
```

The currently-scaffolded test `test_qkv_split_preserves_weights` will run and
verify the QKV split shape invariants. The full `test_sdr_parity` is marked
`pytest.skip` inside until the reference inference is wired in — replace that
skip with the actual comparison once your `torch_infer.py` is in place.

## Expected outcome

- **SDR > 40 dB** between PyTorch and MLX outputs: implementation parity
  confirmed. Safe to submit the PR.
- **SDR 25-40 dB**: a minor numerical issue exists (likely accumulation order
  or dtype). May be acceptable but investigate before submitting.
- **SDR < 25 dB**: a real bug. Most likely QKV head ordering, RoPE frequency
  direction, or a mel-band-gather indexing mistake.

## Debugging tips

- If `test_qkv_split_preserves_weights` fails: `convert.py` isn't splitting
  correctly. Check the packed weight shape — should be `(3 * inner_dim, dim)`.
- If SDR is very low on vocals but plausible on instrumental: the complex
  multiply in the forward pass may have real/imag swapped.
- If SDR is bimodal (high for some frames, low for others): the overlap-add
  `window_sum` normalization in `istft` may be off.
- If output is silent: likely a shape broadcasting issue — check intermediate
  tensors at each pipeline stage.

## Syncing results back

Commits go on the `feat/mel-band-roformer` branch. Push from the M5 Pro after
the test passes:

```bash
git add tests/sts/test_mel_roformer_parity.py  # if you edited it
git commit -m "tests: wire up PyTorch parity test, SDR >40 dB verified"
git push origin feat/mel-band-roformer
```

Your main machine can then pull:
```bash
cd /path/to/mlx-audio
git pull origin feat/mel-band-roformer
```
