# Audio8 TTS (arktts)

Multilingual zero-shot voice cloning across 11 languages, with a bundled 44.1 kHz neural codec.
Based on [Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b).

## Supported Models

- `mlx-community/Audio8-TTS-Preview-0.6b-bf16`

## Usage

Python API:

```python
import soundfile as sf
from mlx_audio.tts.utils import load

model = load("mlx-community/Audio8-TTS-Preview-0.6b-bf16")

# Zero-shot cloning. The reference transcript must match what the clip says —
# the model conditions on the (audio, text) pair, not on the audio alone.
for result in model.generate(
    text="Welcome to Audio8 TTS, running on Apple Silicon.",
    ref_audio="reference.wav",
    ref_text="Transcript of the reference clip.",
):
    sf.write("output.wav", result.audio, result.sample_rate)
```

Without `ref_audio` the model synthesizes with its own default voice.

CLI (default voice — the CLI has no flags for a reference clip, so cloning is Python-API only):

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/Audio8-TTS-Preview-0.6b-bf16 \
  --text "Welcome to Audio8 TTS, running on Apple Silicon."
```

## Languages

Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese, Korean, Polish, Spanish.

Language coverage is intentionally limited in this preview release; results are best within the
list above.

## Options

| | default | |
|---|---|---|
| `temperature` | 0.7 | |
| `top_p` | 0.9 | |
| `top_k` | 50 | |
| `max_tokens` | 512 | frames; one frame ≈ 46 ms of audio, so 512 ≈ 23.8 s |
| `do_sample` | `True` | set `False` for deterministic argmax decoding |
| `seed` | — | reproducible sampling |

The upstream demo Space runs hotter (`0.8 / 0.95 / 1024`) than the model card's defaults, which
are used here.

## Architecture

DualAR, in the style of Fish Audio S2 Pro:

- **Slow AR** — 24 layers, width 896, 14 heads / 2 KV heads. Emits one semantic token per audio
  frame.
- **Fast AR** — 4 layers, same width. Emits that frame's 10 residual codec codebooks, conditioned
  on the slow hidden state and the preceding codebooks.
- **Codec** — 44.1 kHz, 2048 samples per frame (~21.5 frames/s). DAC-style encoder/decoder with a
  split semantic + residual RVQ and windowed transformer pre/post modules. Handles both
  reference-audio encoding and waveform decoding, so no separate codec checkpoint is needed.

Sampling reproduces the reference implementation exactly: semantic-range logit filtering, the
legacy top-k/top-p order (filter *before* temperature), exponential-race sampling, and RAS
repetition rescue.

## Conversion notes

The upstream repository ships its codec as a PyTorch `.pth` with unfused weight norm. The
converted `mlx-community` weights are **pre-sanitized** — weight norm folded, conv weights in
MLX's channels-last layout, Snake alphas reshaped — so `sanitize()` passes them through
unchanged. It stays able to consume a raw upstream checkpoint as well.

## Parity

Verified against the PyTorch reference on CPU in fp32:

| | |
|---|---|
| unit / block outputs | max-abs 1e-6 … 1e-4 |
| reference-audio codec encode | **100% code-exact** |
| greedy generation | **100% token-exact** over a 102-frame utterance |
| decoded waveform | max-abs 7.5e-6 |
