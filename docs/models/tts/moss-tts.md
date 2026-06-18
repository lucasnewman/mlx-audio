# MOSS-TTS

MOSS-TTS covers OpenMOSS's 8B `MossTTSDelay` v1.5 and v1.0 models plus the local-transformer variants. The MLX port uses the upstream Qwen3 backbone weights, RVQ generation path for each variant, and the matching MOSS Audio Tokenizer.

## Quick Start

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("OpenMOSS-Team/MOSS-TTS-v1.5", lazy=True)

result = next(model.generate(
    text="Hello, this is MOSS-TTS running on MLX.",
    max_tokens=120,
))
audio_write("moss_tts.wav", result.audio, result.sample_rate)
```

For the local-transformer variant, load:

```python
model = load("OpenMOSS-Team/MOSS-TTS-Local-Transformer", lazy=True)
```

For the v1.5 local-transformer checkpoint, load:

```python
model = load("OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5", lazy=True)
```

The v1.5 local checkpoint uses `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`, 48 kHz output, and fixed 12-codebook RVQ generation.

For non-English v1.5 prompts, pass `language` when known:

```python
result = next(model.generate(
    text="Bonjour, je voudrais essayer une voix francaise naturelle.",
    language="French",
    max_tokens=120,
))
```

The v1.5 checkpoints accept inline pause markers such as `[pause 3.2s]`.

## Voice Cloning

```python
result = next(model.generate(
    text="This is a short cloned voice sample.",
    ref_audio="speaker.wav",
    max_tokens=120,
))
audio_write("moss_tts_clone.wav", result.audio, result.sample_rate)
```

## Notes

- The model uses 24 kHz audio.
- The full audio tokenizer is required for reference encoding and waveform decoding.
- Batch generation and streaming are not implemented in the first MLX port.
