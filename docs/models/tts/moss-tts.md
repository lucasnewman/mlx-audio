# MOSS-TTS

MOSS-TTS covers OpenMOSS's 8B `MossTTSDelay` model and the 1.7B local-transformer variant. The MLX port uses the upstream Qwen3 backbone weights, RVQ generation path for each variant, and the full `OpenMOSS-Team/MOSS-Audio-Tokenizer`.

## Quick Start

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("OpenMOSS-Team/MOSS-TTS", lazy=True)

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
