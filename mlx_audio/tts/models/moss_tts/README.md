# MOSS-TTS

MOSS-TTS covers the 8B `MossTTSDelay` model and the 1.7B local-transformer variant from OpenMOSS. It uses a Qwen3 backbone with the matching multi-codebook audio generation path and the full MOSS Audio Tokenizer.

## Supported Models

- `OpenMOSS-Team/MOSS-TTS`
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer`

## Usage

Python API:

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("OpenMOSS-Team/MOSS-TTS", lazy=True)

result = next(model.generate(
    text="Hello, this is MOSS-TTS running on MLX.",
    max_tokens=120,
))
audio_write("output.wav", result.audio, result.sample_rate)
```

Voice cloning:

```python
result = next(model.generate(
    text="This is a short cloned voice sample.",
    ref_audio="speaker.wav",
    max_tokens=120,
))
audio_write("cloned.wav", result.audio, result.sample_rate)
```
