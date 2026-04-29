# MOSS-TTS

MOSS-TTS is the 8B `MossTTSDelay` model from OpenMOSS. It uses a Qwen3 backbone with delay-pattern multi-codebook audio generation and the full MOSS Audio Tokenizer.

## Supported Models

- `OpenMOSS-Team/MOSS-TTS`
- Audio tokenizer: `OpenMOSS-Team/MOSS-Audio-Tokenizer`

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

Current limitations:

- Batch generation is not implemented.
- Streaming is not implemented.
