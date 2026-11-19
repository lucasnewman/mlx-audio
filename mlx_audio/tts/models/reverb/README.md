# Reverb TTS

[Software Mansion's Reverb](https://huggingface.co/software-mansion/Reverb-TTS) generates mono 24 kHz speech on Apple silicon. It supports English, German, French, Spanish, Italian, Polish, and Portuguese, with preset voices and voice cloning from a reference recording.

## Supported Models

| Hugging Face repository | Reverb weights |
| --- | --- |
| [nativ-community/Reverb-TTS](https://huggingface.co/nativ-community/Reverb-TTS) | FP32 |
| [nativ-community/Reverb-TTS-bf16](https://huggingface.co/nativ-community/Reverb-TTS-bf16) | BF16 |
| [nativ-community/Reverb-TTS-8bit](https://huggingface.co/nativ-community/Reverb-TTS-8bit) | 8-bit affine |
| [nativ-community/Reverb-TTS-4bit](https://huggingface.co/nativ-community/Reverb-TTS-4bit) | 4-bit affine |

Each bundle includes the FP32 Higgs Audio v2 codec.

## Quick start

```bash
pip install mlx-audio
```

### Python

```python
from mlx_audio.tts.utils import load
from mlx_audio.audio_io import write

model = load("nativ-community/Reverb-TTS")
result = next(model.generate(
    text="Hello from Reverb running locally on Apple Silicon.",
    voice="standard_male",
    lang_code="en",
))
write("speech.wav", result.audio, result.sample_rate)
```

### CLI

```bash
python -m mlx_audio.tts.generate \
  --model nativ-community/Reverb-TTS \
  --voice standard_male \
  --text "Hello from Reverb on Apple Silicon." \
  --lang_code en
```

Preset voices are `standard_male` (default), `angry_male`, and `sensual_female`. Use `model.voices` to list the available presets.

## Voice cloning

```python
result = next(model.generate(
    text="Welcome back. Here's an update for today.",
    ref_audio="reference.wav",
    ref_text="The exact words spoken in reference.wav.",
    lang_code="en",
))
write("cloned.wav", result.audio, result.sample_rate)
```

Reverb downmixes and resamples reference files to 24 kHz. If you pass an audio array, supply mono 24 kHz audio. Use a matching transcript and a recording up to 10 seconds long, or increase `ref_audio_max_duration_s` for longer clips. To reuse pre-encoded reference tokens, pass `ref_tokens` with shape `[frames, 8]` and the corresponding `ref_text`.

## Inference controls

| Python argument | Default | Meaning |
| --- | --- | --- |
| `lang_code` | `en` | Output language (`en`, `de`, `fr`, `es`, `it`, `pl`, `pt`) |
| `duration_s` | automatic | Output duration estimated from the reference speaking rate when omitted |
| `speed` | 1.0 | Speaking speed multiplier |
| `num_steps` | 16 | Number of generation steps |

Use `model.generate_batch()` to pass a list of texts and get one complete waveform per text.

## License

Reverb weights are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See the [upstream acknowledgments](https://huggingface.co/software-mansion/Reverb-TTS/blob/6957eceee689b1accf43e8b2a064587de8942743/README.md#license-and-acknowledgments) and the [bundled Higgs Audio v2 codec license](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/LICENSE).
