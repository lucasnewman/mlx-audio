# Reverb TTS

Native MLX implementation of [Software Mansion's Reverb](https://huggingface.co/software-mansion/Reverb-TTS), a 297M-parameter derivative of OmniVoice. It generates mono 24 kHz audio and supports English, German, French, Spanish, Italian, Polish, and Portuguese.

The text model, audio decoder, and reference-audio encoder run in MLX. Runtime inference does not require PyTorch or ExecuTorch.

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

```bash
python -m mlx_audio.tts.generate \
  --model nativ-community/Reverb-TTS \
  --voice standard_male \
  --text "Hello from Reverb on Apple Silicon." \
  --lang_code en
```

Preset voices are `standard_male` (default), `angry_male`, and `sensual_female`. Use `model.voices` to inspect the voices in a converted bundle.

## Voice cloning

```python
result = next(model.generate(
    text="The entire voice cloning pipeline runs locally using MLX.",
    ref_audio="reference.wav",
    ref_text="The exact words spoken in reference.wav.",
    lang_code="en",
))
```

File references are downmixed and resampled to 24 kHz. Array references must already be mono 24 kHz audio. Supply an accurate reference transcript. References longer than `ref_audio_max_duration_s` (10 seconds by default) raise an error so that the audio is never silently truncated independently of its transcript. Increase that limit explicitly for longer clips. Pre-encoded `ref_tokens` with shape `[frames, 8]` are also supported with `ref_text`.

## Conversion

Use the converter to create a local MLX bundle from the original Software Mansion checkpoint.

```bash
python -m mlx_audio.tts.models.reverb.convert \
  --model software-mansion/Reverb-TTS \
  --output ./reverb-mlx \
  --dtype bfloat16

# Optional native MLX 4-bit weights (8-bit is also available):
python -m mlx_audio.tts.models.reverb.convert \
  --output ./reverb-mlx-4bit --bits 4
```

Load either directory with `mlx_audio.tts.utils.load`. The converter supports `float32`, `float16`, and `bfloat16`, preserves the codec's original precision, and includes tokenizer data, voice prompts, and upstream attribution. `--revision` pins the Reverb revision; `--codec` can point to a local Higgs Audio v2 tokenizer directory for fully local conversion.

MLX quantization is produced from the FP32 source. Upstream Torch int4/int8/int8a8 safetensors and ExecuTorch `.pte` files are not MLX checkpoints and are not used.

## Inference controls

| Python argument | Default | Meaning |
| --- | --- | --- |
| `num_steps` | 16 | Iterative unmasking passes |
| `duration_s` | automatic | Output duration estimated from the reference speaking rate when omitted |
| `gen_duration` | unset | CLI-compatible alias for `duration_s` |
| `speed` | 1.0 | Duration divisor; larger values produce faster speech |
| `class_temperature` | 0.0 | Codec-token sampling temperature |
| `position_temperature` | 5.0 | Noise used to select masked positions to reveal |
| `layer_penalty_factor` | 5.0 | Prioritization of earlier codec codebooks |
| `t_shift` | 0.1 | Unmasking schedule shift |

This port reuses MLX Audio's OmniVoice unmasking schedule with configurable sampling parameters. The checkpoint's `echo.cfg.mode="student"` indicates distilled guidance: an additional unconditional CFG pass is not applied. The linked upstream Echo-TTS source was unavailable when this port was implemented, so exact parity with its optimized sampler has not been established.

The transformer uses 24 layers, 12 query heads, 6 KV heads, a 256-dimensional factorized text embedding, and prefix-blocked attention. Text and reference KV states are cached once per generation. The target remains bidirectional and is recomputed at each unmasking step. `generate_batch()` processes requests sequentially; streaming is not supported.

## License

Reverb weights are **CC BY-NC 4.0**. Conversion retains upstream attribution in `README.upstream.md`. The codec comes from [eustlb/higgs-audio-v2-tokenizer](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer).
