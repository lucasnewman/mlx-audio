# Mega-ASR

Mega-ASR is a routed Qwen3-ASR variant: a tiny audio-quality router decides whether an utterance is clean or degraded, then the model either stays on the dense Qwen3-ASR base path or applies the merged LoRA deltas in place.

## Convert from Hugging Face

Convert the upstream `zhifeixie/Mega-ASR` repository into an MLX Mega-ASR model directory:

```bash
uv run python -m mlx_audio.stt.models.mega_asr.convert \
  --hf-path zhifeixie/Mega-ASR \
  --mlx-path /path/to/mega-asr-mlx \
  --dtype bfloat16
```

The base Qwen3-ASR weights are converted as **dense bf16** shards on purpose. Mega-ASR adds LoRA deltas at inference time, so the base must stay dense rather than quantized.

## Python usage

```python
from mlx_audio.stt import load

model = load("/path/to/mega-asr-mlx")
result = model.generate("audio.wav", temperature=0.0, language="en")
print(result.text)
```

Routing is automatic:

- clean audio → base Qwen3-ASR path
- degraded audio → LoRA-enhanced path

You do not need to toggle adapters manually.

## CLI usage

```bash
uv run python -m mlx_audio.stt.generate \
  --model /path/to/mega-asr-mlx \
  --audio audio.wav \
  --output-path transcript \
  --verbose
```
