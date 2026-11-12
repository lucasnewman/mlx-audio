---
title: SOV-Transcribe-1
---

# SOV-Transcribe-1

Salesforce's 26.6B parameter speech model combines Whisper large-v3 with a Gemma 4
MoE decoder (4.6B active parameters). It transcribes English, French, German,
Spanish, Italian, Portuguese, and Dutch, with natural-language instructions and
contextual vocabulary. Recordings can be up to 30 minutes long.

| MLX checkpoint | Download size |
| --- | --- |
| [salesforce-sovereign-intelligence/SOV-Transcribe-1 (official BF16)](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1) | 53.3 GB |
| [nativ-community/SOV-Transcribe-1-8bit](https://huggingface.co/nativ-community/SOV-Transcribe-1-8bit) | 28.2 GB |
| [nativ-community/SOV-Transcribe-1-4bit](https://huggingface.co/nativ-community/SOV-Transcribe-1-4bit) | 15.6 GB |

Both checkpoints preserve the audio encoder and projector in BF16 and the MoE
routers in FP32.

```python
from mlx_audio.stt import load

model = load("nativ-community/SOV-Transcribe-1-4bit")
result = model.generate(
    "audio.wav",
    prompt="Transcribe in written form with punctuation.",
    hotwords=["Salesforce", "MLX"],
)
print(result.text)
```

```bash
mlx_audio.stt.generate \
  --model nativ-community/SOV-Transcribe-1-4bit \
  --audio audio.wav --output-path transcript --verbose
```

The original BF16 checkpoint also loads directly and occupies about 53 GB on disk.
Audio files are resampled to 16 kHz;
waveform arrays must already be mono at 16 kHz. Transcription runs offline.
The returned `finish_reason` indicates whether decoding stopped normally or reached `max_tokens`.

See the [model README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/sovereignai_transcribe/README.md)
for additional usage and license information, and the
[upstream model](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1)
for model details.
