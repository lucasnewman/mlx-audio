# SOV-Transcribe-1

[Salesforce SOV-Transcribe-1](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1)
combines a Whisper large-v3 audio encoder with a Gemma 4 mixture-of-experts
decoder. It has 26.6B total parameters and 4.6B active parameters.

| Checkpoint | Download size |
| --- | --- |
| [salesforce-sovereign-intelligence/SOV-Transcribe-1 (official BF16)](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1) | 53.3 GB |
| [nativ-community/SOV-Transcribe-1-8bit](https://huggingface.co/nativ-community/SOV-Transcribe-1-8bit) | 28.2 GB |
| [nativ-community/SOV-Transcribe-1-4bit](https://huggingface.co/nativ-community/SOV-Transcribe-1-4bit) | 15.6 GB |

The quantized checkpoints preserve the audio encoder and projector in BF16 and the MoE
routers in FP32. The original BF16 checkpoint also loads directly.

Supported languages: English, French, German, Spanish, Italian, Portuguese, and
Dutch. Input can be a file or a mono waveform sampled at 16 kHz, up to 30 minutes.
File inputs are automatically resampled. Transcription runs on the full recording,
using 30-second encoder windows and a shared decoder context.
Set `audio_batch_size` to encode multiple windows together (default 1).

```python
from mlx_audio.stt import load

model = load("nativ-community/SOV-Transcribe-1-4bit")
result = model.generate("audio.wav")
print(result.text)

result = model.generate(
    "meeting.wav",
    prompt="Transcribe in written form with punctuation and capitalization.",
    hotwords=["Salesforce", "MLX"],
    max_tokens=8192,
)
print(result.text)
```

Text can be returned incrementally after audio encoding:

```python
for chunk in model.generate("audio.wav", stream=True):
    print(chunk.text, end="", flush=True)
```

```bash
mlx_audio.stt.generate \
  --model nativ-community/SOV-Transcribe-1-4bit \
  --audio audio.wav \
  --output-path transcript \
  --prompt "Transcribe the audio verbatim." \
  --verbose
```

The result includes token counts, timing, and `finish_reason` (`stop` or `length`).
Increase `max_tokens` if generation reaches its limit. The original weights
occupy approximately 53 GB on disk. The port supports offline transcription with
instructions, language hints, and contextual vocabulary.

The model and Salesforce reference code are licensed under
[Apache 2.0](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1#license).
The MLX Gemma 4 implementation is adapted from
[mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
under the [MIT license](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE).
