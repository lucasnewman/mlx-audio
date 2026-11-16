def model_card(source: str, repo: str) -> str:
    from pathlib import Path

    if Path(source).exists():
        source = "FluidVoice-Labs/fluid-speech-pro-mlx"
    return f"""---
license: cc-by-4.0
library_name: mlx-audio
pipeline_tag: automatic-speech-recognition
base_model: FluidVoice-Labs/fluid-speech-pro
tags:
- mlx
- mlx-audio
- automatic-speech-recognition
---

# Fluid Speech Pro by FluidVoice

MLX Audio checkpoint derived from [{source}](https://huggingface.co/{source}).
Includes the frame-cleaning block and supports the optional clean-mode adapter.
The FastConformer-TDT model transcribes 16 kHz mono speech in 25 European languages
with punctuation and token/sentence timestamps, using greedy decoding.

```python
from mlx_audio.stt import load

model = load("{repo}")
print(model.generate("speech.wav").text)
print(model.generate("speech.wav", clean_mode=True).text)
```

```bash
python -m mlx_audio.stt.generate --model {repo} --audio speech.wav
```

For long audio, use `chunk_duration=30, overlap_duration=2`. Progressive output
with `stream=True` transcribes overlapping offline chunks and uses future context.
Clean mode removes fillers and repetitions, so its output is edited text.

Model and upstream reference code: [CC BY 4.0](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/blob/main/LICENSE).
Credit **Fluid Speech Pro by FluidVoice**. Based on NVIDIA Parakeet TDT 0.6B v3
and Moondream Parakeet Ultra. See the [upstream model card](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro)
for language coverage, evaluation, and limitations.
"""
