---
title: Phonon-2
---

# Phonon-2

English speech recognition using Fermion Research's five-value Parakeet TDT model.
The [preconverted MLX weights](https://huggingface.co/mlx-community/Phonon-2)
load directly, without a converter or Fermion runtime.

```bash
pip install 'mlx-audio[stt]'
```

```python
from mlx_audio.stt import load

model = load("mlx-community/Phonon-2")
result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)
print(result.text)
```

Requires mlx-audio with Phonon-2 support. `result.sentences` contains sentences
and timestamped token pieces. `stream=True` emits overlapping chunks from a
recording; live-input streaming, translation, diarization, and hotwords are
not supported.

Weights are CC-BY-4.0; adapted code is Apache-2.0.
See the [model README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/phonon2/README.md)
for input requirements, CLI usage, and optional conversion.
