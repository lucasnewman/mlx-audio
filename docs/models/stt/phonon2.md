---
title: Phonon-2
---

# Phonon-2

Phonon-2 is Fermion Research's English-only derivative of NVIDIA Parakeet TDT
0.6B v3. Its encoder uses five learned values per row. The official download
is approximately 164 MB; runtime memory also includes MLX packing, dense
tables, and inference buffers.

```bash
pip install 'mlx-audio[stt]'
```

```python
from mlx_audio.stt import load

model = load("FermionResearch/Phonon-2")
result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)
print(result.text)
```

The first load verifies the official archive and extracted files, then caches
the converted Safetensors model under `~/.cache/mlx_audio/phonon2/`. Set
`MLX_AUDIO_CACHE_DIR` to change that location. Preparation and inference use
NumPy/MLX, without Fermion's Python runtime or native libraries. Revision pins,
forced refreshes, local snapshots, and unpacked official directories work
through the standard loader.

`result.sentences` contains sentences and timestamped token pieces. Pieces
may need joining to form words. For long audio use explicit chunking, or
`stream=True` for overlapping chunk results from a supplied recording; this
is not live-input streaming. Translation, diarization, and hotword biasing
are not supported by this integration.

The runtime adaptations are Apache-2.0; model weights are CC-BY-4.0.
See the [implementation README](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/phonon2/README.md)
for CLI usage, cache behavior, input requirements, and attribution.
