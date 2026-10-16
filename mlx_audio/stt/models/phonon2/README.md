# Phonon-2

English speech recognition with [Fermion Research's Phonon-2](https://huggingface.co/FermionResearch/Phonon-2),
a five-value derivative of NVIDIA Parakeet TDT. The
[preconverted MLX checkpoint](https://huggingface.co/mlx-community/Phonon-2)
loads directly; no conversion or Fermion runtime is needed. Requires mlx-audio
with Phonon-2 support.

```python
from mlx_audio.stt import load

model = load("mlx-community/Phonon-2")
result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)
print(result.text)
```

`result.sentences` contains timestamped token pieces, which may need joining
into words. Inputs can be file paths or mono 16 kHz MLX arrays. File inputs
are resampled automatically; short nonempty inputs are padded to 320 samples.
Empty inputs are rejected.

## CLI and streaming

```bash
mlx_audio.stt.generate \
  --model mlx-community/Phonon-2 --audio audio.wav --output-path transcript
```

`stream=True` or CLI `--stream` emits overlapping chunks from a recording.
Live-input streaming, translation, diarization, and hotword biasing are not supported.

## Optional conversion

To reproduce the MLX checkpoint from the official weights:

```bash
python -m mlx_audio.stt.models.phonon2.convert \
  --hf-path FermionResearch/Phonon-2 --mlx-path ./phonon2-mlx
```

The converter verifies the source archive and extracted files, preserves the
five-value codes and fp16 row palettes, and writes dense tables in bf16.
`--hf-path` also accepts local snapshots or unpacked official directories.
Use `--revision` to pin a remote source and `--force-download` to refresh it.
Output directories must be empty or absent. Load the result with
`load("./phonon2-mlx")`. Generic conversion is disabled to protect packed weights.

## Attribution

Parsing, weight mapping, and packing are adapted from fermion-research 0.2.11
(Copyright 2026 Fermion Research, Apache-2.0). License and notices are packaged
in `mlx_audio/licenses/`. Weights derive from NVIDIA Parakeet TDT 0.6B v3
and are CC-BY-4.0. Preserve the included notices when redistributing.
