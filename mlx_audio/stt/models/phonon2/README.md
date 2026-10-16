# Phonon-2

English speech recognition using Fermion Research's five-value encoder and
the existing Parakeet TDT graph. The official model download is approximately
164 MB. Runtime memory is larger: packed weights are repacked into two MLX
2-bit affine planes, and the remaining tables use bf16. This is a packed
encoder implementation, not Fermion's default densified fp16 encoder mode.

```bash
pip install 'mlx-audio[stt]'
```

```python
from mlx_audio.stt import load

model = load("FermionResearch/Phonon-2")
result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)
print(result.text)

for sentence in result.sentences:
    for token in sentence.tokens:
        print(token.start, token.end, token.text)
```

Token pieces carry timestamps; a word may span multiple pieces. Sentence
timestamps are available on `result.sentences`. Inputs can be paths or mono
MLX waveform arrays at 16 kHz. Path inputs are resampled by the normal audio
loader. Short nonempty inputs and final chunks are zero-padded to 320 samples (20 ms).
Empty inputs are rejected.

The first load fetches the official `.bps.tar.zst` artifact, verifies its
published SHA-256 and each extracted file, and converts the container into
ordinary Safetensors weights and config under `~/.cache/mlx_audio/phonon2/`.
Set `MLX_AUDIO_CACHE_DIR` to relocate the cache. Later loads verify and reuse
the conversion. `revision=` and `force_download=` follow the standard loader
API; `force_download=True` rebuilds the conversion. Source snapshots are never
modified, and no downloaded Python code is executed.

Local Hub snapshots, already unpacked official directories containing the
inner NeMo `config.json` plus `model.fermion`, and converted cache directories
can also be passed to `load()`. The Hub's outer artifact config is different
from the unpacked architecture config. NumPy and MLX handle all preparation;
neither the `fermion-research` package nor its native libraries are needed.
The generic `mlx_audio.convert` command is intentionally rejected: casting or
requantizing these packed tensors would corrupt the codes or change the model.
To redistribute a conversion, copy the loader's cache directory with its notices.

## CLI and chunk streaming

```bash
mlx_audio.stt.generate \
  --model FermionResearch/Phonon-2 \
  --audio audio.wav \
  --output-path transcript
```

For long recordings use explicit chunking as above, or the inherited
`model.generate("audio.wav", stream=True)` interface. This emits results from
overlapping chunks of a supplied recording; it is not a cache-aware live-input
streaming model. The CLI also accepts `--stream`.

## Boundaries and provenance

The model is English-only. Translation, speaker diarization, and Fermion's
hotword-biasing API are not included. Generic generation arguments do not turn
TDT decoding into an autoregressive language model. No new speed or WER claim
is made by this integration.

Container parsing, HF-to-MLX layout mapping, five-value packing design, and
16-step compiled TDT scheduling are adapted from `fermion-research` 0.2.11
(Apache-2.0, Copyright 2026 Fermion Research). LICENSE and NOTICE are included.
The [model weights](https://huggingface.co/FermionResearch/Phonon-2) are
CC-BY-4.0, inherited from NVIDIA's Parakeet TDT 0.6B v3. Retain their notices
when redistributing model files.
