# FastDialogueSidon

Separate and restore recordings of two speakers on Apple Silicon. The model
uses a single CTM transformer step and a two-iteration WaveFitNeXt decoder,
producing two mono tracks at **24 kHz** with the same duration as the input.

## Supported models

| Model | Precision | Weight size |
|-------|-----------|-------------|
| [higurashi-ai/FastDialogueSidon](https://huggingface.co/higurashi-ai/FastDialogueSidon) | FP32 | 1.73 GB |

The original checkpoint loads directly. No conversion, PyTorch, Triton, or
TensorRT installation is required. Local MLX checkpoints are also supported.

## Python

```python
from mlx_audio.sts import load
from mlx_audio.audio_io import write

model = load("higurashi-ai/FastDialogueSidon")
result = model.separate("dialogue.wav", seed=0)
for i, speaker in enumerate(result.speakers, 1):
    write(f"speaker_{i}.wav", speaker, result.sample_rate)
```

`result.speakers` has shape `[2, samples]`. For NumPy or MLX arrays, pass
`sample_rate`: `model.separate(audio, sample_rate=16000, seed=0)`.
Arrays use `[samples]` or `[samples, channels]` layout; multichannel audio is
averaged to mono and resampled to 16 kHz before processing.

The shared STS interface is also available through `model.generate(...)`,
which returns an `STSOutput` with `speaker_1` and `speaker_2` in `audios`.

## Command line

```bash
python -m mlx_audio.sts.generate \
    --model higurashi-ai/FastDialogueSidon \
    --audio dialogue.wav --output-path separated.wav --seed 0
```

This writes `separated_speaker_1.wav` and `separated_speaker_2.wav`.

## Long recordings and reproducibility

Recordings use 20-second chunks with 5-second overlap, speaker alignment, and
crossfading. Adjust `chunk_seconds` / `--chunk-seconds` and `overlap_seconds` /
`--overlap-seconds` as needed. Smaller chunks reduce memory use. In Python,
`chunk_seconds=None` processes the entire input at once; attention memory grows
quadratically with duration. The upstream model is evaluated on windows up to
120 seconds.

Each chunk is padded to at least 3.2 seconds and a multiple of 20 ms before
feature extraction, matching the upstream runtime's exact-length path. Outputs
are trimmed to the input duration. A fixed `seed` makes runs repeatable with
the same input and settings without resetting MLX's global random stream.

FastDialogueSidon always uses **one CTM step**. Omit `num_steps` / `--num-steps`
or set it to `1`; other values are rejected. The two vocoder iterations are
part of the trained decoder and are separate from this step count.

The model processes recorded audio offline and supports exactly two speakers.
Tracks are anonymous speaker slots; speaker identities may switch after long
silences. Restoration can change the signal, so the sum of the tracks need
not reproduce the input exactly.

## Attribution and license

FastDialogueSidon by Rikuto Kotoge (higurashi-ai), based on SaruLab's
DialogueSidon. The model weights and reference code use
[Apache-2.0](https://github.com/higurashi-ai/FastDialogueSidon/blob/main/LICENSE).
The encoder and DiT build on the MIT-licensed Sidon implementation, and the
WaveFitNeXt decoder derives from Yukara Ikemiya's MIT-licensed wavefit-pytorch.
See the upstream
[third-party notices](https://github.com/higurashi-ai/FastDialogueSidon/blob/main/THIRD_PARTY_NOTICES.md).
