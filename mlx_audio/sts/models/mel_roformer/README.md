# Mel-Band-RoFormer

Mel-Band-RoFormer is a transformer-based architecture for audio source separation,
particularly effective for vocal isolation from music. This is an MLX port of the
architecture described in Lu et al., "Mel-Band RoFormer for Music Source Separation"
(2024), with reference implementations in
[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) and
[ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).

## What this contribution is

**This module ships the architecture and tooling.** No checkpoint is bundled,
mirrored, or endorsed as the "default" or "recommended" choice. Users bring
their own weights — either by converting a community PyTorch checkpoint with
the included `convert.py` script, or by training their own via the
MSS-Training framework.

This decoupling is deliberate: different published Mel-Band-RoFormer checkpoints
have different licenses, some declared and some not, and it's better to let
users make an informed choice about which checkpoint to pair with this
architecture than to pick one implicitly via a default.

## Architecture

- **Input**: Stereo audio at 44.1 kHz (`[B, 2, samples]`)
- **Pipeline**: STFT → CaC interleave → BandSplit → N× DualAxisTransformer → MaskEstimate → complex multiply → iSTFT
- **Output**: Separated stem (e.g., vocals) as stereo audio (`[B, 2, samples]`)

Key design elements:
- **Mel-scale band splitting** into 60 frequency bands (Slaney mel scale, binarized)
- **Dual-axis attention** — alternating time-axis and frequency-axis RoPE transformers
- **Per-head sigmoid gates** on attention outputs
- **Per-band MLP mask estimation** with GLU activation
- **Complex-valued masking** via real/imaginary interleaved representation

## License Posture

The architecture code is MIT-licensed (inherited from the reference implementations).
Individual checkpoint weights have independent licenses:

| Checkpoint family | Source | License |
|-------------------|--------|---------|
| Kim Vocal 2 | [`KimberleyJSN/melbandroformer`](https://huggingface.co/KimberleyJSN/melbandroformer) | **MIT** (relicensed April 2026 by author) |
| viperx vocals | [`TRvlvr/model_repo`](https://github.com/TRvlvr/model_repo/releases) | Undeclared (default copyright) |
| anvuew dereverb | [`anvuew/dereverb_mel_band_roformer`](https://huggingface.co/anvuew/dereverb_mel_band_roformer) | GPL-3.0 |
| ZFTurbo MSS-Training release assets | [ZFTurbo MSS-Training releases](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases) | MIT (inherited from repo) |
| Community variants (dereverb, denoise, crowd) | various | Varies — review each |

## Running inference vs. redistributing weights

**This is the distinction that matters for most users.** A permissive-license
weight file (MIT, Apache-2.0) can be freely redistributed as part of a product.
A GPL-3.0 weight file can be run locally on your own audio without any
obligation — the GPL-3 constraints only kick in when you *distribute* the
weights or weight-derived artifacts as part of a product.

Concretely:

- ✅ **Running any of these checkpoints locally on your own audio**:
  No obligation regardless of license. You're the end user.
- ✅ **Publishing a research paper or demo that used GPL-3.0 weights**
  (e.g., anvuew dereverb): Attribution is polite; distribution of the weights
  triggers GPL.
- ⚠️ **Shipping an app or service that bundles or auto-downloads GPL-3.0
  weights**: Your product becomes subject to GPL-3 obligations.
- ❌ **Shipping a closed-source commercial product that bundles GPL-3.0
  weights**: Not permitted without dual-licensing from the weight author.
- ⚠️ **Using weights with an undeclared license (e.g., viperx) in any
  distributed product**: No explicit redistribution right exists — proceed
  only after clarifying with the original author or choosing a different
  checkpoint.

For commercial products and distributed applications, MIT-licensed checkpoints
— including Kim Vocal 2 (as of April 2026) and the ZFTurbo MSS-Training
release assets — are the safest choice, alongside any checkpoint you train
yourself using the MIT-licensed MSS-Training configs.

## Usage

### Loading a model

Each preset matches a specific checkpoint family's architecture. You must
pick the preset that matches the checkpoint you intend to load.

```python
from mlx_audio.sts.models.mel_roformer import MelRoFormer, MelRoFormerConfig

# Pick the config preset matching your checkpoint family
config = MelRoFormerConfig.kim_vocal_2()           # depth=6
# or:
config = MelRoFormerConfig.viperx_vocals()         # depth=12
# or:
config = MelRoFormerConfig.zfturbo_bs_roformer()   # depth=12, MIT-licensed weights
# or for non-standard variants:
config = MelRoFormerConfig.custom(depth=8, num_bands=48)

# Load from a local directory (must contain a matching safetensors file)
model = MelRoFormer.from_pretrained("./my_weights_dir", config=config)

# Or from a HuggingFace repo ID (if the repo contains a compatible
# `<name>.config.json` produced by convert.py, the config is loaded from there)
model = MelRoFormer.from_pretrained("your-org/mel-band-roformer-mlx")
```

### Running separation

```python
import mlx.core as mx
from mlx_audio.sts.models.mel_roformer import MelRoFormer, MelRoFormerConfig

model = MelRoFormer.from_pretrained("path/to/weights", config=MelRoFormerConfig.kim_vocal_2())

# Stereo audio at 44.1 kHz — shape [1, 2, samples]
audio = mx.random.normal((1, 2, 44100 * 8))  # 8 seconds of dummy audio

# Separate vocals
vocals = model(audio)
# vocals has shape [1, 2, samples]
```

For audio longer than `chunk_size` (default 8 seconds), split into overlapping
chunks and apply crossfade reconstruction — a convenience wrapper for this is
planned but not yet in-tree.

## Converting PyTorch Checkpoints

If you have a PyTorch `.ckpt` or `.pt` checkpoint from MSS-Training or a
community source, convert it with:

```bash
python -m mlx_audio.sts.models.mel_roformer.convert \
    --input path/to/checkpoint.ckpt \
    --output path/to/output_dir/
```

The script:
- Accepts `.ckpt`, `.pt`, or `.safetensors` input
- Writes a content-addressed output file (`<basename>.<sha256[:8]>.safetensors`)
  and companion `<basename>.<sha256[:8]>.config.json`
- Skips if the output already exists (idempotent re-runs are free)
- Warns about known-license sources when detected in the input path
- Splits packed `to_qkv` projections into separate `to_q`, `to_k`, `to_v` weights
- Strips training state (optimizer, schedulers, EMA, running stats)

**No PyTorch parity validation is run inside the script** — MLX avoids
shipping torch as a runtime dependency. For parity verification against the
PyTorch reference, see `tests/sts/test_mel_roformer_parity.py` (requires
torch locally, gated behind `@pytest.mark.requires_torch`).

## References

- Lu et al., "Mel-Band RoFormer for Music Source Separation" (2024)
- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) — original PyTorch (MIT)
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) — training framework and config files (MIT)

## Credits

Architecture by the paper authors at ByteDance AI Labs.
Reference implementations by lucidrains and ZFTurbo.
Individual checkpoints by their respective trainers (Kim, viperx, anvuew,
aufr33, and others in the UVR/MSS community).
