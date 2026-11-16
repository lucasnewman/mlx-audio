# Stable Audio 3

Stable Audio 3 generates stereo 44.1 kHz audio from a text prompt: a T5Gemma
text encoder, a diffusion transformer sampled with ping-pong in 8 steps, and a
SAME decoder. Small uses a 20-layer transformer and SAME-S and generates up to
120 seconds. Medium uses a 24-layer transformer with differential attention and
SAME-L and generates up to 380 seconds.

| Model | Purpose |
|---|---|
| [`mlx-community/stable-audio-3-small-sfx`](https://huggingface.co/mlx-community/stable-audio-3-small-sfx) | Sound effects |
| [`mlx-community/stable-audio-3-small-music`](https://huggingface.co/mlx-community/stable-audio-3-small-music) | Music |
| [`mlx-community/stable-audio-3-medium`](https://huggingface.co/mlx-community/stable-audio-3-medium) | Music and sound effects, up to 380 seconds |

## Generate

=== "CLI"

    ```bash
    python -m mlx_audio.music.generate \
        --model mlx-community/stable-audio-3-small-sfx \
        --caption "Rain on a tin roof with distant thunder" \
        --duration 10 \
        --seed 7 \
        --output rain.wav
    ```

=== "Python"

    ```python
    from mlx_audio.music import load

    model = load("mlx-community/stable-audio-3-small-music")

    for result in model.generate(
        text="Lofi house loop, 120 BPM",
        duration=20,
        steps=8,
        seed=7,
    ):
        print(result.audio.shape, result.sample_rate)
    ```

`--cfg-scale` defaults to 1, which turns guidance off. A larger scale doubles
the model calls per step and enables `--negative-prompt`.

## Convert

The medium weights are converted from the MLX export in
[`stabilityai/stable-audio-3-optimized`](https://huggingface.co/stabilityai/stable-audio-3-optimized)
(revision `30cb2644da54bfa63275d1bc7f1fa67fc718680f`), with the config and text
encoder from `stabilityai/stable-audio-3-medium-base`:

```bash
python -m mlx_audio.music.models.stable_audio_3.convert \
    --mlx-path ./stable-audio-3-medium
```

## License

The code follows Stability AI's MIT-licensed `optimized/mlx` code,
Copyright (c) 2026 Stability AI. The weights are under the
[Stability AI Community License](https://stability.ai/community-license-agreement)
and the text encoder under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Read both before you
use or redistribute the model.
