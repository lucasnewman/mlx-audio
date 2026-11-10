# Stable Audio 3

Stable Audio 3 small generates stereo 44.1 kHz audio from a text prompt: a
T5Gemma text encoder, a 20-layer diffusion transformer sampled with ping-pong
in 8 steps, and the SAME-S decoder. It generates up to 120 seconds.

| Model | Purpose |
|---|---|
| [`mlx-community/stable-audio-3-small-sfx`](https://huggingface.co/mlx-community/stable-audio-3-small-sfx) | Sound effects |
| [`mlx-community/stable-audio-3-small-music`](https://huggingface.co/mlx-community/stable-audio-3-small-music) | Music |

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

## License

The weights are under the
[Stability AI Community License](https://stability.ai/community-license-agreement)
and the text encoder under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Read both before you
use or redistribute the model.
