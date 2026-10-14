# Fish Audio Drama 3

Fish Audio's multi-speaker text-to-speech model for scenes and long-form content. It pairs a 35B MoE backbone (3B active per token) with a small audio tower that predicts the ten codec codebooks, and decodes with the Fish S1 DAC codec at 44.1 kHz. Based on [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3).

## Converting the checkpoint

The release stores its routed experts as block FP8, so convert it once into an MLX checkpoint:

```bash
hf download fishaudio/drama-3 --local-dir drama-3
python -m mlx_audio.tts.models.fish_drama3.convert \
    --src drama-3 --dst drama-3-mlx
```

By default experts are quantized to 4 bits and the dense projections and audio tower to 8 bits, which comes to about 22 GB. `--expert-bits`, `--dense-bits` and `--tower-bits` change that, and `0` keeps bfloat16. Reading `codec.pth` needs PyTorch.

On an M3 Max the default checkpoint generates a frame (46 ms of audio) in about 22 ms, roughly twice real time, and peaks at about 26 GB of memory.

## Usage

=== "CLI"

    ```bash
    python -m mlx_audio.tts.generate \
        --model drama-3-mlx \
        --text "[sighs] Fine. If we miss it, it's on you." \
        --ref_audio reference.wav \
        --ref_text "Transcript of the reference clip." \
        --temperature 0.7 --top_p 0.7 --top_k 30
    ```

=== "Python"

    ```python
    from mlx_audio.tts.utils import load_model

    model = load_model("drama-3-mlx")

    for result in model.generate(
        text="[sighs] Fine. If we miss it, it's on you.",
        ref_audio="reference.wav",
        ref_text="Transcript of the reference clip.",
    ):
        print(result.audio_duration)
    ```

Without a reference the model picks a voice itself. Delivery cues such as `[whispers]` or `[laughs]` go inline where they should happen. In Python, sampling arguments left unset use the reference recipe (temperature 0.7, top-p 0.7, top-k 30); the CLI has its own defaults, so pass them explicitly there.

## Multiple speakers

Speaker tags `<|speaker:0|>` to `<|speaker:15|>` switch characters inside one text. Untagged text is speaker 0. To give each speaker a voice, pass one reference per speaker and start each transcript with that speaker's tag:

```python
results = model.generate(
    text="<|speaker:0|> Are you coming? <|speaker:1|> Give me a minute.",
    ref_audio=["anna.wav", "ben.wav"],
    ref_text=[
        "<|speaker:0|> Transcript of Anna's clip.",
        "<|speaker:1|> Transcript of Ben's clip.",
    ],
)
```

Long text is split into chunks of about 300 bytes with at most three speakers each. Every chunk sees the first and the previous chunk as context, so voices stay consistent. The encoded references and their prompt prefix are cached for the four most recent voices.

## Streaming

With `stream=True`, audio is yielded every `streaming_interval` seconds while the frames are sampled. `first_chunk_frames` makes the first piece smaller for a faster first audio:

```python
from mlx_audio.tts.audio_player import AudioPlayer

player = AudioPlayer(sample_rate=model.sample_rate)
for result in model.generate(
    text="Streaming speech from Drama 3.",
    ref_audio="reference.wav",
    ref_text="Transcript of the reference clip.",
    stream=True,
    streaming_interval=0.32,
    first_chunk_frames=4,
):
    player.queue_audio(result.audio)
player.wait_for_drain()
```

With a cached voice and `first_chunk_frames=4`, the first audio arrives in under 200 ms on an M3 Max.

## License

The Drama 3 weights are released under the Fish Audio Community License. Check its terms before commercial use or hosting.
