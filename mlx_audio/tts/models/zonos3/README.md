# ZONOS3-Mini

[ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) is Zyphra's multilingual
text-to-speech model with 100 million active parameters. It generates 44.1 kHz
mono audio with 27 built-in voices, streaming, and voice cloning from reference
audio without a transcript.

## Supported Models

- [mlx-community/ZONOS3-Mini](https://huggingface.co/mlx-community/ZONOS3-Mini)

## CLI Example

```bash
python -m mlx_audio.tts.generate \
  --model mlx-community/ZONOS3-Mini \
  --text "Hello from ZONOS3 Mini!" \
  --voice jett
```

Use `--ref_audio reference.wav` to clone a voice, or add `--stream` for
streaming playback.

## Python Example

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts import load

model = load("mlx-community/ZONOS3-Mini")

result = next(model.generate(
    text="Hello from ZONOS3 Mini!",
    voice="jett",
))
audio_write("speech.wav", result.audio, result.sample_rate)

# Voice cloning does not require a reference transcript.
result = next(model.generate(
    text="Welcome back.",
    ref_audio="reference.wav",
))
audio_write("cloned.wav", result.audio, result.sample_rate)
```

Use `model.voices` to list the included voices. `voice="none"` generates without
a speaker prompt. Reference audio paths are resampled automatically; array
inputs must be mono PCM at 44,100 Hz. `speaker_embedding` accepts a
192-dimensional vector or a NumPy file, and `speaker_cfg_scale` controls
speaker guidance (default `1.0`).

Set `stream=True` for incremental synthesis, with `streaming_interval` in
seconds. `speed` controls speaking rate. Acoustic controls are available through
`condition_values={"loudness": -20, "fmax": 20000, "speaking_rate": 16}` or
`condition_buckets`. `max_tokens` limits generated delayed audio rows at 15 Hz.

Lightweight English number normalization is enabled by default; other languages
pass through as written. Set `text_normalization=False` for text already
normalized by another frontend. Speech editing, accent adapters, hidden-state
steering, and raw reference-code encoding are not exposed by this implementation.

## License

Model weights are released under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The backbone, codec, and speaker encoder are adapted from
[Zyphra's MIT-licensed reference implementation](https://github.com/Zyphra/ZONOS3-Mini/blob/6ff97cac0c684732de5f2bc6e3671aa783c15054/LICENSE).
The Gated DeltaNet kernel includes MIT-licensed code by Apple, also available
in [mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE).
