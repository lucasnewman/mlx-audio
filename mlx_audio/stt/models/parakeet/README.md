# Parakeet

MLX implementation of NVIDIA's Parakeet speech recognition family, with a
FastConformer encoder and support for transcription, timestamps, and overlapping
chunked transcription on Apple silicon. Parakeet TDT v2 supports English; v3
supports 25 European languages with automatic language detection, punctuation,
and capitalization.

The family also supports Moondream's Parakeet Redux, a ternary version of
Parakeet TDT v3 with the same language coverage, and dense Parakeet TDT v3
checkpoints in Hugging Face format, such as Parakeet Ultra and Orukeet.

## Supported Models

| Model | Languages | Description |
|-------|-----------|-------------|
| [mlx-community/parakeet-tdt-0.6b-v2](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2) | English | Parakeet TDT v2 |
| [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) | 25 European languages | Multilingual Parakeet TDT v3 |
| [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) | 25 European languages | Ternary encoder; approximately 217 MB of MLX weights |
| [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) | 25 European languages | Dense Hugging Face `ParakeetForTDT` export |
| [oruk/orukeet](https://huggingface.co/oruk/orukeet) | 25 European languages | Dense Hugging Face `ParakeetForTDT` export; CC-BY-SA-4.0 |
| [FluidVoice-Labs/fluid-speech-pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) | 25 European languages | Dense Transformers export; see [Fluid Speech Pro](../fluid_speech/README.md) for the full model with frame cleaning and clean mode |

All listed repositories load directly with `mlx_audio.stt.load`. A local
checkpoint directory can also be passed to `load`.

## Python Usage

```python
from mlx_audio.stt import load

model = load("mlx-community/parakeet-tdt-0.6b-v3")
result = model.generate("speech.wav")
print(result.text)

# Sentence timestamps, in seconds.
for sentence in result.sentences:
    print(f"[{sentence.start:.2f}s - {sentence.end:.2f}s] {sentence.text}")
```

To use the ternary model:

```python
from mlx_audio.stt import load

model = load("moondream/parakeet-redux")
print(model.generate("speech.wav").text)
```

### Long Audio and Streaming Output

Use overlapping chunks for long recordings:

```python
result = model.generate(
    "long_audio.wav",
    chunk_duration=30.0,
    overlap_duration=2.0,
)
print(result.text)
```

Receive text progressively as overlapping chunks are transcribed:

```python
for chunk in model.generate("long_audio.wav", stream=True):
    print(chunk.text, end="", flush=True)
```

## CLI Usage

```bash
python -m mlx_audio.stt.generate \
    --model mlx-community/parakeet-tdt-0.6b-v3 \
    --audio speech.wav \
    --output-path transcript \
    --format txt
```

Use Redux and save a JSON transcript with timestamps:

```bash
python -m mlx_audio.stt.generate \
    --model moondream/parakeet-redux \
    --audio speech.wav \
    --output-path transcript \
    --format json
```

The CLI also supports `--format srt` and `--format vtt` for subtitles,
`--chunk-duration 30` for chunk length in seconds, and `--stream` for progressive
output. `--output-path transcript` writes `transcript` with the selected format's
file extension.

## Supported Languages

Parakeet TDT v3, Redux, Ultra, and Orukeet support Bulgarian, Croatian, Czech,
Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian,
Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian,
Slovak, Slovenian, Spanish, Swedish, and Ukrainian.
