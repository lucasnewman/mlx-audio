# Fluid Speech Pro

[Fluid Speech Pro by FluidVoice](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro)
is a multilingual FastConformer-TDT speech recognition model based on Parakeet.
This implementation includes its frame-cleaning block, NeMo audio preprocessing,
and optional clean mode, using MLX without PyTorch or parakeet-mlx at runtime.

## Supported Models

| Model | Support |
|-------|---------|
| [FluidVoice-Labs/fluid-speech-pro-mlx](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx) | Full model, including frame cleaner and clean mode; loads directly |
| [FluidVoice-Labs/fluid-speech-pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) | Transformers export; loads through Parakeet, without the frame cleaner or clean mode |

Both accept 16 kHz mono audio and support Bulgarian, Croatian, Czech, Danish,
Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian,
Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Russian, Slovak,
Slovenian, Spanish, Swedish, and Ukrainian. Language is detected automatically.

## Python

```python
from mlx_audio.stt import load

model = load("FluidVoice-Labs/fluid-speech-pro-mlx")
result = model.generate("speech.wav")
print(result.text)

for sentence in result.sentences:
    print(f"[{sentence.start:.2f}s - {sentence.end:.2f}s] {sentence.text}")
    for token in sentence.tokens:
        print(token.start, token.end, token.text)

# Edited text: removes fillers and repetitions.
print(model.generate("speech.wav", clean_mode=True).text)
```

The default is verbatim transcription. Clean mode requires the publisher's
`clean-mode.safetensors` sidecar, downloaded with the full MLX repository.
Local MLX Audio checkpoints, including quantized checkpoints with that sidecar,
can also be passed to `load`.

For long recordings, use overlapping chunks:

```python
result = model.generate("meeting.wav", chunk_duration=30, overlap_duration=2)

for chunk in model.generate("meeting.wav", stream=True, clean_mode=True):
    print(chunk.text, end="", flush=True)
```

Progressive output transcribes overlapping offline chunks. The encoder and
frame cleaner use future context; this is not a causal streaming model.
Decoding is greedy. Upstream leaderboard results use beam search.

## CLI

```bash
python -m mlx_audio.stt.generate \
    --model FluidVoice-Labs/fluid-speech-pro-mlx \
    --audio speech.wav \
    --output-path transcript \
    --format srt

python -m mlx_audio.stt.generate \
    --model FluidVoice-Labs/fluid-speech-pro-mlx \
    --audio speech.wav \
    --gen-kwargs '{"clean_mode": true}'
```

## License

Model and upstream reference code: [CC BY 4.0](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro/blob/main/LICENSE).
Credit **Fluid Speech Pro by FluidVoice**. The model builds on NVIDIA Parakeet
TDT 0.6B v3 and Moondream Parakeet Ultra.
