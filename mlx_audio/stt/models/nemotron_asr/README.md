# Nemotron 3.5 ASR (streaming) — MLX

MLX implementation of NVIDIA's
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b):
a 600M-parameter **cache-aware streaming FastConformer-RNNT** with **language-ID prompt
conditioning**, covering 40 language-locales with punctuation and capitalization.

## Architecture

```
audio ─► log-mel (128, no norm) ─► FastConformer encoder (24 layers, d=1024)
                                         │  causal dw-striding subsampling ×8
                                         │  causal depthwise conv, layer-norm
                                         │  rel-pos attn, chunked-limited mask
                                         ▼
                  language one-hot (128) ─┴─► concat ─► prompt_kernel (1152→2048→1024)
                                         ▼
                              RNN-T joint + 2-layer LSTM prediction net ─► greedy decode
```

Run offline, the `chunked_limited` attention mask reproduces the training-time look-ahead,
so a single full-utterance pass matches the streaming model's output.

## Convert

The model ships only as a NeMo `.nemo` archive, so convert it once (needs `torch` + `pyyaml`):

```bash
python -m mlx_audio.stt.models.nemotron_asr.convert \
    --nemo-path nvidia/nemotron-3.5-asr-streaming-0.6b \
    --mlx-path  ./nemotron-3.5-asr-streaming-0.6b-mlx
```

`--nemo-path` takes a local `.nemo` file **or** a HuggingFace repo id. Use
`--dtype float16` for a smaller model.

## Use

```python
from mlx_audio.stt import load

model = load("./nemotron-3.5-asr-streaming-0.6b-mlx")

# auto language detection (default)
print(model.generate("speech.wav").text)

# force a language via its prompt key
print(model.generate("speech.wav", language="en-US").text)
```

CLI:

```bash
python -m mlx_audio.stt.generate \
    --model ./nemotron-3.5-asr-streaming-0.6b-mlx --audio speech.wav --format txt
```

## Language prompt

Pass `language=<key>` where `<key>` is from the model's `prompt_dictionary`
(`en-US`, `es-ES`, `zh-CN`, `fr-FR`, …, or `auto`). In `auto` mode the model
infers the language and would emit a leading `<lang>` tag; these tags are stripped
from the returned text and surfaced via the detected language instead.

## Look-ahead / latency

`generate(..., att_context_size=[left, right])` selects one of the trained
look-aheads (`[56,3]`, `[56,0]`, `[56,6]`, `[56,13]`). The default `[56,13]`
gives the best offline accuracy; smaller right contexts trade accuracy for lower latency.
