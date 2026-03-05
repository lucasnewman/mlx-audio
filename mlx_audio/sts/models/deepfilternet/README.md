# DeepFilterNet (MLX)

DeepFilterNet speech enhancement in pure MLX with support for model versions 1, 2, and 3.

## Quick Start

```python
from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

model = DeepFilterNetModel.from_pretrained(version=3)
model.enhance_file("noisy.wav", "clean.wav")
```

Or load a custom/finetuned checkpoint path directly (version auto-detected from `config.json`):

```python
model = DeepFilterNetModel.from_pretrained(model_path="./models/MyDeepFilterNet")
```

Streaming/chunked mode (true per-hop stateful processing for DF2/DF3):

```python
streamer = model.create_streamer(pad_end_frames=3, compensate_delay=True)
out_1 = streamer.process_chunk(chunk_a)
out_2 = streamer.process_chunk(chunk_b)
out_tail = streamer.flush()
```

## Model Selection

- `version=1`: DeepFilterNet
- `version=2`: DeepFilterNet2
- `version=3`: DeepFilterNet3

Optional explicit model path:

```python
model = DeepFilterNetModel.from_pretrained(model_path="./models/DeepFilterNet2/model.safetensors")
```

## Example Script

```bash
python examples/deepfilternet.py examples/denoise/test_audio_10s.wav -m 3
python examples/deepfilternet.py examples/denoise/test_audio_10s.wav --model-path ./models/DeepFilterNet3
python examples/deepfilternet.py examples/denoise/test_audio_10s.wav -m 3 --stream
```
