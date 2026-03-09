# DeepFilterNet (MLX)

DeepFilterNet speech enhancement in pure MLX with support for model versions 1, 2, and 3.

## Quick Start

```python
from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

model = DeepFilterNetModel.from_pretrained()
model.enhance_file("noisy.wav", "clean.wav")
```

Or load from a local model directory (must contain `config.json` and weights):

```python
model = DeepFilterNetModel.from_pretrained("./models/MyDeepFilterNet")
```

Or load from a Hugging Face repo id:

```python
model = DeepFilterNetModel.from_pretrained("iky1e/DeepFilterNet3-MLX")
```

Streaming/chunked mode (true per-hop stateful processing for DF2/DF3):

```python
streamer = model.create_streamer(pad_end_frames=3, compensate_delay=True)
out_1 = streamer.process_chunk(chunk_a)
out_2 = streamer.process_chunk(chunk_b)
out_tail = streamer.flush()
```

## Model Selection

Model architecture is selected from `config.json` (`model_version`).

## Example Script

```bash
python examples/deepfilternet.py examples/denoise/test_audio_10s.wav
python examples/deepfilternet.py examples/denoise/test_audio_10s.wav --model ./models/DeepFilterNet3
python examples/deepfilternet.py examples/denoise/test_audio_10s.wav --model iky1e/DeepFilterNet3-MLX
python examples/deepfilternet.py examples/denoise/test_audio_10s.wav --stream
```
