# Fun-ASR-Nano-2512

MLX runtime and converter for `FunAudioLLM/Fun-ASR-Nano-2512`.

Official MLX checkpoint: `mlx-community/Fun-ASR-Nano-2512`.

Convert the upstream PyTorch checkpoint before loading:

```bash
python -m mlx_audio.stt.models.fun_asr_nano.convert \
  --hf-path FunAudioLLM/Fun-ASR-Nano-2512 \
  --mlx-path Fun-ASR-Nano-2512-mlx
```

Load the published MLX checkpoint:

```python
from mlx_audio.stt.utils import load_model

model = load_model("mlx-community/Fun-ASR-Nano-2512")
result = model.generate("audio.wav", language="zh")
print(result.text)
```

ISO language hints include `zh`, Chinese dialect codes that map to the Chinese prompt (`yue`, `wuu`, `nan`, `hak`, `gan`, `hsn`, `cjy`), `en`, and `ja`.

This runtime is transcription-only. The current public checkpoint does not include timestamp head weights.
