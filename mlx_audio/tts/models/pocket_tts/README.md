# Phonon

Phonon is a text-to-speech model that generates 24 kHz speech on Apple Silicon, with four bundled voices and support
for streaming audio generation.

## Supported Models

- `nativ-community/phonon`

## Usage

Python API:

```python
import numpy as np

from mlx_audio.audio_io import write
from mlx_audio.tts.utils import load

model = load("nativ-community/phonon")

result = next(model.generate(
    text="Hello, this is Phonon running on Apple Silicon.",
    voice="Freya",
))
write("phonon.wav", np.array(result.audio), result.sample_rate)
```

CLI:

```bash
python -m mlx_audio.tts.generate \
  --model nativ-community/phonon \
  --text "Hello, this is Phonon running on Apple Silicon." \
  --voice Freya \
  --play
```

Add `--stream` to the CLI command for streaming playback, or pass `stream=True`
to `model.generate()` to receive audio chunks in Python.

## Voices

Available voices: `Freya`, `Harper`, `Sterling`, and `Toby`. The default is `Freya`.
