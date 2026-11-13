# Haru-4B

[higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) is a
full-duplex spoken dialogue base model. It consumes and produces mono 24 kHz
audio in 3200-sample frames (7.5 Hz), together with a text stream. It supports
English, Spanish, Portuguese, German, French, Chinese, Japanese, Russian and
Korean. The original checkpoint loads directly; no separate codec download or
PyTorch runtime is required.

## Python

```python
from mlx_audio import audio_io
from mlx_audio.sts.utils import load_model
from mlx_audio.sts.models.haru import render_card

model = load_model("higurashi-ai/Haru-4B")
card = render_card({
    "persona": "Haru is warm, curious, and enjoys talking about everyday life.",
    "name": "Haru",
    "backchannel": "mid",
    "other_name": "Sam",
    "relationship": "Old friends.",
    "situation": "A phone call to catch up.",
})
result = model.generate(
    "user.wav",
    ref_audio="voice.wav",
    system_prompt=card,
    tail_seconds=5,
    seed=42,
)
audio_io.write("reply.wav", result.audio, result.sample_rate)
print(result.text)
```

Use a clean single-speaker reference clip to condition the model's voice.
File inputs are resampled automatically. Audio arrays must be mono; set
`sample_rate` if they are not already 24 kHz. `generate()` appends silence
after the input so the model can finish speaking; `tail_seconds` controls
that duration. `max_new_tokens` caps output **frames**, each 133 ms long.
Frame zero precedes the first user frame by one frame interval.

When `system_prompt` is omitted or `None`, the Python APIs and live CLI use
this short, neutral default in the example character-card format:

```text
You are the person described below, talking with the other person by voice.
[Your Persona]
A calm, attentive conversational partner who responds naturally, clearly, and concisely.
[Your Name]
Haru
[Your Backchannels]
moderate
[Relationship]
You and the other person are getting to know each other.
[Situation]
A relaxed voice conversation with no set topic.
```

For a custom persona, `render_card()` formats the English character-card layout;
pass a complete localized card directly as `system_prompt` for other languages.
An explicit prompt replaces the default, and `system_prompt=""` disables it.
`speak_first=True` prompts the model to open the conversation.

## Live sessions

Run a conversation directly from your microphone, with speech played through
your output device and text printed as it arrives:

```bash
python -m mlx_audio.sts.live \
  --model higurashi-ai/Haru-4B \
  --ref-audio voice.wav
```

Use `--system-prompt "..."` or `--system-prompt-file card.txt` to override the
default, or `--system-prompt ""` to run without one. Add `--speak-first` for an
opening turn.

Use `--steps 1`, `--steps 2`, or `--steps 4` to compare
MeanFlow sampling steps per audio frame. Any positive integer is accepted;
omitting the flag uses the checkpoint default (1 for Haru-4B). More steps run
the audio head more times per frame, including warmup. Keep the voice, prompt,
and `--seed` fixed when comparing.

The installed command is also available as `mlx_audio.sts.live`. Use
`--list-devices` to list microphones and speakers, then select them with
`--input-device` and `--output-device` (device indices or names). Use headphones
to prevent the model from hearing its own output. Press Ctrl-C to stop, or set
`--max-frames` for a bounded conversation. The default is one hour.

`--speak-first` encourages an opening turn. Omitting it does not impose a
speech-detection gate: Haru can still choose to speak before microphone speech
arrives, and interruption behavior is determined by the model.

The CLI adds a 100 ms playback cushion to absorb inference timing variations;
adjust it with `--playback-buffer` (0–0.5 seconds). Model inference stays on the
main thread, and audio callbacks only transfer PCM. An input backlog over one
second stops the session with an error. Streaming shapes are warmed up before
capture starts, using separate conversation state. No recordings are saved.

The live CLI always decodes audio on a separate MLX stream.
The backbone, audio head, and decoder are submitted with `mx.async_eval()`;
pending frames defer the current token read and PCM conversion. The live loop
can encode incoming microphone audio or advance the next backbone step while
PCM is decoding. It keeps at most two output frames in flight and plays them in
order, then drains pending output before ending the session.

To integrate your own microphone and audio output, use the session API:

```python
from mlx_audio.sts.models.haru import DuplexConfig
from mlx_audio.sts.models.haru.audio import load_audio, prepare_reference

voice = prepare_reference(load_audio("voice.wav"))
session = model.create_session(
    system_prompt=card,
    reference=voice,
    config=DuplexConfig(seed=42, max_frames=750),
)
frame = session.start()  # first model frame, before receiving user audio
# Send frame.audio to your audio output and frame.text to your text display.

# For each microphone frame (3200 mono float32 samples at 24 kHz):
# frame = session.step(user_audio)
# Stop when frame.done is True.
```

For microphone buffers of arbitrary size, call `session.feed(samples)` and
then `session.step()` whenever `session.frames_ready > 0`. Each session owns
its language-model and codec caches. Use a session from one thread.
Calling `start()` again resets the conversation, random seed, and repetition
history. The audio head uses guidance in the range 1–2 (default 1.5).
Set `DuplexConfig(meanflow_steps=2)` for a session override, or pass
`meanflow_steps=2` to `model.generate()`. File generation also accepts
`--gen-kwargs '{"meanflow_steps": 2}'`.

`start()` and `step()` resolve frames from the same asynchronous pipeline and
return NumPy PCM. For custom pipelines, `start_async()` and `step_async()` return
a `PendingFrame` with asynchronously evaluated token and audio data.
Accessing `.token` or `.text`
waits for the token if necessary; `.resolve()` returns the usual `Frame` with
NumPy PCM. Schedule the next frame before resolving the previous one to overlap
work. Repetition penalties still need the previous token before sampling the
next, and that dependency can wait. These are synchronous Python methods with
asynchronous GPU evaluation. Keep all session calls and resolution on the
loading thread, resolve frames in order, bound the pending
queue, and call `session.synchronize_decoder()` when stopping. A restart waits
for old decoder work before resetting state.

## CLI

```bash
python -m mlx_audio.sts.generate \
  --model higurashi-ai/Haru-4B \
  --audio user.wav \
  --ref-audio voice.wav \
  --system-prompt "$(cat card.txt)" \
  --gen-kwargs '{"tail_seconds": 5, "speak_first": false}' \
  --seed 42 \
  --output-path reply.wav
```

The CLI saves audio to `reply.wav` and the model's text stream to `reply.txt`.
For microphone conversations, use `mlx_audio.sts.live` as shown above.

## Architecture and license

Haru combines the Qwen3.5-4B text backbone, a causal VibeVoice acoustic codec,
two audio projectors, and a MeanFlow audio head. Text, previous model audio,
and current user audio are summed into each language-model position. The
runtime preserves float32 zero-centred normalization gains, recurrent decay,
and PAD/EOP logits.
The standard MLX Audio converter supports the model; its quantization
predicate keeps the codec, audio head, and projectors at their original
precision.

This is a pretrained dialogue base model, not an instruction-tuned assistant.
The weights and adapted Haru reference components are Apache-2.0. See the
[upstream source](https://github.com/higurashi-ai/Haru),
[license](https://github.com/higurashi-ai/Haru/blob/main/LICENSE), and
[third-party notices](https://github.com/higurashi-ai/Haru/blob/main/THIRD_PARTY_NOTICES.md).
