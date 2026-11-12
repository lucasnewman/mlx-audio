"""MLX inference for Salesforce SOV-Transcribe-1."""

import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_audio.lm.models import gemma4_text
from mlx_audio.lm.sample_utils import make_sampler
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Qwen2AudioEncoderLayer
from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram
from mlx_audio.stt.utils import load_audio, merge_hotwords

from .config import ModelConfig

DEFAULT_PROMPT = (
    "Transcribe the audio verbatim, outputting only spoken words in sequence."
)
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [
            Qwen2AudioEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(config.d_model)
        self._frames = 2 * config.max_source_positions

    def __call__(self, features):
        if features.shape[-1] != self._frames:
            raise ValueError(f"Expected {self._frames} mel frames per audio window")
        x = features.astype(self.conv1.weight.dtype).transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = (x + self.embed_positions.weight).astype(x.dtype)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class AudioProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.d_model * config.downsample_factor,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def __call__(self, x):
        return self.linear_2(nn.gelu(self.linear_1(x)))


class Float32Router(gemma4_text.Router):
    def __call__(self, x):
        return super().__call__(x.astype(mx.float32))


class RoutedExperts(gemma4_text.Experts):
    def __call__(self, x, top_k_indices, top_k_weights):
        y = self.switch_glu(x, top_k_indices)
        # Cast each weighted expert contribution back to the hidden-state dtype
        # before accumulating, as in the reference implementation.
        return (y * top_k_weights[..., None]).astype(x.dtype).sum(-2)


class TranscribeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_tower = AudioEncoder(config.audio_config)
        self.embed_audio = AudioProjector(config)
        self.language_model = gemma4_text.Gemma4TextModel(config.text_config)
        # SOV scales text embeddings before replacing the audio placeholders.
        # Projected audio enters the decoder without the text embedding scale.
        self.language_model.embed_scale = 1.0
        for layer in self.language_model.layers:
            if layer.enable_moe:
                layer.router = Float32Router(config.text_config)
                layer.experts = RoutedExperts(config.text_config)


@dataclass
class TranscriptionOutput(STTOutput):
    finish_reason: str = "stop"


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = TranscribeModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        self._tokenizer = None
        self._eos_token_ids = {1, 106, 50}

    @property
    def sample_rate(self):
        return SAMPLE_RATE

    @property
    def layers(self):
        return self.model.language_model.layers

    def make_cache(self):
        return gemma4_text.Model.make_cache(self)

    @property
    def args(self):
        return self.config.text_config

    def _embed_text(self, input_ids):
        x = self.model.language_model.embed_tokens(input_ids)
        return x * mx.array(self.args.hidden_size**0.5, dtype=x.dtype)

    def __call__(
        self, input_ids=None, cache=None, input_embeddings=None, logits_to_keep=1
    ):
        if input_embeddings is None:
            input_embeddings = self._embed_text(input_ids)
        h = self.model.language_model(cache=cache, input_embeddings=input_embeddings)
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        if self.config.tie_word_embeddings:
            logits = self.model.language_model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        if self.args.final_logit_softcapping is not None:
            logits = gemma4_text.logit_softcap(
                self.args.final_logit_softcapping, logits
            )
        return logits

    def sanitize(self, weights):
        language = {}
        sanitized = {}
        for key, value in weights.items():
            if key == "lm_head.weight" and self.config.tie_word_embeddings:
                continue
            if key.startswith("model.language_model."):
                language[key.removeprefix("model.language_model.")] = value
            else:
                if (
                    key
                    in {
                        "model.audio_tower.conv1.weight",
                        "model.audio_tower.conv2.weight",
                    }
                    and value.shape[1] != 3
                ):
                    value = value.transpose(0, 2, 1)
                sanitized[key] = value
        language = gemma4_text.Model.sanitize(self, language)
        for key, value in language.items():
            if ".router." in key:
                value = value.astype(mx.float32)
            sanitized["model.language_model." + key] = value
        return sanitized

    def model_quant_predicate(self, path, module):
        return hasattr(module, "to_quantized") and (
            (path.startswith("model.language_model.") and ".router." not in path)
            or path == "lm_head"
        )

    @property
    def cast_predicate(self):
        return lambda key: ".router." not in key

    @classmethod
    def post_load_hook(cls, model, model_path):
        from transformers import PreTrainedTokenizerFast

        model_path = Path(model_path)
        model._tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        template = model_path / "chat_template.jinja"
        if template.exists():
            model._tokenizer.chat_template = template.read_text()
        generation_config = model_path / "generation_config.json"
        if generation_config.exists():
            eos = json.loads(generation_config.read_text()).get("eos_token_id")
            if eos is not None:
                model._eos_token_ids = set(eos if isinstance(eos, list) else [eos])
        if (
            model._tokenizer.convert_tokens_to_ids("<|audio|>")
            != model.config.audio_token_id
        ):
            raise ValueError("Checkpoint audio token does not match audio_token_id")
        return model

    def _extract_features(self, audio):
        # Whisper normalizes against the maximum over the full padded recording,
        # including across window boundaries; split only after mel extraction.
        windows = max(1, math.ceil(audio.shape[0] / WINDOW_SAMPLES))
        audio = mx.pad(audio, (0, windows * WINDOW_SAMPLES - audio.shape[0]))
        mel = log_mel_spectrogram(audio, n_mels=self.config.audio_config.num_mel_bins)
        return mel.reshape(windows, 3000, -1).transpose(0, 2, 1)

    def get_audio_features(self, input_features):
        outputs = []
        for window in input_features:
            h = self.model.audio_tower(window[None])
            h = h.reshape(
                -1, self.config.audio_config.d_model * self.config.downsample_factor
            )
            h = self.model.embed_audio(h)
            mx.eval(h)
            outputs.append(h)
        return mx.concatenate(outputs, axis=0)

    def _prepare_inputs(self, audio, prompt):
        features = self.get_audio_features(self._extract_features(audio))
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "audio"}],
            }
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        replacement = "<|audio>" + "<|audio|>" * features.shape[0] + "<audio|>"
        if text.count("<|audio|>") != 1:
            raise ValueError(
                "Transcription prompt must contain exactly one audio input"
            )
        ids = self._tokenizer.encode(
            text.replace("<|audio|>", replacement), add_special_tokens=False
        )
        positions = [
            i for i, token in enumerate(ids) if token == self.config.audio_token_id
        ]
        if len(positions) != features.shape[0]:
            raise ValueError("Audio placeholder count does not match encoded audio")
        embeddings = self._embed_text(mx.array([ids]))
        embeddings[:, mx.array(positions), :] = features.astype(embeddings.dtype)
        return embeddings, len(ids)

    def generate(
        self,
        audio: str | Path | np.ndarray | mx.array,
        max_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prompt: str | None = None,
        language: str | None = None,
        hotwords: list[str] | None = None,
        prefill_step_size: int = 512,
        verbose: bool = False,
        generation_stream=None,
    ) -> TranscriptionOutput:
        if max_tokens <= 0 or prefill_step_size <= 0:
            raise ValueError("max_tokens and prefill_step_size must be positive")
        if self._tokenizer is None:
            raise RuntimeError(
                "Load the model with mlx_audio.stt.load before transcribing"
            )
        start = time.perf_counter()
        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio), sr=self.sample_rate)
        else:
            audio = mx.array(audio, dtype=mx.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("Expected a nonempty mono waveform sampled at 16 kHz")
        if audio.size > 1800 * self.sample_rate:
            raise ValueError("SOV-Transcribe-1 supports recordings up to 30 minutes")
        instruction = prompt or DEFAULT_PROMPT
        if language:
            instruction += f"\nThe spoken language is {language}."
        if hotwords:
            terms = merge_hotwords(None, hotwords)
            if terms:
                instruction += f"\nContextual vocabulary: {terms}"
        embeddings, prompt_tokens = self._prepare_inputs(audio, instruction)
        cache = self.make_cache()
        for offset in range(0, prompt_tokens, prefill_step_size):
            logits = self(
                cache=cache,
                input_embeddings=embeddings[:, offset : offset + prefill_step_size],
            )
            mx.eval(logits, [c.state for c in cache])
        prefill_time = time.perf_counter() - start
        sampler = make_sampler(temperature, top_p=top_p)
        tokens = []
        finish_reason = "length"
        for _ in range(max_tokens):
            scores = logits[:, -1, :].astype(mx.float32)
            logprobs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
            token = int(sampler(logprobs).item())
            tokens.append(token)
            if token in self._eos_token_ids:
                finish_reason = "stop"
                break
            if len(tokens) < max_tokens:
                logits = self(mx.array([[token]]), cache=cache)
        text = self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - start
        if verbose:
            print(text)
        return TranscriptionOutput(
            text=text,
            language=language,
            prompt_tokens=prompt_tokens,
            generation_tokens=len(tokens),
            total_tokens=prompt_tokens + len(tokens),
            prompt_tps=prompt_tokens / max(prefill_time, 1e-9),
            generation_tps=len(tokens) / max(elapsed - prefill_time, 1e-9),
            total_time=elapsed,
            finish_reason=finish_reason,
        )


def copy_supporting_files(source, dest):
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        path = Path(source) / name
        if path.exists():
            shutil.copy2(path, Path(dest) / name)


def write_model_card(path, source, model_id, config):
    if Path(source).exists():
        source = "salesforce-sovereign-intelligence/SOV-Transcribe-1"
    bits = config.get("quantization", {}).get("bits")
    precision = f"{bits}-bit" if bits else "BF16"
    (Path(path) / "README.md").write_text(
        "---\nlibrary_name: mlx-audio\npipeline_tag: automatic-speech-recognition\n"
        f"license: apache-2.0\nbase_model: {source}\n"
        "language: [en, fr, de, es, it, pt, nl]\ntags: [mlx, speech-to-text]\n---\n\n"
        f"# SOV-Transcribe-1 ({precision})\n\n"
        "Whisper large-v3 audio encoder and Gemma 4 MoE decoder for transcription "
        "of recordings up to 30 minutes, with natural-language instructions.\n\n"
        "```python\nfrom mlx_audio.stt import load\n\n"
        f'model = load("{model_id}")\n'
        'print(model.generate("audio.wav").text)\n```\n\n'
        f"Model and reference code: [Apache 2.0](https://huggingface.co/{source}#license).\n"
    )
