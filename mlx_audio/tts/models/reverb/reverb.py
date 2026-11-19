"""Native MLX inference for Software Mansion's Reverb (EchoModel) checkpoint."""

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..base import GenerationResult
from ..omnivoice.backbone import MLP, Attention, BackboneConfig, RMSNorm
from ..omnivoice.duration import RuleDurationEstimator
from ..omnivoice.generation import iterative_unmask
from ..omnivoice.omnivoice import Model as OmniVoiceModel
from .config import ModelConfig


class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, rank):
        super().__init__()
        self.table = nn.Embedding(vocab_size, rank)
        self.project = nn.Linear(rank, hidden_size, bias=False)

    def __call__(self, tokens):
        return self.project(self.table(tokens))


class PrefixAttention(Attention):
    def __call__(self, x, mask=None, prefix=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        q, k, v = (value.transpose(0, 2, 1, 3) for value in (q, k, v))
        offset = 0 if prefix is None else prefix[0].shape[2]
        q = self.rope(self.q_norm(q), offset=offset)
        k = self.rope(self.k_norm(k), offset=offset)
        if prefix is not None:
            k = mx.concatenate([prefix[0], k], axis=2)
            v = mx.concatenate([prefix[1], v], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out), (k, v)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = PrefixAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x, mask=None, prefix=None):
        attention, kv = self.self_attn(self.input_layernorm(x), mask, prefix)
        h = x + attention
        return h + self.mlp(self.post_attention_layernorm(h)), kv


class Backbone(nn.Module):
    def __init__(self, config, embedding_rank):
        super().__init__()
        self.embed_tokens = FactorizedEmbedding(
            config.vocab_size, config.hidden_size, embedding_rank
        )
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x, mask=None, prefix_cache=None):
        cache = []
        for i, layer in enumerate(self.layers):
            prefix = None if prefix_cache is None else prefix_cache[i]
            x, kv = layer(x, mask, prefix)
            cache.append(kv)
        return self.norm(x), cache


def prefix_attention_mask(length, prefix_length):
    """Prefix queries see the prefix; target queries see the whole sequence."""
    positions = mx.arange(length)
    return (positions[:, None] >= prefix_length) | (positions[None, :] < prefix_length)


class _CachedGeneration:
    """Expose only target positions to OmniVoice's iterative unmasking sampler."""

    def __init__(self, model, prefix_ids, prefix_audio_mask):
        self.model = model
        self.config = model.config
        embeds = model._prepare_embed_inputs(prefix_ids, prefix_audio_mask)
        _, self.cache = model.backbone(embeds)
        mx.eval(self.cache)

    def __call__(self, input_ids, audio_mask):
        embeds = self.model._prepare_embed_inputs(input_ids, audio_mask)
        hidden, _ = self.model.backbone(embeds, prefix_cache=self.cache)
        return self.model._audio_logits(hidden)


class Model(OmniVoiceModel):
    # The CLI must leave file paths intact so reference preprocessing happens once.
    preserve_ref_audio_path = True

    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        if not config.llm_config:
            raise ValueError("Reverb requires llm_config from the checkpoint.")
        llm = dict(config.llm_config)
        llm.setdefault(
            "rope_theta", llm.get("rope_parameters", {}).get("rope_theta", 1e6)
        )
        backbone_config = BackboneConfig(
            **{k: v for k, v in llm.items() if k in BackboneConfig.__dataclass_fields__}
        )
        self.backbone = Backbone(backbone_config, config.embedding_rank)
        self.audio_embeddings = [
            nn.Embedding(config.audio_vocab_size, backbone_config.hidden_size)
            for _ in range(config.num_audio_codebook)
        ]
        self.audio_heads = [
            nn.Linear(backbone_config.hidden_size, config.audio_vocab_size, bias=False)
            for _ in range(config.num_audio_codebook)
        ]
        self._voices = {}

    def _audio_logits(self, hidden):
        return mx.stack([head(hidden) for head in self.audio_heads], axis=2)

    def __call__(self, input_ids, audio_mask, attention_mask=None, prefix_length=None):
        if prefix_length is not None and self.config.prefix_blocking:
            mask = prefix_attention_mask(input_ids.shape[1], prefix_length)
            attention_mask = mask if attention_mask is None else mask & attention_mask
        embeds = self._prepare_embed_inputs(input_ids, audio_mask)
        hidden, _ = self.backbone(embeds, mask=attention_mask)
        return self._audio_logits(hidden)

    @property
    def voices(self):
        return sorted(self._voices)

    def _voice_prompt(self, voice):
        if voice not in self._voices:
            raise ValueError(
                f"Unknown voice {voice!r}. Available voices: {self.voices}"
            )
        data = self._voices[voice]
        # The distributed prompts are flattened [codebook, frame] arrays.
        tokens = (
            mx.array(data["codes"], dtype=mx.int32)
            .reshape(self.config.num_audio_codebook, data["frames"])
            .T
        )
        return tokens, data["transcript"]

    def generate(
        self,
        text,
        voice=None,
        ref_audio=None,
        ref_text=None,
        ref_tokens=None,
        language=None,
        lang_code="en",
        instruct=None,
        duration_s=None,
        gen_duration=None,
        speed=1.0,
        num_steps=16,
        class_temperature=0.0,
        position_temperature=5.0,
        layer_penalty_factor=5.0,
        t_shift=0.1,
        ref_audio_max_duration_s=10.0,
        stream=False,
        **kwargs,
    ):
        """Generate one complete 24 kHz waveform using distilled guidance.

        ``ref_tokens`` has shape [frames, codebooks]. Array references are mono
        waveforms at 24 kHz; file references are resampled automatically.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if stream:
            raise ValueError(
                "Reverb generates a complete waveform; streaming is not supported."
            )
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be positive and finite")
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if not math.isfinite(t_shift) or t_shift <= 0:
            raise ValueError("t_shift must be positive and finite")
        if not self.config.prefix_blocking:
            raise ValueError(
                "This Reverb implementation requires prefix_blocking.enabled."
            )
        if self.config.echo.get("cfg", {}).get("mode", "student") != "student":
            raise ValueError(
                "Only Reverb's distilled CFG student checkpoint is supported."
            )
        if kwargs.get("guidance_scale", 0) != 0 or kwargs.get("cfg_scale", 0) != 0:
            raise ValueError(
                "Reverb's student has distilled guidance; do not apply additional CFG."
            )
        codec = getattr(self, "audio_tokenizer", None)
        text_tokenizer = getattr(self, "text_tokenizer", None)
        if codec is None or text_tokenizer is None:
            raise RuntimeError(
                "Load Reverb with mlx_audio.tts.utils.load before generation."
            )
        if ref_tokens is not None and ref_audio is not None:
            raise ValueError("Pass either ref_tokens or ref_audio, not both.")
        start = time.perf_counter()
        if ref_audio is not None:
            if not ref_text or not ref_text.strip():
                raise ValueError("ref_text is required with ref_audio.")
            if isinstance(ref_audio, (str, Path)):
                from mlx_audio.utils import load_audio

                wav = load_audio(str(ref_audio), sample_rate=self.sample_rate)
            else:
                wav = mx.array(ref_audio)
            wav = wav.squeeze()
            if wav.ndim != 1 or wav.size == 0:
                raise ValueError("ref_audio must be a non-empty mono waveform.")
            # Truncating audio independently of its transcript breaks alignment.
            if wav.size / self.sample_rate > ref_audio_max_duration_s:
                raise ValueError(
                    "Reference exceeds ref_audio_max_duration_s; provide a shorter clip and matching transcript."
                )
            ref_tokens = codec.encode(wav[None, :, None])[0]
        elif ref_tokens is None:
            ref_tokens, voice_text = self._voice_prompt(voice or "standard_male")
            if ref_text is None:
                ref_text = voice_text
        if ref_tokens is None or not ref_text or not ref_text.strip():
            raise ValueError("A reference prompt and its transcript are required.")
        ref_tokens = mx.array(ref_tokens)
        if (
            ref_tokens.ndim != 2
            or ref_tokens.shape[1] != self.config.num_audio_codebook
            or not ref_tokens.shape[0]
        ):
            raise ValueError("ref_tokens must have shape [frames, num_audio_codebook].")
        if not mx.issubdtype(ref_tokens.dtype, mx.integer):
            raise ValueError("ref_tokens must contain integer codec tokens.")
        if bool(mx.any((ref_tokens < 0) | (ref_tokens >= self.config.audio_mask_id))):
            raise ValueError("Reference codec tokens are outside the valid vocabulary.")
        duration_s = gen_duration if duration_s is None else duration_s
        if duration_s is None:
            frames = (
                RuleDurationEstimator().estimate_duration(
                    text, ref_text, ref_tokens.shape[0]
                )
                / speed
            )
        else:
            if not math.isfinite(duration_s) or duration_s <= 0:
                raise ValueError("duration_s must be positive and finite")
            frames = duration_s * (self.sample_rate / 960) / speed
        target_length = max(1, math.ceil(frames))
        style_ids, text_ids = self._tokenize_style_and_text(
            text,
            language=language or lang_code or "None",
            instruct=instruct or "None",
            text_tokenizer=text_tokenizer,
            denoise=True,
            ref_text=ref_text,
        )
        length = len(style_ids) + len(text_ids) + ref_tokens.shape[0] + target_length
        if length > self.config.llm_config.get("max_position_embeddings", 40960):
            raise ValueError(
                "Text, reference and target exceed the model context length."
            )
        inputs = self._prepare_inference_inputs(
            style_ids, text_ids, target_length, ref_tokens
        )
        prefix_length = length - target_length
        sampler = _CachedGeneration(
            self,
            inputs["input_ids"][:, :prefix_length],
            inputs["audio_mask"][:, :prefix_length],
        )
        tokens = iterative_unmask(
            sampler,
            inputs["input_ids"][:, prefix_length:],
            inputs["audio_mask"][:, prefix_length:],
            T=target_length,
            num_steps=num_steps,
            guidance_scale=0.0,
            class_temperature=class_temperature,
            position_temperature=position_temperature,
            layer_penalty_factor=layer_penalty_factor,
            t_shift=t_shift,
        )
        audio = codec.decode(tokens).astype(mx.float32).reshape(-1)
        mx.eval(audio)
        elapsed = time.perf_counter() - start
        samples = audio.size
        seconds = samples / self.sample_rate
        whole = int(seconds)
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=target_length,
            audio_duration=f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}.{int(seconds % 1 * 1000):03d}",
            real_time_factor=seconds / elapsed,
            prompt={"tokens": prefix_length, "tokens-per-sec": prefix_length / elapsed},
            audio_samples={"samples": samples, "samples-per-sec": samples / elapsed},
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def generate_batch(self, text, **kwargs):
        # Each request has a different immutable prefix cache; process sequentially.
        texts = text if isinstance(text, list) else [text]
        results = []
        for i, item in enumerate(texts):
            item_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list):
                    if len(value) != len(texts):
                        raise ValueError(f"{key} must have one value per text.")
                    value = value[i]
                item_kwargs[key] = value
            result = next(self.generate(text=item, **item_kwargs))
            result.segment_idx = i
            results.append(result)
        return results

    @staticmethod
    def post_load_hook(model, model_path):
        from transformers import Qwen2Tokenizer

        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer

        model.text_tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
        model.audio_tokenizer = HiggsAudioTokenizer.from_pretrained(str(model_path))
        model.audio_tokenizer.eval()
        for path in sorted((Path(model_path) / "voices").glob("*.json")):
            model._voices[path.stem] = json.loads(path.read_text())
        return model
