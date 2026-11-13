"""Haru: summed text/audio streams, Qwen3.5, and a MeanFlow audio head."""

from copy import deepcopy
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mlx_audio.lm.models.cache import ArraysCache, KVCache
from mlx_audio.lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs
from mlx_audio.sts.models.base import STSOutput

from .codec import VibeVoiceCodec
from .config import ModelConfig
from .head import MeanFlowHead


class PreciseRMSNorm(nn.RMSNorm):
    """Keep the checkpoint's zero-centred gains in float32 after adding one."""

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(
            x.dtype
        )


class AudioProjector(nn.Module):
    def __init__(self, latent_dim, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x):
        return self.fc2(self.norm(self.fc1(x)))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.sample_rate = config.sample_rate
        args = TextModelArgs.from_dict(
            {"hidden_size": config.hidden_size, **deepcopy(config.lm)}
        )
        self.lm = Qwen3_5TextModel(args)
        # Haru's PAD/EOP decisions are sensitive to rounding of these gains.
        self.lm.norm = PreciseRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        for layer in self.lm.layers:
            layer.input_layernorm = PreciseRMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            layer.post_attention_layernorm = PreciseRMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            if not layer.is_linear:
                layer.self_attn.q_norm = PreciseRMSNorm(
                    args.head_dim, eps=args.rms_norm_eps
                )
                layer.self_attn.k_norm = PreciseRMSNorm(
                    args.head_dim, eps=args.rms_norm_eps
                )
        self.model_audio_proj = AudioProjector(config.latent_dim, config.hidden_size)
        self.user_audio_proj = AudioProjector(config.latent_dim, config.hidden_size)
        self.head = MeanFlowHead(config.head)
        self.codec = VibeVoiceCodec(config.codec)
        self.latent_scale = mx.array(1.0)
        self.latent_bias = mx.array(0.0)
        self.tokenizer = None

    @classmethod
    def from_pretrained(cls, path_or_repo, **kwargs):
        from mlx_audio.sts.utils import load_model

        return load_model(path_or_repo, **kwargs)

    @classmethod
    def post_load_hook(cls, model, model_path):
        from tokenizers import Tokenizer

        model.tokenizer = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
        return model

    @property
    def dtype(self):
        return self.model_audio_proj.fc1.weight.dtype

    @property
    def layers(self):
        return self.lm.layers

    def new_cache(self):
        return [
            ArraysCache(size=2) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    def scale_latents(self, raw):
        return (raw.astype(mx.float32) + self.latent_bias) * self.latent_scale

    def unscale_latents(self, normalized):
        return normalized.astype(mx.float32) / self.latent_scale - self.latent_bias

    def embed(
        self,
        tokens,
        model_latents=None,
        user_latents=None,
        model_mask=None,
        user_mask=None,
    ):
        x = self.lm.embed_tokens(tokens)
        for projector, latents, mask in (
            (self.model_audio_proj, model_latents, model_mask),
            (self.user_audio_proj, user_latents, user_mask),
        ):
            if latents is not None:
                y = projector(self.scale_latents(latents).astype(x.dtype))
                if mask is not None:
                    y = y * mask[..., None].astype(y.dtype)
                x = x + y
        return x

    def __call__(self, embeddings, cache=None):
        return self.lm(None, cache=cache, input_embeddings=embeddings)

    def logits(self, hidden):
        return self.lm.embed_tokens.as_linear(hidden).astype(mx.float32)

    def logits_fp32(self, hidden, tokens):
        rows = self.lm.embed_tokens(tokens).astype(mx.float32)
        return hidden.astype(mx.float32) @ rows.T

    def sanitize(self, weights):
        """Accept original PyTorch weights and already converted MLX checkpoints."""
        weights = dict(weights)
        original = any(
            k.startswith("lm.") and k.endswith("conv1d.weight") and v.shape[-1] != 1
            for k, v in weights.items()
        )
        for name, value in list(weights.items()):
            if name.startswith("lm."):
                if name.endswith("conv1d.weight") and value.shape[-1] != 1:
                    weights[name] = value.transpose(0, 2, 1)
                elif name.endswith(".dt_bias"):
                    # Promote a + dt_bias and softplus to FP32, as in Haru's
                    # reference. BF16 decay errors accumulate in the recurrence.
                    # Apply this to existing MLX checkpoints as well.
                    weights[name] = value.astype(mx.float32)
                elif (
                    original
                    and not self.cast_predicate(name)
                    and name.endswith(".weight")
                ):
                    weights[name] = value.astype(mx.float32) + 1.0
            elif (
                name.startswith("codec.")
                and name.endswith(".weight")
                and value.ndim == 3
            ):
                if name.startswith("codec.decoder.up."):
                    weights[name] = value.transpose(1, 2, 0).reshape(-1, value.shape[0])
                else:
                    weights[name] = value.transpose(0, 2, 1).reshape(value.shape[0], -1)
            elif name in {"latent_scale", "latent_bias"}:
                weights[name] = value.astype(mx.float32).reshape(())
        if "head.final.ada.weight" in weights:
            names = [
                f"head.layers.{i}.ada.weight"
                for i in range(self.config.head.num_layers)
            ]
            names.append("head.final.ada.weight")
            weights["head.ada.weight"] = mx.concatenate([weights.pop(k) for k in names])
        return weights

    @staticmethod
    def cast_predicate(name):
        precise = name.startswith("lm.") and (
            name == "lm.norm.weight"
            or name.endswith(
                (
                    "_layernorm.weight",
                    ".q_norm.weight",
                    ".k_norm.weight",
                    ".A_log",
                    ".dt_bias",
                )
            )
        )
        return not precise and name not in {"latent_scale", "latent_bias"}

    @staticmethod
    def model_quant_predicate(path, module):
        # Keep the codec, audio head, and input projectors at checkpoint precision.
        # Include a/b projections: upstream quantizes them in its fused in_proj.
        return path.startswith("lm.")

    def create_session(self, system_prompt=None, reference=None, config=None):
        """Use the neutral default prompt when omitted; an empty string disables it."""
        from .session import DuplexSession

        if self.tokenizer is None:
            raise RuntimeError("Load a tokenizer before starting a Haru session")
        return DuplexSession(self, system_prompt, reference, config)

    def generate(
        self,
        audio=None,
        *,
        ref_audio=None,
        system_prompt=None,
        sample_rate=None,
        max_new_tokens=None,
        tail_seconds=3.0,
        speak_first=False,
        temperature=0.7,
        top_k=25,
        top_p=1.0,
        guidance_scale=1.5,
        meanflow_steps=None,
        seed=None,
    ):
        """Run a duplex conversation over a clip and trailing silence.

        Arrays must be mono at ``sample_rate`` (24 kHz by default); file inputs
        are resampled. ``max_new_tokens`` caps output frames, one per 133 ms.
        Output frame zero precedes the first user frame by one frame interval.
        For live microphone input use ``create_session`` instead.
        Omit ``system_prompt`` for the neutral default, or pass ``""`` for none.
        """
        from .audio import load_audio, prepare_reference
        from .frames import DuplexConfig

        if tail_seconds < 0:
            raise ValueError("tail_seconds must be nonnegative")
        user = (
            np.zeros(0, dtype=np.float32)
            if audio is None
            else load_audio(audio, self.sample_rate, sample_rate)
        )
        voice = (
            None
            if ref_audio is None
            else prepare_reference(
                load_audio(ref_audio, self.sample_rate, sample_rate),
                sample_rate=self.sample_rate,
            )
        )
        hop = self.config.frame_samples
        n_user = (len(user) + hop - 1) // hop
        n_tail = int(np.ceil(tail_seconds * self.config.frame_rate))
        count = 1 + n_user + n_tail
        if max_new_tokens is not None:
            if max_new_tokens < 1:
                raise ValueError("max_new_tokens must be positive")
            count = min(count, max_new_tokens)
        session = self.create_session(
            system_prompt,
            voice,
            DuplexConfig(
                seed=seed,
                max_frames=count,
                speak_first=speak_first,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=guidance_scale,
                meanflow_steps=meanflow_steps,
            ),
        )
        frame = session.start()
        audio_parts, text_parts = [frame.audio], [frame.text]
        for i in range(count - 1):
            chunk = user[i * hop : (i + 1) * hop]
            if len(chunk) < hop:
                chunk = np.pad(chunk, (0, hop - len(chunk)))
            frame = session.step(chunk)
            audio_parts.append(frame.audio)
            text_parts.append(frame.text)
        return STSOutput(
            audio=np.concatenate(audio_parts),
            sample_rate=self.sample_rate,
            text="".join(text_parts).strip(),
        )
