import logging
import random
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .utils.mask import make_pad_mask
from .configs import CFM_PARAMS


def normalize(x, axis=-1, eps=1e-12):
    """L2 normalization along specified axis"""
    norm = mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


def interpolate_nearest(x, target_shape):
    """Nearest neighbor interpolation for MLX
    Args:
        x: Input array of shape (batch, channels, length)
        target_shape: Target shape (height, width) - we only use width for 1D
    Returns:
        Interpolated array
    """
    batch, channels, length = x.shape
    target_length = target_shape[1]

    # Calculate indices for nearest neighbor sampling
    indices = mx.arange(target_length) * (length / target_length)
    indices = mx.floor(indices).astype(mx.int32)
    indices = mx.clip(indices, 0, length - 1)

    # Gather values using the indices
    output = mx.take(x, indices, axis=2)

    return output


class MaskedDiffWithXvec(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        encoder: nn.Module = None,
        length_regulator: nn.Module = None,
        decoder: nn.Module = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": CFM_PARAMS,
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def __call__(
        self,
        batch: dict,
    ) -> Dict[str, Optional[mx.array]]:
        token = batch["speech_token"]
        token_len = batch["speech_token_len"]
        feat = batch["speech_feat"]
        feat_len = batch["speech_feat_len"]
        embedding = batch["embedding"]

        # xvec projection
        embedding = normalize(embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = mx.expand_dims((~make_pad_mask(token_len)).astype(mx.float32), axis=-1)
        token = self.input_embedding(mx.clip(token, 0, None)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = mx.zeros(feat.shape)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = mx.transpose(conds, (0, 2, 1))

        mask = (~make_pad_mask(feat_len)).astype(h.dtype)
        feat_interpolated = interpolate_nearest(mx.expand_dims(feat, axis=1), target_shape=(1, h.shape[1]))
        feat_interpolated = mx.squeeze(feat_interpolated, axis=1)

        loss, _ = self.decoder.compute_loss(
            mx.transpose(feat_interpolated, (0, 2, 1)), mx.expand_dims(mask, axis=1), mx.transpose(h, (0, 2, 1)), embedding, cond=conds
        )
        return {"loss": loss}

    def inference(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, flow_cache):
        # Note: fp16 handling removed as MLX handles precision differently

        assert token.shape[0] == 1
        # xvec projection
        embedding = normalize(embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).expand_dims(-1).astype(embedding.dtype)
        token = self.input_embedding(mx.clip(token, 0, None)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = mx.zeros([1, mel_len1 + mel_len2, self.output_size], dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = mx.transpose(conds, (0, 2, 1))

        mask = (~make_pad_mask(mx.array([mel_len1 + mel_len2]))).astype(h.dtype)
        feat, flow_cache = self.decoder(
            mu=mx.transpose(h, (0, 2, 1)),
            mask=mx.expand_dims(mask, axis=1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            flow_cache=flow_cache,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.astype(mx.float32), flow_cache


class CausalMaskedDiffWithXvec(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": CFM_PARAMS,
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        self.fp16 = False

    def inference(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, finalize):
        assert token.shape[0] == 1
        # xvec projection
        embedding = normalize(embedding, axis=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token = mx.concatenate([prompt_token, token], axis=1)
        token_len = prompt_token_len + token_len
        mask = mx.expand_dims((~make_pad_mask(token_len)), axis=-1).astype(embedding.dtype)
        token = self.input_embedding(mx.clip(token, 0, None)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = mx.zeros([1, mel_len1 + mel_len2, self.output_size], dtype=h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = mx.transpose(conds, (0, 2, 1))

        mask = (~make_pad_mask(mx.array([mel_len1 + mel_len2]))).astype(h.dtype)
        feat, _ = self.decoder(mu=mx.transpose(h, (0, 2, 1)), mask=mx.expand_dims(mask, axis=1), spks=embedding, cond=conds, n_timesteps=10)
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.astype(mx.float32), None
