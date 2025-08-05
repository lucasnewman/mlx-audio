from dataclasses import dataclass
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .perceiver import Perceiver
from .t3_config import T3Config


@dataclass
class T3Cond:
    """
    Dataclass container for most / all conditioning info.
    TODO: serialization methods aren't used, keeping them around for convenience
    """
    speaker_emb: mx.array
    clap_emb: Optional[mx.array] = None
    cond_prompt_speech_tokens: Optional[mx.array] = None
    cond_prompt_speech_emb: Optional[mx.array] = None
    emotion_adv: Optional[mx.array] = 0.5
    
    def to(self, *, dtype=None):
        """Cast to a dtype. MLX doesn't have explicit device management."""
        for k, v in self.__dict__.items():
            if isinstance(v, mx.array):
                # Check if it's a floating point type
                is_fp = v.dtype in [mx.float16, mx.float32, mx.bfloat16]
                if is_fp and dtype is not None:
                    setattr(self, k, v.astype(dtype))
        return self
    
    def save(self, fpath):
        """Save conditioning to file using MLX's save function."""
        # Convert to dict of numpy arrays for saving
        save_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, mx.array):
                save_dict[k] = np.array(v)
            else:
                save_dict[k] = v
        
        # MLX uses mx.save_safetensors or we can use numpy
        np.savez(fpath, **save_dict)
    
    @staticmethod
    def load(fpath):
        """Load conditioning from file."""
        # Load numpy arrays and convert to MLX
        data = np.load(fpath, allow_pickle=True)
        kwargs = {}
        
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.shape != ():
                # Convert numpy arrays to MLX arrays
                kwargs[k] = mx.array(v)
            elif isinstance(v, np.ndarray) and v.shape == ():
                # Handle scalar values
                kwargs[k] = v.item()
            else:
                kwargs[k] = v
        
        return T3Cond(**kwargs)


class T3CondEnc(nn.Module):
    """
    Handle all non-text conditioning, like speaker embeddings / prompts, CLAP, emotion, etc.
    """
    
    def __init__(self, hp: T3Config):
        super().__init__()
        self.hp = hp
        
        if hp.encoder_type == "voice_encoder":
            self.spkr_enc = nn.Linear(hp.speaker_embed_size, hp.n_channels)
        else:
            raise NotImplementedError(str(hp.encoder_type))
        
        # emotion adv
        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)
        
        # perceiver resampler
        self.perceiver = None
        if hp.use_perceiver_resampler:
            self.perceiver = Perceiver()
    
    def __call__(self, cond: T3Cond):
        # Validate
        assert (cond.cond_prompt_speech_tokens is None) == (cond.cond_prompt_speech_emb is None), \
            "no embeddings for cond_prompt_speech_tokens"
        
        # Speaker embedding projection
        cond_spkr = self.spkr_enc(cond.speaker_emb.reshape(-1, self.hp.speaker_embed_size))
        cond_spkr = mx.expand_dims(cond_spkr, axis=1)  # (B, 1, dim)
        
        empty = mx.zeros_like(cond_spkr[:, :0])  # (B, 0, dim)
        
        # TODO: CLAP
        assert cond.clap_emb is None, "clap_embed not implemented"
        cond_clap = empty  # (B, 0, dim)
        
        # Cond prompt
        cond_prompt_speech_emb = cond.cond_prompt_speech_emb
        if cond_prompt_speech_emb is None:
            cond_prompt_speech_emb = empty  # (B, 0, dim)
        elif self.hp.use_perceiver_resampler:
            cond_prompt_speech_emb = self.perceiver(cond_prompt_speech_emb)
        
        # Emotion Adv: must provide a value if this model uses emotion conditioning
        cond_emotion_adv = empty  # (B, 0, dim)
        if self.hp.emotion_adv:
            assert cond.emotion_adv is not None
            # Handle both scalar and array inputs
            if isinstance(cond.emotion_adv, (int, float)):
                emotion_val = mx.array([[cond.emotion_adv]])
            else:
                emotion_val = cond.emotion_adv.reshape(-1, 1, 1)
            cond_emotion_adv = self.emotion_adv_fc(emotion_val)
        
        # Concat and return
        cond_embeds = mx.concatenate([
            cond_spkr,
            cond_clap,
            cond_prompt_speech_emb,
            cond_emotion_adv,
        ], axis=1)
        
        return cond_embeds
