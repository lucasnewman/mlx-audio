import logging
from typing import Union, Optional, List, Tuple

from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.llama import LlamaModel, ModelArgs as LlamaConfig

from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
# from .inference.t3_hf_backend import T3HuggingfaceBackend
from ..utils import AttrDict


logger = logging.getLogger(__name__)


def _ensure_BOT_EOT(text_tokens: mx.array, hp):
    B = text_tokens.shape[0]
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=T3Config()):
        super().__init__()
        self.hp = hp
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False
    
    def sanitize(self, weights: dict):
        sanitized_weights = {}
        for k, v in weights.items():
            sanitized_weights[k] = v
        return sanitized_weights
    
    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)
    
    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        speech_tokens: mx.array,
        cfg_weight: float = 0.0,
    ) -> Tuple[mx.array, int]:
        """
        Prepare input embeddings by combining conditioning, text, and speech embeddings.
        
        Args:
            t3_cond: T3Cond object with conditioning information
            text_tokens: Integer array of text tokens
            speech_tokens: Integer array of speech tokens  
            cfg_weight: Classifier-free guidance weight
            
        Returns:
            Tuple of (combined embeddings, length of conditioning)
        """
        # prepare input embeddings (skip backbone transformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        
        if cfg_weight > 0.0:
            # CFG uncond - zero out the second batch element
            B, L, D = text_emb.shape
            if B > 1:
                mask = mx.ones((B, 1, 1))
                mask[1] = 0
                text_emb = text_emb * mask
        
        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        
        len_cond = cond_emb.shape[1]
        
        if cond_emb.shape[0] != text_emb.shape[0]:
            repeat_times = text_emb.shape[0] // cond_emb.shape[0]
            cond_emb = mx.repeat(cond_emb, repeat_times, axis=0)
        
        # Concatenate embeddings
        embeds = mx.concatenate([cond_emb, text_emb, speech_emb], axis=1)  # (B, length, dim)
        
        return embeds, len_cond
    
def __call__(
    self,
    *,
    t3_cond: T3Cond,
    text_tokens: mx.array,
    text_token_lens: mx.array,
    speech_tokens: mx.array,
    speech_token_lens: mx.array,
    training=False,
):
    """
    Optimized forward pass using list comprehension instead of loops.
    """
    _ensure_BOT_EOT(text_tokens, self.hp)
    
    # prepare custom input embeds
    embeds, len_cond = self.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_tokens,
    )
    
    # backbone transformer forward
    tfmr_out = self.tfmr(
        input_ids=None,
        inputs_embeds=embeds,
        output_hidden_states=True,
        return_dict=True,
        use_cache=(not training),
    )
    
    # Extract hidden states
    hidden_states = tfmr_out.hidden_states[-1] if hasattr(tfmr_out, 'hidden_states') else tfmr_out[-1]
    
    # Setup dimensions
    B, _, dim = hidden_states.shape
    len_text = text_tokens.shape[1]
    len_speech = speech_tokens.shape[1]
    dtype = hidden_states.dtype
    
    text_latents_list = []
    speech_latents_list = []
    
    for i in range(B):
        ttl_i = int(text_token_lens[i])
        stl_i = int(speech_token_lens[i])
        
        # Extract text latents
        text_end = len_cond + ttl_i
        text_latent_i = mx.zeros((len_text, dim), dtype=dtype)
        text_values = hidden_states[i, len_cond:text_end]
        # Pad or truncate as needed
        text_latent_i = mx.concatenate([
            text_values,
            mx.zeros((len_text - ttl_i, dim), dtype=dtype)
        ], axis=0)[:len_text]
        text_latents_list.append(text_latent_i)
        
        # Extract speech latents
        speech_start = len_cond + len_text
        speech_end = speech_start + stl_i
        speech_latent_i = mx.zeros((len_speech, dim), dtype=dtype)
        speech_values = hidden_states[i, speech_start:speech_end]
        # Pad or truncate as needed
        speech_latent_i = mx.concatenate([
            speech_values,
            mx.zeros((len_speech - stl_i, dim), dtype=dtype)
        ], axis=0)[:len_speech]
        speech_latents_list.append(speech_latent_i)
    
    text_latents = mx.stack(text_latents_list)
    speech_latents = mx.stack(speech_latents_list)
    
    text_logits = self.text_head(text_latents)
    speech_logits = self.speech_head(speech_latents)
    
    return AttrDict(
        text_logits=text_logits,
        text_latents=text_latents,
        speech_logits=speech_logits,
        speech_latents=speech_latents,
        hidden_states=hidden_states,
    )
