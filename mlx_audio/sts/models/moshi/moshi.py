import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, Generator, Any, List, Tuple
from pathlib import Path
import sentencepiece

from huggingface_hub import snapshot_download

from mlx_audio.sts.models.moshi_backend import models as moshi_models
from mlx_audio.sts.models.moshi_backend import utils as moshi_utils
import rustymimi

@dataclass
class MoshiConfig:
    hf_repo: str = "kyutai/moshiko-mlx-bf16"
    quantized: Optional[int] = None
    
    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

class MoshiSTSModel:
    def __init__(self, config: MoshiConfig):
        self.config = config
        self.lm_config = moshi_models.config_v0_1()
        self.model = moshi_models.Lm(self.lm_config)
        self.model.set_dtype(mx.bfloat16)
        
        if config.quantized is not None:
            group_size = 32 if config.quantized == 4 else 64
            nn.quantize(self.model, bits=config.quantized, group_size=group_size)
            
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self._is_warmed_up = False

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, quantized: Optional[int] = None) -> "MoshiSTSModel":
        if Path(model_name_or_path).exists():
            model_path = Path(model_name_or_path)
        else:
            model_path = Path(snapshot_download(model_name_or_path))
            
        config = MoshiConfig(hf_repo=model_name_or_path, quantized=quantized)
        model = cls(config)
        model.load_weights(str(model_path))
        return model

    def load_weights(self, model_path: str):
        path = Path(model_path)
        
        # Load the LLM weights
        model_file = path / "model.safetensors"
        if not model_file.exists() and self.config.quantized == 4:
            model_file = path / "model.q4.safetensors"
        elif not model_file.exists() and self.config.quantized == 8:
            model_file = path / "model.q8.safetensors"
            
        self.model.load_weights(str(model_file), strict=True)
        self.model.warmup()
        
        # Load the tokenizers
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(str(path / "tokenizer_spm_32k_3.model"))
        self.audio_tokenizer = rustymimi.StreamTokenizer(str(path / "tokenizer-e351c8d8-checkpoint125.safetensors"))

    def warmup_tokenizer(self):
        """Moshi tokenizer runs in background threads and needs warmup to not drop the first frame"""
        import numpy as np
        if not self._is_warmed_up:
            for _ in range(2):
                self.audio_tokenizer.encode(np.zeros(1920, dtype=np.float32))
                self.audio_tokenizer.get_encoded()
                self.audio_tokenizer.decode(np.zeros((1, 8), dtype=np.uint32))
                self.audio_tokenizer.get_decoded()
            self._is_warmed_up = True

    def generate(self, audio_tokens: Optional[mx.array] = None, steps: int = 150) -> Generator[Tuple[Optional[str], Optional[mx.array]], None, None]:
        """
        Generate audio and text streams.
        Args:
            audio_tokens: Input audio tokens from user microphone. If None, assumes silence.
            steps: Number of generation steps to perform.
        Yields:
            Tuple of (text_word, raw_pcm_audio_frame). 
            Note that text_word might be None if no valid word was formed.
            raw_pcm_audio_frame might be None if the decoder is still buffering.
        """
        self.warmup_tokenizer()
        
        gen = moshi_models.LmGen(
            model=self.model,
            max_steps=steps + 5,
            text_sampler=moshi_utils.Sampler(),
            audio_sampler=moshi_utils.Sampler(),
        )
        
        other_cb = self.model.cfg.other_codebooks
        
        for i in range(steps):
            if audio_tokens is None:
                dummy_tokens = mx.full((1, other_cb), gen.zero_token, dtype=mx.int32)
            else:
                dummy_tokens = audio_tokens[i:i+1] # assuming streamed input format

            text_token, model_audio_tokens = gen.step(dummy_tokens)
            mx.eval(text_token, model_audio_tokens)
            
            # 1. Yield text
            tok = text_token.item()
            word = None
            if tok not in [0, 3]: # not <unk> or <pad>
                try:
                    raw_word = self.text_tokenizer.id_to_piece(tok).replace(' ', ' ')
                    if not any(c.isdigit() for c in raw_word):
                        word = raw_word
                except Exception:
                    pass
            
            # 2. Yield audio
            pcm_frame = None
            last_audio = gen.last_audio_tokens()
            if last_audio is not None:
                import numpy as np
                import time
                tokens_np = np.array(last_audio).astype(np.uint32)
                self.audio_tokenizer.decode(tokens_np)
                
                # The Rust stream decoder runs asynchronously, so we poll for the decoded frame
                while True:
                    pcm_data = self.audio_tokenizer.get_decoded()
                    if pcm_data is not None:
                        pcm_frame = mx.array(pcm_data)
                        break
                    time.sleep(0.001)
                    
            yield word, pcm_frame
