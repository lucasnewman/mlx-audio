from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.llama import ModelArgs as LlamaConfig, LlamaModel

@dataclass
class T3ModelOutput:
    """Output format for T3 model forward pass."""
    logits: mx.array
    past_key_values: Optional[Any] = None
    hidden_states: Optional[list] = None
    attentions: Optional[list] = None


class T3Backend(nn.Module):
    """
    MLX implementation of T3 backend with custom embedding/logit layers
    for speech generation using a Llama model.
    """

    def __init__(
        self,
        config: LlamaConfig,
        llama: LlamaModel,
        *,
        speech_enc,
        speech_head,
        latents_queue=None,
        logits_queue=None,
        alignment_stream_analyzer=None,
    ):
        super().__init__()
        self.config = config
        self.model = llama
        self.speech_enc = speech_enc
        self.speech_head = speech_head
        self._added_cond = False
        self.alignment_stream_analyzer = alignment_stream_analyzer
        
        # Store queues if provided
        self.latents_queue = latents_queue
        self.logits_queue = logits_queue

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        decoder_cond: Optional[mx.array] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Union[T3ModelOutput, tuple]:
        """
        Forward pass that combines embedding preparation and model forward.
        
        Args:
            input_ids: (B, S) int32 array of input tokens
            inputs_embeds: (B, S, C) float32 array of input embeddings (if provided, input_ids is ignored)
            decoder_cond: (B, T, C) float32 array of conditioning (prefixed to inputs_embeds)
            past_key_values: Cached key-value pairs from previous forward passes
            use_cache: Whether to use/return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return T3ModelOutput object or tuple
        """
        
        # Prepare inputs (logic from prepare_inputs_for_generation)
        if inputs_embeds is None:
            assert input_ids is not None, "Either input_ids or inputs_embeds must be provided"
            
            # Make use of the kv cache: only the last input ID is new
            if use_cache and past_key_values is not None:
                input_ids = input_ids[:, -1:]
            
            # Custom speech token embedding layer
            inputs_embeds = self.speech_enc(input_ids)
        
        # Prefix decoder conditioning if applicable
        if decoder_cond is not None and not self._added_cond:
            # Should be first step if we're adding conditioning
            if past_key_values is not None and use_cache:
                # In MLX, we might need to handle this differently
                # For now, we'll add conditioning only on first forward
                pass
            
            # Expand conditioning to match batch size if needed
            if decoder_cond.shape[0] != inputs_embeds.shape[0]:
                repeat_times = inputs_embeds.shape[0] // decoder_cond.shape[0]
                decoder_cond = mx.repeat(decoder_cond, repeat_times, axis=0)
            
            # Concatenate conditioning with inputs
            inputs_embeds = mx.concatenate([decoder_cond, inputs_embeds], axis=1)
            self._added_cond = True
        
        # Forward through Llama model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            cache=past_key_values if use_cache else None,
        )
        
        # Handle different possible output formats from MLX Llama
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            new_cache = outputs[1] if len(outputs) > 1 and use_cache else None
            attentions = outputs[2] if len(outputs) > 2 and output_attentions else None
        else:
            # Assume it's the hidden states directly
            hidden_states = outputs
            new_cache = None
            attentions = None
        
        # Get the final layer's hidden states
        if isinstance(hidden_states, list):
            final_hidden = hidden_states[-1]
        else:
            final_hidden = hidden_states
        
        # Apply speech head to get logits
        logits = self.speech_head(final_hidden)
        
        # Apply alignment stream analyzer if available
        if self.alignment_stream_analyzer is not None:
            logits = self.alignment_stream_analyzer.step(logits)
        
        # Store in queues if provided
        if self.latents_queue is not None:
            self.latents_queue.put(final_hidden)
        if self.logits_queue is not None:
            self.logits_queue.put(logits)
        
        if return_dict:
            return T3ModelOutput(
                logits=logits,
                past_key_values=new_cache,
                hidden_states=[hidden_states] if not isinstance(hidden_states, list) else hidden_states,
                attentions=attentions,
            )
        else:
            return (logits, new_cache, hidden_states, attentions)
    
    def generate(
        self,
        input_ids: mx.array,
        decoder_cond: mx.array,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> mx.array:
        """
        Simple autoregressive generation loop.
        
        Args:
            input_ids: Starting token IDs
            decoder_cond: Conditioning to prefix
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs
        """
        self.reset_conditioning()
        
        # Initialize
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self(
                input_ids=generated,
                decoder_cond=decoder_cond,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k if specified
            if top_k is not None:
                # Get top k values and indices
                top_k_values, top_k_indices = mx.topk(next_token_logits, k=top_k, axis=-1)
                
                # Create mask for non-top-k values
                mask = mx.ones_like(next_token_logits) * float('-inf')
                # This is a bit tricky in MLX, might need a different approach
                # For now, simplified sampling
                probs = mx.softmax(top_k_values, axis=-1)
                next_token_idx = mx.random.categorical(probs)
                next_token = top_k_indices[mx.arange(top_k_indices.shape[0]), next_token_idx]
            else:
                # Sample from full distribution
                probs = mx.softmax(next_token_logits, axis=-1)
                next_token = mx.random.categorical(probs)
            
            # Append to generated sequence
            next_token = mx.expand_dims(next_token, axis=1)
            generated = mx.concatenate([generated, next_token], axis=1)
            
            # Update cache
            past_key_values = outputs.past_key_values
            
            # Check for EOS
            if eos_token_id is not None and mx.any(next_token == eos_token_id):
                break
        
        return generated
    
    def reset_conditioning(self):
        """Reset the conditioning flag for new generation."""
        self._added_cond = False
