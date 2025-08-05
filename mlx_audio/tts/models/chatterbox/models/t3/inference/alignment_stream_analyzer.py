import logging
import mlx.core as mx
from dataclasses import dataclass
from types import MethodType


logger = logging.getLogger(__name__)


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = mx.zeros((0, j - i))
        # self.alignment_bin = mx.zeros((0, j-i))
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        # Store reference to the transformer and layer index
        self.tfmr = tfmr
        self.alignment_layer_idx = alignment_layer_idx
        self.last_aligned_attn = None

        self._patch_attention_layer(tfmr, alignment_layer_idx)

    def _patch_attention_layer(self, tfmr, alignment_layer_idx):
        """
        Patches the forward method of a specific attention layer to collect outputs.
        """
        target_layer = tfmr.layers[alignment_layer_idx].self_attn

        # Store original forward method
        self.original_forward = target_layer.__call__

        # Create patched forward method
        def patched_forward(layer_self, *args, **kwargs):
            # Force output_attentions=True
            kwargs["output_attentions"] = True

            # Call original forward
            output = self.original_forward(*args, **kwargs)

            # Extract attention weights
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # (B, H, N, N)
                # Store mean across heads for first batch
                self.last_aligned_attn = mx.mean(attn_weights[0], axis=0)  # (N, N)

            return output

        # Bind the patched method
        target_layer.__call__ = MethodType(patched_forward, target_layer)

    def unpatch(self):
        """Restore the original forward method"""
        if hasattr(self, "original_forward"):
            target_layer = self.tfmr.layers[self.alignment_layer_idx].self_attn
            target_layer.__call__ = self.original_forward

    def step(self, logits):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        aligned_attn = self.last_aligned_attn  # (N, N)
        i, j = self.text_tokens_slice

        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j]  # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j]  # (1, S)

        # TODO: monotonic masking; could have issue b/c spaces are often skipped.
        # Create mask and apply
        mask_start = self.curr_frame_pos + 1
        if mask_start < A_chunk.shape[1]:
            # Create a mask array
            mask = mx.ones_like(A_chunk)
            mask[:, mask_start:] = 0
            A_chunk = A_chunk * mask

        # Concatenate to alignment history
        self.alignment = mx.concatenate([self.alignment, A_chunk], axis=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = mx.argmax(A_chunk[-1]).item()
        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)  # NOTE: very lenient!
        if not discontinuity:
            self.text_position = cur_text_posn

        # Hallucinations at the start of speech show up as activations at the bottom of the attention maps!
        # To mitigate this, we just wait until there are no activations far off-diagonal in the last 2 tokens,
        # and there are some strong activations in the first few tokens.
        false_start = (not self.started) and (mx.max(A[-2:, -2:]).item() > 0.1 or mx.max(A[:, :4]).item() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Is generation likely complete?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # NOTE: EOS rarely assigned activations, and second-last token is often punctuation, so use last 3 tokens.
        # NOTE: due to the false-start behaviour, we need to make sure we skip activations for the first few tokens.
        if T > 15:
            last_text_token_duration = mx.sum(A[15:, -3:]).item()
        else:
            last_text_token_duration = 0

        # Activations for the final token that last too long are likely hallucinations.
        long_tail = False
        if self.complete and self.completed_at is not None:
            tail_sum = mx.sum(A[self.completed_at :, -3:], axis=0)
            long_tail = mx.max(tail_sum).item() >= 10  # 400ms

        # If there are activations in previous tokens after generation has completed, assume this is a repetition error.
        repetition = False
        if self.complete and self.completed_at is not None and self.completed_at < T:
            max_vals = mx.max(A[self.completed_at :, :-5], axis=1)
            repetition = mx.sum(max_vals).item() > 5

        # If a bad ending is detected, force emit EOS by modifying logits
        # NOTE: this means logits may be inconsistent with latents!
        if long_tail or repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {repetition=}")
            # (±2**15 is safe for all dtypes >= 16bit)
            logits = -(2**15) * mx.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        # Suppress EoS to prevent early termination
        if cur_text_posn < S - 3:  # FIXME: arbitrary
            # Create a copy and modify
            new_logits = mx.array(logits)
            new_logits[..., self.eos_idx] = -(2**15)
            logits = new_logits

        self.curr_frame_pos += 1
        return logits
