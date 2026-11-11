import re
import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Union

import mlx.core as mx
from tqdm import tqdm
from transformers import AutoTokenizer

from mlx_audio.codec.models.snac import SNAC
from mlx_audio.lm.generate import stream_generate
from mlx_audio.lm.models.llama import Model as LlamaModel
from mlx_audio.lm.models.llama import ModelArgs as LlamaModelConfig
from mlx_audio.lm.sample_utils import make_logits_processors, make_sampler
from mlx_audio.utils import load_audio

from ..base import GenerationResult


@dataclass
class ModelConfig(LlamaModelConfig):
    tokenizer_name: str = "mlx-community/orpheus-3b-0.1-ft-bf16"
    sample_rate: int = 24000

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


snac_model = SNAC.from_pretrained("mlx-community/snac_24khz").eval()

CODES_PER_FRAME = 7  # SNAC layers give 1 + 2 + 4 codes per frame
DECODE_WINDOW_FRAMES = 32
DECODE_LEFT_CONTEXT_FRAMES = 2
DECODE_RIGHT_CONTEXT_FRAMES = 2
SEGMENT_MAX_CHARS = 300
SEGMENT_MAX_SENTENCES = 8
MIN_MAX_TOKENS = 1200
MAX_TOKENS_PER_TEXT_TOKEN = 80
DEFAULT_VOICE = "tara"
SENTENCE_END = re.compile(r"""[.!?]["')\]]*\s+|[。！？]""")
CLAUSE_END = re.compile(r"[,;:]\s+|[，；：、]")
WHITESPACE = re.compile(r"\s+")


def last_cut(pattern, window, max_chars, limit=None):
    cuts = [
        match.end()
        for match in pattern.finditer(window)
        if len(window[: match.end()].rstrip()) <= max_chars
    ]
    return cuts[:limit][-1] if cuts else 0


def split_long_segment(
    text, max_chars=SEGMENT_MAX_CHARS, max_sentences=SEGMENT_MAX_SENTENCES
):
    """Split text into pieces of at most max_chars and max_sentences, at sentence ends where it can."""
    segments = []
    rest = text.strip()
    while len(rest) > max_chars or len(SENTENCE_END.findall(rest)) >= max_sentences:
        window = rest[: max_chars + 1]
        cut = (
            last_cut(SENTENCE_END, window, max_chars, max_sentences)
            or last_cut(CLAUSE_END, window, max_chars)
            or last_cut(WHITESPACE, window, max_chars)
            or max_chars
        )
        segments.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        segments.append(rest)
    return segments


def codes_to_layers(code_list):
    """Convert flat code list to layered format for SNAC."""
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))
    return [
        mx.expand_dims(mx.array(layer_1), 0),
        mx.expand_dims(mx.array(layer_2), 0),
        mx.expand_dims(mx.array(layer_3), 0),
    ]


def decode_audio_stream(code_list, start_frame, end_frame):
    """Decode frames [start_frame, end_frame) a window at a time."""
    layers = codes_to_layers(code_list)
    chunks = []
    for start in range(start_frame, end_frame, DECODE_WINDOW_FRAMES):
        audio = snac_model.decode_range(
            layers,
            start,
            min(start + DECODE_WINDOW_FRAMES, end_frame),
            DECODE_LEFT_CONTEXT_FRAMES,
            DECODE_RIGHT_CONTEXT_FRAMES,
        ).squeeze(-1)
        mx.eval(audio)
        mx.clear_cache()
        chunks.append(audio)
    return mx.concatenate(chunks, axis=1)


def decode_audio_from_codes(code_list):
    """Decode a flat code list to audio."""
    return decode_audio_stream(code_list, 0, len(code_list) // CODES_PER_FRAME)


def encode_audio_to_codes(audio):
    audio = audio[None, None, :]

    codes = snac_model.encode(audio)

    layer_1 = codes[0].squeeze(0).tolist()
    layer_2 = codes[1].squeeze(0).tolist()
    layer_3 = codes[2].squeeze(0).tolist()

    code_list = []
    num_groups = len(layer_1)
    for i in range(num_groups):
        code_list.append(layer_1[i])
        code_list.append(layer_2[2 * i] + 4096)
        code_list.append(layer_3[4 * i] + 2 * 4096)
        code_list.append(layer_3[4 * i + 1] + 3 * 4096)
        code_list.append(layer_2[2 * i + 1] + 4 * 4096)
        code_list.append(layer_3[4 * i + 2] + 5 * 4096)
        code_list.append(layer_3[4 * i + 3] + 6 * 4096)

    return mx.array(code_list)[None, :]


class Model(LlamaModel):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.model_type = config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def parse_output(self, input_ids):
        token_to_find = 128257
        token_to_remove = 128258

        mask = input_ids == token_to_find
        indices = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    indices.append((i, j))
        token_indices = [[], []]
        for i, j in indices:
            token_indices[0].append(i)
            token_indices[1].append(j)

        token_indices = mx.array(token_indices)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = int(token_indices[1][-1])
            cropped_tensor = input_ids[:, last_occurrence_idx + 1 :]
        else:
            cropped_tensor = input_ids

        mask = cropped_tensor != token_to_remove

        processed_rows = []

        for row in cropped_tensor:
            # Create a mask and filter manually since boolean indexing isn't supported
            row_list = row.tolist()
            masked_row = mx.array([val for val in row_list if val != token_to_remove])
            processed_rows.append(masked_row)

        code_lists = []

        for row in processed_rows:
            row_length = row.shape[0]
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        return code_lists

    def prepare_zeroprompt(
        self,
        ref_audio: mx.array,
        ref_text: str,
    ):
        """Prepare the reference audio context (zeroprompt) for voice cloning."""
        print(
            "\033[93mWARNING: Audio cloning doesn't work reliably on Orpheus.\033[0m \n"
            "A known issue affecting Torch and MLX versions. \n"
            "Will be fixed once the Canopy labs repo update their code or the model."
        )
        audio_input_ids = encode_audio_to_codes(ref_audio) + 128266
        audio_transcript_ids = self.tokenizer(ref_text, return_tensors="mlx").input_ids

        start_token = mx.array([[128259]], dtype=mx.int64)  # Start of human
        end_tokens = mx.array(
            [[128009, 128260]], dtype=mx.int64
        )  # End of text, End of human
        audio_start_tokens = mx.array([[128261, 128257]], dtype=mx.int64)
        audio_end_tokens = mx.array([[128258, 128262]], dtype=mx.int64)

        # [SOH] [transcript] [EOT EOH] [SOA SOS] [audio_tokens] [EOS EOA]
        zeroprompt = mx.concatenate(
            [
                start_token,
                audio_transcript_ids,
                end_tokens,
                audio_start_tokens,
                audio_input_ids,
                audio_end_tokens,
            ],
            axis=1,
        )
        return zeroprompt

    def prepare_input_ids(
        self,
        prompt: Union[str, List[str]],
        voice: Optional[str] = None,
        zeroprompt: Optional[mx.array] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
    ):
        """Prepare input ids for a single prompt or batch of prompts, optionally with zeroprompt prefix.

        Args:
            prompt: Single prompt string or list of prompts for batch processing
            voice: Voice name to prepend to prompts (e.g., "zoe")
            zeroprompt: Pre-computed zeroprompt array for voice cloning
            ref_audio: Reference audio for voice cloning (alternative to zeroprompt)
            ref_text: Reference text caption for voice cloning (used with ref_audio)

        Returns:
            If ref_audio/ref_text provided: tuple of (input_ids, input_mask)
            Otherwise: input_ids array
        """
        # Compute zeroprompt from ref_audio/ref_text if provided
        return_mask = False
        if ref_audio is not None and ref_text is not None:
            zeroprompt = self.prepare_zeroprompt(ref_audio, ref_text)
            return_mask = True

        # Handle single prompt vs batch
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        all_input_ids = []
        all_lengths = []
        for p in prompts:
            if voice is not None and zeroprompt is None:
                p = f"{voice}: {p}"

            start_token = mx.array([[128259]], dtype=mx.int64)  # Start of human
            end_tokens = mx.array(
                [[128009, 128260]], dtype=mx.int64
            )  # End of text, End of human

            input_ids = self.tokenizer(p, return_tensors="mlx").input_ids
            prompt_input_ids = mx.concatenate(
                [start_token, input_ids, end_tokens], axis=1
            )  # [SOH] [text] [EOT EOH]

            if zeroprompt is not None:
                prompt_input_ids = mx.concatenate(
                    [zeroprompt, prompt_input_ids], axis=1
                )

            all_input_ids.append(prompt_input_ids)
            all_lengths.append(prompt_input_ids.shape[1])

        # If only one prompt, return as-is (keeps original behavior)
        if len(all_input_ids) == 1:
            if return_mask:
                mask = mx.ones_like(all_input_ids[0], dtype=mx.int64)
                return all_input_ids[0], mask
            return all_input_ids[0]

        # Pad to same length and stack for batch
        max_len = max(ids.shape[1] for ids in all_input_ids)
        padded = []
        masks = []
        for ids, length in zip(all_input_ids, all_lengths):
            if ids.shape[1] < max_len:
                padding = mx.zeros((1, max_len - ids.shape[1]), dtype=mx.int64)
                ids = mx.concatenate([ids, padding], axis=1)
            padded.append(ids)
            # Create attention mask: 1 for real tokens, 0 for padding
            mask = mx.concatenate(
                [
                    mx.ones((1, length), dtype=mx.int64),
                    mx.zeros((1, max_len - length), dtype=mx.int64),
                ],
                axis=1,
            )
            masks.append(mask)

        input_ids = mx.concatenate(padded, axis=0)
        if return_mask:
            input_mask = mx.concatenate(masks, axis=0)
            return input_ids, input_mask
        return input_ids

    def generate_result(
        self,
        audio,
        start_time: float,
        token_count: int,
        segment_idx: int,
        is_streaming_chunk: bool = False,
        is_final_chunk: bool = False,
        **kwargs,
    ) -> GenerationResult:
        """Helper to create a GenerationResult from audio."""
        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0 or is_final_chunk, "No audio generated"

        sample_rate = self.config.sample_rate
        audio_duration_seconds = samples / sample_rate

        elapsed_time = time.perf_counter() - start_time
        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        duration_hours = int(audio_duration_seconds // 3600)
        duration_mins = int((audio_duration_seconds % 3600) // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = (
            f"{duration_hours:02d}:{duration_mins:02d}:"
            f"{duration_secs:02d}.{duration_ms:03d}"
        )

        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=sample_rate,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=is_streaming_chunk,
            is_final_chunk=is_final_chunk,
        )

    def generate(
        self,
        text,
        voice: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.8,
        split_pattern: str = "\n",
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        ref_audio: Optional[Union[str, mx.array]] = None,
        ref_text: Optional[str] = None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ):
        voice = voice or DEFAULT_VOICE
        # Load reference audio if provided (handles file paths and mx.array)
        if ref_audio is not None:
            ref_audio = load_audio(ref_audio, sample_rate=self.sample_rate)

        prompt_text = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = [
            segment
            for p in prompt_text.split(split_pattern)
            if p.strip()
            for segment in split_long_segment(p)
        ]

        # Prepare zeroprompt once if voice cloning is requested
        zeroprompt = None
        if ref_audio is not None and ref_text is not None:
            zeroprompt = self.prepare_zeroprompt(ref_audio, ref_text)

        sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", -1))
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.3),
            kwargs.get("repetition_context_size", 20),
        )

        streaming_token_interval = int(streaming_interval * 137.5)

        # Process each prompt segment individually for memory efficiency
        for segment_idx, segment_prompt in enumerate(prompts):
            time_start = time.perf_counter()
            is_last_segment = segment_idx == len(prompts) - 1

            # Prepare input_ids for this segment (with zeroprompt if voice cloning)
            input_ids = self.prepare_input_ids(
                segment_prompt,
                voice,
                zeroprompt,
            )

            text_tokens = len(self.tokenizer(segment_prompt).input_ids)
            segment_max_tokens = (
                max_tokens
                if max_tokens is not None
                else max(MIN_MAX_TOKENS, MAX_TOKENS_PER_TEXT_TOKEN * text_tokens)
            )

            generated_token_count = 0
            yielded_token_count = 0
            decoded_frames = 0
            yielded_final = False
            reached_end_of_speech = False

            # Generate tokens for this segment
            for i, response in enumerate(
                tqdm(
                    stream_generate(
                        self,
                        tokenizer=self.tokenizer,
                        prompt=input_ids.squeeze(0),
                        max_tokens=segment_max_tokens,
                        sampler=sampler,
                        logits_processors=logits_processors,
                    ),
                    total=segment_max_tokens,
                    disable=not verbose,
                    desc=f"Segment {segment_idx + 1}/{len(prompts)}",
                )
            ):
                next_token = mx.array([response.token])
                input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
                generated_token_count += 1

                if i % 50 == 0:
                    mx.clear_cache()

                # Stream partial audio at intervals
                if stream and generated_token_count % streaming_token_interval == 0:
                    code_lists = self.parse_output(input_ids)
                    if code_lists and len(code_lists[0]) > 0:
                        all_codes = code_lists[0]
                        end_frame = len(all_codes) // CODES_PER_FRAME
                        if next_token != 128258:  # End of speech
                            end_frame -= DECODE_RIGHT_CONTEXT_FRAMES

                        if end_frame > decoded_frames:
                            audio = decode_audio_stream(
                                all_codes, decoded_frames, end_frame
                            )[0]

                            if audio.shape[0] > 0:
                                yield self.generate_result(
                                    audio=audio,
                                    start_time=time_start,
                                    token_count=generated_token_count
                                    - yielded_token_count,
                                    segment_idx=segment_idx,
                                    is_streaming_chunk=True,
                                )
                                yielded_token_count = generated_token_count
                                decoded_frames = end_frame
                                time_start = time.perf_counter()

                if next_token == 128258:  # End of speech
                    reached_end_of_speech = True
                    break

            if (
                generated_token_count >= segment_max_tokens
                and not reached_end_of_speech
            ):
                print(
                    f"\033[93mWARNING: Segment {segment_idx + 1} reached max_tokens "
                    f"({segment_max_tokens}) before the end of speech, so its audio "
                    "is cut short.\033[0m"
                )

            # Decode and yield remaining audio for this segment
            code_lists = self.parse_output(input_ids)

            for code_list in code_lists:
                if len(code_list) == 0:
                    continue

                if stream:
                    end_frame = len(code_list) // CODES_PER_FRAME
                    if end_frame > decoded_frames:
                        audio = decode_audio_stream(
                            code_list, decoded_frames, end_frame
                        )[0]

                        if audio.shape[0] > 0:
                            yield self.generate_result(
                                audio=audio,
                                start_time=time_start,
                                token_count=generated_token_count - yielded_token_count,
                                segment_idx=segment_idx,
                                is_streaming_chunk=True,
                                is_final_chunk=is_last_segment,
                            )
                            yielded_final = is_last_segment
                else:
                    audio = decode_audio_from_codes(code_list)[0]

                    if audio.shape[0] > 0:
                        yield self.generate_result(
                            audio=audio,
                            start_time=time_start,
                            token_count=generated_token_count - yielded_token_count,
                            segment_idx=segment_idx,
                        )

            # The last audio may already have gone out as a partial chunk.
            if stream and is_last_segment and not yielded_final:
                yield self.generate_result(
                    audio=mx.zeros((0,), dtype=mx.float32),
                    start_time=time_start,
                    token_count=generated_token_count - yielded_token_count,
                    segment_idx=segment_idx,
                    is_streaming_chunk=True,
                    is_final_chunk=True,
                )

            # Clear cache after each segment to avoid memory buildup
            mx.clear_cache()

    def stream_generate(
        self,
        text: str,
        voice: str = None,
        temperature: float = 0.6,
        top_p: float = 0.8,
        split_pattern: str = "\n",
        max_tokens: Optional[int] = None,
        streaming_interval: float = 2.0,
        ref_audio: Optional[Union[str, mx.array]] = None,
        ref_text: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """
        Stream generate speech from text, yielding audio chunks as they're generated.

        This method generates audio in a streaming fashion, yielding audio chunks
        as soon as they're ready. This allows for lower latency audio playback.

        Args:
            text: Input text to synthesize
            voice: Voice name to use (default: "tara")
            temperature: Sampling temperature (default: 0.6)
            top_p: Nucleus sampling threshold (default: 0.8)
            split_pattern: Pattern to split text into segments (default: "\\n")
            max_tokens: Maximum tokens per segment (default: scaled from the text length)
            streaming_interval: Time interval in seconds between audio chunk yields (default: 2.0)
            ref_audio: Optional reference audio for voice cloning
            ref_text: Optional caption for reference audio (for voice cloning)
            verbose: Whether to show progress bar (default: False)
            **kwargs: Additional arguments (top_k, logit_bias, repetition_penalty, etc.)

        Yields:
            GenerationResult with generated audio chunks and metrics
        """
        yield from self.generate(
            text=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            split_pattern=split_pattern,
            max_tokens=max_tokens,
            verbose=verbose,
            ref_audio=ref_audio,
            ref_text=ref_text,
            stream=True,
            streaming_interval=streaming_interval,
            **kwargs,
        )
