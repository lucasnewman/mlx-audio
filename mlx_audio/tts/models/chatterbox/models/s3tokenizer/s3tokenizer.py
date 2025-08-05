from typing import List, Tuple, Optional
import numpy as np
import mlx.core as mx

from mlx_audio.codec.models.s3.model_v2 import S3TokenizerV2
from mlx_audio.codec.models.s3.utils import padding, log_mel_spectrogram

# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_N_FFT = 400
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    def pad(self, wavs: List[mx.array], sr: int) -> List[mx.array]:
        """
        Given a list of wavs with the same `sample_rate`, pad them so that the length
        is multiple of 40ms (S3 runs at 25 token/sec).
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = mx.array(wav)
            if wav.ndim == 1:
                wav = mx.expand_dims(wav, axis=0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)

            # Pad to intended length
            pad_amount = intended_wav_len - wav.shape[-1]
            if pad_amount > 0:
                wav = mx.pad(wav, [(0, 0), (0, pad_amount)], constant_values=0)

            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs: List[mx.array]) -> List[mx.array]:
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = mx.array(wav)
            if wav.ndim == 1:
                wav = mx.expand_dims(wav, axis=0)

            processed_wavs.append(wav)
        return processed_wavs

    def forward(
        self,
        wavs: List[mx.array],
        max_len: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Process audio waveforms through the S3 tokenizer.

        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).

        Args
        ----
        - `wavs`: List of 16 kHz speech audio arrays
        - `max_len`: max length to truncate the output sequence to (25 token/sec).
          NOTE: please pad the waveform if longer sequence is needed.

        Returns
        -------
        Tuple of (speech_tokens, speech_token_lens)
        """
        processed_wavs = self._prepare_audio(wavs)
        mels = []

        for wav in processed_wavs:
            mel = log_mel_spectrogram(
                wav.squeeze(0),
                sample_rate=S3_SR,
                n_mels=self.config.n_mels,
                n_fft=S3_N_FFT,
                hop_length=S3_HOP,
            )

            mel = mx.expand_dims(mel, axis=0)  # [1, F, T]

            if max_len is not None:
                mel = mel[..., : max_len * 4]  # num_mel_frames = 4 * num_tokens

            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)

        speech_tokens, speech_token_lens = self.quantize(mels, mel_lens)

        return (
            speech_tokens.astype(mx.int32),
            speech_token_lens.astype(mx.int32),
        )
