from typing import List, Union, Optional

import numpy as np
from numpy.lib.stride_tricks import as_strided
import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.utils import resample_audio

from .config import VoiceEncConfig
from .melspec import melspectrogram


def pack(arrays, seq_len: int = None, pad_value=0):
    """
    Given a list of length B of array-like objects of shapes (Ti, ...), packs them in a single tensor of
    shape (B, T, ...) by padding each individual array on the right.

    :param arrays: a list of array-like objects of matching shapes except for the first axis.
    :param seq_len: the value of T. It must be the maximum of the lengths Ti of the arrays at
    minimum. Will default to that value if None.
    :param pad_value: the value to pad the arrays with.
    :return: a (B, T, ...) tensor
    """
    if seq_len is None:
        seq_len = max(len(array) for array in arrays)
    else:
        assert seq_len >= max(len(array) for array in arrays)

    # Convert lists to np.array
    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]

    # Convert to MLX arrays
    if isinstance(arrays[0], mx.array):
        mlx_arrays = arrays
    else:
        mlx_arrays = [mx.array(array) for array in arrays]

    # Fill the packed tensor with the array data
    packed_shape = (len(mlx_arrays), seq_len, *mlx_arrays[0].shape[1:])
    packed_array = mx.full(packed_shape, pad_value, dtype=mlx_arrays[0].dtype)

    for i, array in enumerate(mlx_arrays):
        packed_array[i, : array.shape[0]] = array

    return packed_array


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
):
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(
    overlap: float,
    rate: float,
    hp: VoiceEncConfig,
):
    # Compute how many frames separate two partial utterances
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


def stride_as_partials(
    mel: np.ndarray,
    hp: VoiceEncConfig,
    overlap=0.5,
    rate: float = None,
    min_coverage=0.8,
):
    """
    Takes unscaled mels in (T, M) format
    TODO: doc
    """
    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)

    # Compute how many partials can fit in the mel
    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    # Trim or pad the mel spectrogram to match the number of partials
    if target_len > len(mel):
        mel = np.concatenate((mel, np.full((target_len - len(mel), hp.num_mels), 0)))
    elif target_len < len(mel):
        mel = mel[:target_len]

    # Ensure the numpy array data is float32 and contiguous in memory
    mel = mel.astype(np.float32, order="C")

    # Re-arrange the array in memory to be of shape (N, P, M) with partials overlapping eachother,
    # where N is the number of partials, P is the number of frames of each partial and M the
    # number of channels of the mel spectrograms.
    shape = (n_partials, hp.ve_partial_frames, hp.num_mels)
    strides = (mel.strides[0] * frame_step, mel.strides[0], mel.strides[1])
    partials = as_strided(mel, shape, strides)
    return partials


class VoiceEncoder(nn.Module):
    def __init__(self, hp=VoiceEncConfig()):
        super().__init__()

        self.hp = hp

        lstm_layers = []
        for i in range(3):
            lstm_layers.append(nn.LSTM(hp.num_mels if i == 0 else hp.ve_hidden_size, hp.ve_hidden_size))
        self.lstm = nn.Sequential(*lstm_layers)
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        self.similarity_weight = mx.array([10.0])
        self.similarity_bias = mx.array([-5.0])

    def __call__(self, mels: mx.array):
        """
        Computes the embeddings of a batch of partial utterances.

        :param mels: a batch of unscaled mel spectrograms of same duration as a float32 tensor
        of shape (B, T, M) where T is hp.ve_partial_frames
        :return: the embeddings as a float32 tensor of shape (B, E) where E is
        hp.speaker_embed_size. Embeddings are L2-normed and thus lay in the range [-1, 1].
        """
        if self.hp.normalized_mels and (mels.min() < 0 or mels.max() > 1):
            raise Exception(f"Mels outside [0, 1]. Min={mels.min()}, Max={mels.max()}")

        _, (hidden, _) = self.lstm(mels)
        raw_embeds = self.proj(hidden[-1])

        if self.hp.ve_final_relu:
            raw_embeds = nn.relu(raw_embeds)

        return raw_embeds / mx.linalg.norm(raw_embeds, axis=1, keepdims=True)

    def inference(self, mels: mx.array, mel_lens, overlap=0.5, rate: float = None, min_coverage=0.8, batch_size=None):
        """
        Computes the embeddings of a batch of full utterances with gradients.

        :param mels: (B, T, M) unscaled mels
        :return: (B, E) embeddings as numpy array
        """
        mel_lens = mel_lens.tolist() if isinstance(mel_lens, (mx.array, np.ndarray)) else mel_lens

        # Compute where to split the utterances into partials
        frame_step = get_frame_step(overlap, rate, self.hp)
        n_partials, target_lens = zip(*(get_num_wins(l, frame_step, min_coverage, self.hp) for l in mel_lens))

        # Possibly pad the mels to reach the target lengths
        len_diff = max(target_lens) - mels.shape[1]
        if len_diff > 0:
            pad = mx.full((mels.shape[0], len_diff, self.hp.num_mels), 0, dtype=mx.float32)
            mels = mx.concatenate((mels, pad), axis=1)

        # Group all partials together so that we can batch them easily
        partials = [
            mel[i * frame_step : i * frame_step + self.hp.ve_partial_frames]
            for mel, n_partial in zip(mels, n_partials)
            for i in range(n_partial)
        ]
        assert all(partials[0].shape == partial.shape for partial in partials)
        partials = mx.stack(partials)

        # Forward the partials
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        chunk_size = len(partials) // n_chunks
        partial_embeds = []

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(partials))
            batch = partials[start:end]
            partial_embeds.append(self(batch))

        partial_embeds = mx.concatenate(partial_embeds, axis=0)

        # Reduce the partial embeds into full embeds and L2-normalize them
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = [mx.mean(partial_embeds[start:end], axis=0) for start, end in zip(slices[:-1], slices[1:])]
        raw_embeds = mx.stack(raw_embeds)
        embeds = raw_embeds / mx.linalg.norm(raw_embeds, axis=1, keepdims=True)

        # Convert to numpy for compatibility
        return np.array(embeds)

    @staticmethod
    def utt_to_spk_embed(utt_embeds: np.ndarray):
        """
        Takes an array of L2-normalized utterance embeddings, computes the mean embedding and L2-normalize it to get a
        speaker embedding.
        """
        assert utt_embeds.ndim == 2
        utt_embeds = np.mean(utt_embeds, axis=0)
        return utt_embeds / np.linalg.norm(utt_embeds, 2)

    @staticmethod
    def voice_similarity(embeds_x: np.ndarray, embeds_y: np.ndarray):
        """
        Cosine similarity for L2-normalized utterance embeddings or speaker embeddings
        """
        embeds_x = embeds_x if embeds_x.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_x)
        embeds_y = embeds_y if embeds_y.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_y)
        return embeds_x @ embeds_y

    def embeds_from_mels(self, mels: Union[mx.array, List[np.ndarray]], mel_lens=None, as_spk=False, batch_size=32, **kwargs):
        """
        Convenience function for deriving utterance or speaker embeddings from mel spectrograms.

        :param mels: unscaled mels strictly within [0, 1] as either a (B, T, M) tensor or a list of (Ti, M) arrays.
        :param mel_lens: if passing mels as a tensor, individual mel lengths
        :param as_spk: whether to return utterance embeddings or a single speaker embedding
        :param kwargs: args for inference()

        :returns: embeds as a (B, E) float32 numpy array if <as_spk> is False, else as a (E,) array
        """
        # Load mels in memory and pack them
        if isinstance(mels, List):
            mels = [np.asarray(mel) for mel in mels]
            assert all(m.shape[1] == mels[0].shape[1] for m in mels), "Mels aren't in (B, T, M) format"
            mel_lens = [mel.shape[0] for mel in mels]
            mels = pack(mels)

        # Convert to MLX array if needed
        if not isinstance(mels, mx.array):
            mels = mx.array(mels)

        # Embed them
        utt_embeds = self.inference(mels, mel_lens, batch_size=batch_size, **kwargs)

        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wavs(
        self, wavs: List[np.ndarray], sample_rate, as_spk=False, batch_size=32, trim_top_db: Optional[float] = 20, **kwargs
    ):
        """
        Wrapper around embeds_from_mels

        :param trim_top_db: this argument was only added for the sake of compatibility with metavoice's implementation
        """
        if sample_rate != self.hp.sample_rate:
            wavs = [resample_audio(wav, orig_sr=sample_rate, target_sr=self.hp.sample_rate) for wav in wavs]

        # if trim_top_db:
        #     wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]

        if "rate" not in kwargs:
            kwargs["rate"] = 1.3  # Resemble's default value.

        mels = [melspectrogram(w, self.hp).T for w in wavs]

        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, **kwargs)

    def sanitize(self, weights: dict) -> dict:
        sanitized_weights = {}
        bias_buffer = {}
        for k, v in weights.items():
            if k.startswith("lstm."):
                for layer in ("0", "1", "2"):
                    if k.endswith(f"bias_ih_l{layer}"):
                        bias_buffer[f"l{layer}_ih"] = v
                        break
                    if k.endswith(f"bias_hh_l{layer}"):
                        bias_buffer[f"l{layer}_hh"] = v
                        break

                if "lstm.bias_" in k:
                    continue

                if k.endswith("weight_ih_l0"):
                    k = "lstm.layers.0.Wx"
                elif k.endswith("weight_hh_l0"):
                    k = "lstm.layers.0.Wh"
                elif k.endswith("weight_ih_l1"):
                    k = "lstm.layers.1.Wx"
                elif k.endswith("weight_hh_l1"):
                    k = "lstm.layers.1.Wh"
                elif k.endswith("weight_ih_l2"):
                    k = "lstm.layers.2.Wx"
                elif k.endswith("weight_hh_l2"):
                    k = "lstm.layers.2.Wh"

            sanitized_weights[k] = v

        for layer in ("0", "1", "2"):
            ih = bias_buffer.get(f"l{layer}_ih")
            hh = bias_buffer.get(f"l{layer}_hh")
            if ih is not None and hh is not None:
                summed = ih + hh
                sanitized_weights[f"lstm.layers.{layer}.bias"] = summed
            else:
                raise KeyError(f"Missing bias in layer {layer}: ih={ih is not None}, hh={hh is not None}")

        return sanitized_weights