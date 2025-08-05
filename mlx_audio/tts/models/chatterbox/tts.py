from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import soundfile as sf

from .models.t3.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen, drop_invalid_tokens
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

from mlx_audio.stt.utils import resample_audio


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """

    t3: T3Cond
    gen: dict

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        # torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        # if isinstance(map_location, str):
        #     map_location = torch.device(map_location)
        # kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        # return cls(T3Cond(**kwargs['t3']), kwargs['gen'])
        return None


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        conds: Conditionals | None = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.conds = conds

    @classmethod
    def from_local(cls, ckpt_dir) -> "ChatterboxTTS":
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve_weights = mx.load((ckpt_dir / "ve.safetensors").as_posix(), format="safetensors")
        ve_weights = ve.sanitize(ve_weights)
        ve.load_weights(list(ve_weights.items()))
        mx.eval(ve.parameters())

        t3 = T3()
        t3_state = mx.load((ckpt_dir / "t3_cfg.safetensors").as_posix(), format="safetensors")
        t3_weights = t3.sanitize(t3_state)
        t3.load_weights(list(t3_weights.items()))
        mx.eval(t3.parameters())

        s3gen = S3Gen()
        s3gen_weights = mx.load((ckpt_dir / "s3gen.safetensors").as_posix(), format="safetensors")
        s3gen_weights = s3gen.sanitize(s3gen_weights)
        s3gen.load_weights(list(s3gen_weights.items()))
        mx.eval(s3gen.parameters())

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice)

        return cls(t3, s3gen, ve, tokenizer, conds=conds)

    @classmethod
    def from_pretrained(cls) -> "ChatterboxTTS":
        local_path = snapshot_download(repo_id=REPO_ID, allow_patterns=["*.safetensors", "*.json", "conds.pt"])
        assert Path(local_path).exists(), f"File {local_path} not found in the hub."
        return cls.from_local(Path(local_path))

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _ = sf.read(wav_fpath)

        ref_16k_wav = resample_audio(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            if t3_cond_prompt_tokens.ndim == 1:
                t3_cond_prompt_tokens = t3_cond_prompt_tokens[None, :]

        # Voice-encoder speaker embedding
        ve_embed = mx.array(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdims=True)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * mx.ones(1, 1, 1),
        )
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * mx.ones(1, 1, 1),
            )

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text)

        if cfg_weight > 0.0:
            text_tokens = mx.concat([text_tokens, text_tokens], axis=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = mx.pad(text_tokens, (1, 0), constant_values=sot)  # Add start token
        text_tokens = mx.pad(text_tokens, (0, 1), constant_values=eot)  # Add end token

        speech_tokens = self.t3.inference(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,  # TODO: use the value in config
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        # Extract only the conditional batch.
        speech_tokens = speech_tokens[0]

        # TODO: output becomes 1D
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
        )
        wav = wav.squeeze(0)
        return mx.array(wav)[None, ...]
