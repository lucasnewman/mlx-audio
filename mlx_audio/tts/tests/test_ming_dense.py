import re
import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.bailingmm.bailingmm import (
    _MING3_VOICE_RE,
    Model,
    _add_period,
    _is_style_instruction,
    _semantic_length,
    _split_fragments,
    _trim_trailing_silence,
    ming3_instruction,
)

SPECIAL = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|vision_start|>",
    "<|vision_pad|>",
    "<|vision_end|>",
    "<role>",
    "</role>",
    "<spk>",
    "</spk>",
    "<audio>",
    "<audioPatch>",
]
SPECIAL_RE = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL) + ")")


class FakeTokenizer:
    """Specials get their own id; every other character is one token."""

    def __init__(self):
        self.vocab = {tok: i for i, tok in enumerate(SPECIAL)}

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in SPECIAL_RE.split(text):
            if part in self.vocab:
                ids.append(self.vocab[part])
            else:
                ids.extend(len(SPECIAL) + ord(c) for c in part)
        return ids

    def decode(self, ids):
        inv = {i: tok for tok, i in self.vocab.items()}
        return "".join(inv[i] if i in inv else chr(i - len(SPECIAL)) for i in ids)


def make_model(chat_layout, ming3, hidden=8, latent_dim=4):
    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.tokenizer = FakeTokenizer()
    model.chat_layout = chat_layout
    model.ming3 = ming3
    model.patch_size = 1 if ming3 else 4
    model.latent_dim = latent_dim
    model.llm_args = SimpleNamespace(hidden_size=hidden)
    model.model = SimpleNamespace(
        model=SimpleNamespace(word_embeddings=nn.Embedding(70000, hidden))
    )
    model.linear_proj_audio = lambda x: mx.zeros((x.shape[0], hidden))
    return model


PROMPT = "Please generate speech based on the following description.\n"


class TestMingPromptLayout(unittest.TestCase):
    def decode(self, model, **kwargs):
        ids, embeds = model._prepare_input_embed(PROMPT, "Hi.", **kwargs)
        self.assertEqual(embeds.shape[1], ids.shape[1])
        return model.tokenizer.decode(ids[0].tolist()), embeds

    def test_dense_uses_qwen_chat_template(self):
        text, _ = self.decode(make_model(chat_layout=True, ming3=True))
        self.assertEqual(
            text,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + PROMPT + " Text input:\nHi.<|im_end|>\n"
            "<|im_start|>assistant\n<audio>",
        )

    def test_moe_keeps_role_turns(self):
        text, _ = self.decode(make_model(chat_layout=False, ming3=False))
        self.assertTrue(text.startswith("<role>HUMAN</role>" + PROMPT))
        self.assertTrue(text.endswith("Hi.<role>ASSISTANT</role><audio>"))

    def test_instruction_terminator(self):
        ming3, _ = self.decode(
            make_model(chat_layout=True, ming3=True), instruction="Warm."
        )
        old, _ = self.decode(
            make_model(chat_layout=True, ming3=False), instruction="Warm."
        )
        self.assertTrue(ming3.endswith("assistant\nWarm.<|im_end|><audio>"))
        self.assertTrue(old.endswith("assistant\nWarm.<|endoftext|><audio>"))

    def test_speaker_slot_replaces_pad_row(self):
        model = make_model(chat_layout=True, ming3=True)
        emb = mx.full((8,), 7.0)
        text, embeds = self.decode(model, spk_emb=[emb])
        self.assertIn(
            PROMPT + "  speaker_1:<|vision_start|><|vision_pad|><|vision_end|>\n"
            " Text input:\n",
            text,
        )
        ids = model.tokenizer.encode(text)
        pos = ids.index(model.tokenizer.convert_tokens_to_ids("<|vision_pad|>"))
        self.assertTrue(mx.all(embeds[0, pos] == 7.0).item())
        self.assertFalse(mx.all(embeds[0, pos + 1] == 7.0).item())

    def test_prompt_latents_follow_audio_token(self):
        model = make_model(chat_layout=True, ming3=True)
        text, embeds = self.decode(
            model, prompt_text="Ref.", prompt_latent=mx.ones((1, 3, 4))
        )
        self.assertIn(" Text input:\nRef.Hi.<|im_end|>", text)
        self.assertTrue(text.endswith("<audio>" + "<audioPatch>" * 3))

    def test_zero_speaker_embedding_is_projected_zero(self):
        model = make_model(chat_layout=True, ming3=True)
        (emb,) = model._speaker_embeddings(None, use_zero_spk_emb=True)
        self.assertEqual(emb.shape, (8,))
        self.assertTrue(mx.all(emb == 0).item())
        self.assertIsNone(model._speaker_embeddings(None, use_zero_spk_emb=False))

    def test_prompt_padding_side(self):
        wav = mx.ones((1, 100))
        model = make_model(chat_layout=True, ming3=True)
        model.audio = SimpleNamespace(sample_rate=44100)
        padded = model._pad_prompt_waveform(wav)
        self.assertEqual(padded.shape[1], 3528)
        self.assertEqual(padded[0, 0].item(), 0.0)
        self.assertEqual(padded[0, -1].item(), 1.0)

        old = make_model(chat_layout=True, ming3=False)
        old.audio = SimpleNamespace(sample_rate=44100)
        padded = old._pad_prompt_waveform(wav)
        self.assertEqual(padded.shape[1], 14112)
        self.assertEqual(padded[0, 0].item(), 1.0)
        self.assertEqual(padded[0, -1].item(), 0.0)


class TestMing3Instruction(unittest.TestCase):
    def test_ip_caption_matches_reference(self):
        # Output of Ming-3.0-tts tts_tasks.build_task_instruction("IP", ip="US_James").
        self.assertEqual(
            ming3_instruction(ip="US_James"),
            '{"audio_sequence": [{"序号": 1, "说话人": "speaker_1", "方言": null, '
            '"风格": null, "语速": null, "基频": null, "音量": null, "情感": null, '
            '"BGM": {"Genre": null, "Mood": null, "Instrument": null, "Theme": null, '
            '"ENV": null, "SNR": null}, "IP": "US_James"}]}',
        )

    def test_dialect_uses_english_schema(self):
        caption = ming3_instruction(dialect="cantonese")
        self.assertTrue(caption.startswith('{"audio_sequence": [{"id": 1, '))
        self.assertIn('"dialect": "cantonese"', caption)
        self.assertTrue(caption.endswith('"ip": null}]}'))


class TestMingText(unittest.TestCase):
    def test_add_period(self):
        self.assertEqual(_add_period("Hello"), "Hello.")
        self.assertEqual(_add_period("你好"), "你好。")
        self.assertEqual(_add_period("Hello,"), "Hello.")
        self.assertEqual(_add_period("Really?"), "Really?")
        self.assertEqual(_add_period("[Laughter]"), "[Laughter]")

    def test_short_text_is_one_fragment(self):
        self.assertEqual(_split_fragments("One. Two.", 50), ["One. Two."])

    def test_fragments_respect_length_and_keep_text(self):
        text = " ".join(f"Sentence number {i} has five words." for i in range(30))
        fragments = _split_fragments(text, 50)
        self.assertGreater(len(fragments), 1)
        self.assertTrue(all(_semantic_length(f) <= 50 for f in fragments))
        self.assertEqual(" ".join(fragments), text)

    def test_long_sentence_splits_at_commas(self):
        text = ", ".join(["one two three four five"] * 15) + "."
        fragments = _split_fragments(text, 20)
        self.assertTrue(all(_semantic_length(f) <= 20 for f in fragments))
        self.assertEqual(" ".join(fragments), text)

    def test_chinese_fragments_join_without_spaces(self):
        text = "今天天气很好。" * 12
        fragments = _split_fragments(text, 20)
        self.assertGreater(len(fragments), 1)
        self.assertEqual("".join(fragments), text)

    def test_fragments_keep_numbers_abbreviations_and_ellipses(self):
        text = (
            "The ratio is 3.14 and it costs 1,000 dollars in the U.S. today. "
            "Wait... then we go. "
        ) * 6
        fragments = _split_fragments(text, 20)
        self.assertGreater(len(fragments), 1)
        self.assertEqual(" ".join(fragments), " ".join(text.split()))
        joined = " | ".join(fragments)
        self.assertNotIn("3. 14", joined)
        self.assertNotIn("1, 000", joined)
        self.assertNotIn(". . .", joined)

    def test_mixed_chinese_english_fragments_keep_spaces(self):
        text = "我们今天去公园散步。 Then we walked home together slowly. " * 5
        fragments = _split_fragments(text, 15)
        self.assertGreater(len(fragments), 1)
        self.assertEqual(" ".join(fragments), " ".join(text.split()))

    def test_style_instructions_are_not_split(self):
        self.assertFalse(_is_style_instruction(None))
        self.assertTrue(_is_style_instruction("Speak warmly and slowly."))
        self.assertTrue(_is_style_instruction(ming3_instruction(style="温柔的女声")))
        self.assertFalse(_is_style_instruction(ming3_instruction(emotion="高兴")))
        self.assertFalse(_is_style_instruction(ming3_instruction(ip="US_James")))

    def test_voice_names(self):
        for name in ("US_James", "UK_Charlotte", "灵小柔", "灵信远_高兴"):
            self.assertTrue(_MING3_VOICE_RE.match(name), name)
        for name in ("alloy", "af_heart", "conversational_a", "vivian", ""):
            self.assertFalse(_MING3_VOICE_RE.match(name), name)

    def test_trim_trailing_silence_keeps_tail(self):
        sr = 1000
        speech = mx.concatenate([mx.ones((500,)), mx.zeros((2000,))])
        trimmed = _trim_trailing_silence(speech, sr)
        self.assertEqual(trimmed.shape[0], 500 + 300)


if __name__ == "__main__":
    unittest.main()
