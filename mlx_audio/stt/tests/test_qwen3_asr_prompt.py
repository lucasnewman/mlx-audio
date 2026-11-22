import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import jinja2
import numpy as np

from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

# chat_template.json shipped with Qwen/Qwen3-ASR-1.7B and Qwen3-ASR-0.6B.
OFFICIAL_CHAT_TEMPLATE = (
    '{%- set ns = namespace(system_text="") -%}\n'
    "{%- for m in messages -%}\n"
    "  {%- if m.role == 'system' -%}\n"
    "    {%- if m.content is string -%}\n"
    "      {%- set ns.system_text = ns.system_text + m.content -%}\n"
    "    {%- else -%}\n"
    "      {%- for c in m.content -%}\n"
    "        {%- if c.type == 'text' and (c.text is defined) -%}\n"
    "          {%- set ns.system_text = ns.system_text + c.text -%}\n"
    "        {%- endif -%}\n"
    "      {%- endfor -%}\n"
    "    {%- endif -%}\n"
    "  {%- endif -%}\n"
    "{%- endfor -%}\n"
    "\n"
    '{%- set ns2 = namespace(audio_tokens="") -%}\n'
    "{%- for m in messages -%}\n"
    "  {%- if m.content is not string -%}\n"
    "    {%- for c in m.content -%}\n"
    "      {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}\n"
    '        {%- set ns2.audio_tokens = ns2.audio_tokens + "<|audio_start|><|audio_pad|><|audio_end|>" -%}\n'
    "      {%- endif -%}\n"
    "    {%- endfor -%}\n"
    "  {%- endif -%}\n"
    "{%- endfor -%}\n"
    "\n"
    "{{- '<|im_start|>system\\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\\n' -}}\n"
    "{{- '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' -}}\n"
    "{%- if add_generation_prompt -%}\n"
    "{{- '<|im_start|>assistant\\n' -}}\n"
    "{%- endif -%}"
)


def _official_prompt(context, num_audio_tokens):
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": "x.wav"}]},
    ]
    text = jinja2.Template(OFFICIAL_CHAT_TEMPLATE).render(
        messages=messages, add_generation_prompt=True
    )
    # The processor expands the single <|audio_pad|> to one per audio frame.
    return text.replace("<|audio_pad|>", "<|audio_pad|>" * num_audio_tokens)


def _built_prompt(context, language=None, num_audio_tokens=3):
    """The exact text Qwen3ASRModel._build_prompt hands to the tokenizer."""
    model = Qwen3ASRModel.__new__(Qwen3ASRModel)
    model.config = SimpleNamespace(support_languages=["English", "Chinese"])
    model._tokenizer = Mock()
    model._tokenizer.encode = Mock(return_value=np.zeros((1, 1), dtype=np.int32))
    model._build_prompt(num_audio_tokens, language, context)
    return model._tokenizer.encode.call_args.args[0]


class TestQwen3ASRPrompt(unittest.TestCase):
    """The prompt must render exactly like the official chat template, so the
    tokenizer produces the same token ids as the reference implementation."""

    def test_matches_official_template_with_context(self):
        for context in ["Hotwords: MLX, Qwen", "交易 停滞", "line one\nline two"]:
            with self.subTest(context=context):
                self.assertEqual(_built_prompt(context), _official_prompt(context, 3))

    def test_matches_official_template_without_context(self):
        for context in [None, ""]:
            with self.subTest(context=context):
                self.assertEqual(_built_prompt(context), _official_prompt(context, 3))

    def test_forced_language_is_appended_after_the_template(self):
        self.assertEqual(
            _built_prompt("terms", language="chinese"),
            _official_prompt("terms", 3) + "language Chinese<asr_text>",
        )


if __name__ == "__main__":
    unittest.main()
