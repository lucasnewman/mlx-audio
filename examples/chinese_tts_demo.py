#!/usr/bin/env python3
"""
Kokoro TTS Demo

This script generates TTS audio using the Kokoro model.
It demonstrates the same functionality as the Swift fix in PR fix/chinese-tts-swift.

Usage:
    python examples/chinese_tts_demo.py [--chinese]

Output (English mode):
    - /tmp/english_hello.wav
    - /tmp/english_test.wav

Output (Chinese mode, with --chinese flag):
    - /tmp/chinese_hello.wav
    - /tmp/chinese_test.wav
    - /tmp/chinese_poem.wav
"""

import sys
import os
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlx_audio.tts.utils import load_model
from mlx_audio.audio_io import write as audio_write

# Global model cache
_model = None

def get_model(model_path: str):
    """Load model once and cache it."""
    global _model
    if _model is None:
        print(f"Loading Kokoro model from HuggingFace: {model_path}")
        _model = load_model(model_path)
        # Set repo_id so voices are loaded from the correct repo
        _model.repo_id = model_path
        print("Model loaded!")
    return _model


def generate_tts_wav(text: str, output_path: str, voice: str, lang_code: str, model_path: str):
    """Generate TTS and save to WAV file."""
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print(f"Language: {lang_code}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    model = get_model(model_path)

    print("Generating audio...")

    # Generate audio using model's generate method
    audio_samples = []
    for result in model.generate(
        text=text,
        voice=voice,
        lang_code=lang_code,
        speed=1.0,
        verbose=False,
    ):
        audio_samples.append(np.array(result.audio))

    # Concatenate all audio chunks
    audio = np.concatenate(audio_samples) if audio_samples else np.array([])

    if len(audio) == 0:
        print("Error: No audio generated")
        return None

    # Save to WAV
    audio_write(output_path, audio, 24000)
    print(f"Saved to: {output_path}")
    print(f"  Duration: {len(audio) / 24000:.2f}s")
    print(f"  Play with: afplay {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS Demo")
    parser.add_argument("--chinese", action="store_true", help="Use Chinese model and examples")
    args = parser.parse_args()

    print("\n" + "="*60)
    if args.chinese:
        print("Chinese Kokoro TTS Demo")
        model_path = "FluidInference/kokoro-82m-v1.1-zh-mlx"
        examples = [
            {
                "text": "你好，世界！",
                "output": "/tmp/chinese_hello.wav",
                "voice": "zf_003",  # Female voice (xiaoxiao)
                "lang_code": "z",
                "description": "Hello World"
            },
            {
                "text": "这是一项中文语音合成测试。",
                "output": "/tmp/chinese_test.wav",
                "voice": "zf_003",
                "lang_code": "z",
                "description": "Test sentence"
            },
            {
                "text": "床前明月光，疑是地上霜。",
                "output": "/tmp/chinese_poem.wav",
                "voice": "zm_010",  # Male voice (yunjian)
                "lang_code": "z",
                "description": "Li Bai's poem"
            },
        ]
        voices_info = """
Chinese voices available:
  Female: zf_001 (xiaobei), zf_002 (xiaoni), zf_003 (xiaoxiao), zf_004 (xiaoyi)
  Male:   zm_010 (yunjian), zm_011 (yunxi), zm_012 (yunxia), zm_013 (yunyang)"""
    else:
        print("English Kokoro TTS Demo")
        model_path = "mlx-community/Kokoro-82M-bf16"
        examples = [
            {
                "text": "Hello, world! This is a text to speech demo.",
                "output": "/tmp/english_hello.wav",
                "voice": "af_heart",  # Female voice
                "lang_code": "en",
                "description": "Hello World"
            },
            {
                "text": "The quick brown fox jumps over the lazy dog.",
                "output": "/tmp/english_test.wav",
                "voice": "am_adam",  # Male voice
                "lang_code": "en",
                "description": "Pangram test"
            },
        ]
        voices_info = """
English voices available:
  Female: af_heart, af_bella, af_nicole, af_sarah, af_sky
  Male:   am_adam, am_michael, am_onyx"""
    print("="*60)

    generated_files = []

    for example in examples:
        print(f"\n--- {example['description']} ---")
        try:
            path = generate_tts_wav(
                text=example["text"],
                output_path=example["output"],
                voice=example["voice"],
                lang_code=example["lang_code"],
                model_path=model_path,
            )
            if path:
                generated_files.append(path)
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Generated {len(generated_files)} files:")
    for f in generated_files:
        print(f"  - {f}")

    if generated_files:
        print("\nPlay all with:")
        print(f"  for f in {' '.join(generated_files)}; do afplay $f; done")

    print(voices_info)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
