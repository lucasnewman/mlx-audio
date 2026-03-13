import re

with open('mlx_audio/sts/models/moshi/moshi.py', 'r') as f:
    code = f.read()

# Replace rustymimi.StreamTokenizer
replacement = """
        # Load the Mimi MLX model
        mimi_config = mimi_202407(8) # Moshi uses 8 codebooks
        mimi_model = Mimi(mimi_config)
        mimi_model.load_pytorch_weights(str(path / "tokenizer-e351c8d8-checkpoint125.safetensors"), strict=True)
        self.audio_tokenizer = StreamTokenizer(mimi_model)
"""

code = re.sub(r'self\.audio_tokenizer = rustymimi\.StreamTokenizer\(\n.*?\n\s+\)', replacement.strip(), code, flags=re.DOTALL)

with open('mlx_audio/sts/models/moshi/moshi.py', 'w') as f:
    f.write(code)
