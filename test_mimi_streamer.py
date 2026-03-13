import mlx.core as mx
from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407

config = mimi_202407(8) # Moshi uses 8 codebooks
model = Mimi(config)

x = mx.random.normal((1, 1, 1920))
encoded = model.encode_step(x)
print("Encoded shape:", encoded.shape)

decoded = model.decode_step(encoded)
print("Decoded shape:", decoded.shape)
