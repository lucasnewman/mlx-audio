import mlx.core as mx
from mlx_audio.sts.models.moshi.mimi_streamer import StreamTokenizer
from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407

def main():
    mimi_config = mimi_202407(8)
    mimi_model = Mimi(mimi_config)
    
    streamer = StreamTokenizer(mimi_model)
    print("Stream tokenizer loaded with local mimi model")
    
if __name__ == "__main__":
    main()
