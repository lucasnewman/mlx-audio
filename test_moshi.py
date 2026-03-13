from mlx_audio.sts.models.moshi.moshi import MoshiSTSModel
import numpy as np

def main():
    print("Loading Moshi using native Mimi...")
    model = MoshiSTSModel.from_pretrained("/Users/mm725821/Downloads/moshi", quantized=4)
    print("Model successfully loaded via from_pretrained!")
    
    print("Warming up...")
    model.warmup_tokenizer()

    print("Generating...")
    for word, pcm in model.generate(steps=10):
        if word: print(word, end="", flush=True)

if __name__ == "__main__":
    main()
