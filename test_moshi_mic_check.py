from mlx_audio.sts.models.moshi.moshi import MoshiSTSModel
import inspect

def test():
    model = MoshiSTSModel.from_pretrained("/Users/mm725821/Downloads/moshi", quantized=4)
    print("Testing type hints")
    print(inspect.signature(model.generate))

if __name__ == '__main__':
    test()
