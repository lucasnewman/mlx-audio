def is_special_token(token_id: int, vocabulary: list[str]) -> bool:
    if token_id < 0 or token_id >= len(vocabulary):
        return False
    piece = vocabulary[token_id]
    return (piece.startswith("<|") and piece.endswith("|>")) or piece in ("<unk>", "<pad>")


def decode(tokens: list[int], vocabulary: list[str]):
    return "".join(
        vocabulary[token].replace("▁", " ")
        for token in tokens
        if 0 <= token < len(vocabulary) and not is_special_token(token, vocabulary)
    )
