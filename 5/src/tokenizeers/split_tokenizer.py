from model.token import Token
from interface.tokenizer import Tokenizer


class SplitTokenizer(Tokenizer):
    def __init__(self, split_char: str):
        self.split_char = split_char

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        position = 0
        for word in document.split(self.split_char):
            token = Token(word, position, len(word))
            tokens.append(token)
            position += len(word) + 1
        return tokens
