import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    URL = 1
    TAG = 2
    EDITION = 3
    EXTENSION = 4
    NUMBER = 5
    WORD = 6
    PUNCT = 7


regex_map = {
    TokenType.URL: r"(http\S+|www\S+)",
    TokenType.TAG: r"(<.*?>)",
    TokenType.EDITION: r"\d+(?:\.)?(?:th|nd|rd|st)?\s*(?:edition|edice|edici|vydani|vydání|vydanie|vyd\.|díl|dil|sérii|serii)\w*",
    TokenType.EXTENSION: r"\d+(?:\.)?(?:th|nd|rd|st)?\s*(?:rozšíření|rozsireni|rozš|rozs)\w*",
    TokenType.NUMBER: r"(\d+[.,]?\d*)",
    TokenType.WORD: r"(\w+){2,}",
    TokenType.PUNCT: r"([^\w\s]+)",
}


@dataclass
class Token:
    processed_form: str
    position: int
    length: int
    token_type: TokenType = TokenType.WORD

    def __repr__(self):
        return self.processed_form


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, document: str) -> list[Token]:
        raise NotImplementedError()


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


class RegexMatchTokenizer(Tokenizer):
    tokenization_order = [
        # TokenType.URL,
        # TokenType.TAG,
        # TokenType.EDITION,
        # TokenType.EXTENSION,
        # TokenType.NUMBER,
        TokenType.WORD,
        # TokenType.PUNCT,
    ]
    default_pattern = "|".join(
        [
            f"(?P<{token_type.name}>{regex_map[token_type]})"
            for token_type in tokenization_order
        ]
    )

    def __init__(self, pattern: str = default_pattern):
        self.pattern = pattern

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        for match in re.finditer(
            re.compile(self.pattern, re.UNICODE | re.IGNORECASE), document
        ):
            for token_type in self.tokenization_order:
                token = match.group(token_type.name)
                if token:
                    if token_type == TokenType.EDITION:
                        num = re.search(r"\d+", token).group()
                        token = f"{num}[ed]"
                    elif token_type == TokenType.EXTENSION:
                        num = re.search(r"\d+", token).group()
                        token = f"{num}[ex]"
                    elif token_type == TokenType.NUMBER:
                        token = "[num]"
                    tokens.append(
                        Token(
                            token,
                            match.start(),
                            match.end() - match.start(),
                            token_type,
                        )
                    )
                    break
        return tokens


if __name__ == "__main__":
    document = (
        "příliš žluťoučký kůň úpěl ďábelské ódy. 20.25 https//www.google.com a konec"
    )
    tokenizer = SplitTokenizer(" ")
    tokens = tokenizer.tokenize(document)
    print(tokens)
    tokenizer = RegexMatchTokenizer()
    tokens = tokenizer.tokenize(document)
    print(tokens)
