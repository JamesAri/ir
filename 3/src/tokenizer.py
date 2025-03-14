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
    num_pattern = r'(\d+[.,]?\d*)'  # matches numbers like 123, 123.123, 123,123
    word_pattern = r'(\w+)'  # matches words
    html_tag_pattern = r'(<.*?>)'  # matches html tags
    punctuation_pattern = r'([^\w\s]+)'  # matches punctuation
    urls_pattern = r'(http\S+|www\S+)'  # matches urls
    # Matches editions
    # NOTE: we could unidecode the string
    edition_pattern = r'(\d+)(?:\.)?(?:th|nd|rd|st)?\s*(?:edition|edice|edici|vydani|vydání|vydanie|vyd\.|díl|dil|sérii|serii)\w*'
    extension_pattern = r'(\d+)(?:\.)?(?:th|nd|rd|st)?\s*(?:rozšíření|rozsireni|rozš|rozs)\w*'

    default_pattern = f"{urls_pattern}|{html_tag_pattern}|{edition_pattern}|{extension_pattern}|{num_pattern}|{word_pattern}|{punctuation_pattern}"

    def __init__(self, pattern: str = default_pattern):
        self.pattern = pattern

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        for match in re.finditer(re.compile(self.pattern, re.UNICODE | re.IGNORECASE), document):
            token_type = TokenType(match.lastindex)
            if token_type == TokenType.EDITION:
                token = Token(match.group(3) + '[EDICE]', match.start(), match.end() - match.start(), token_type)
            elif token_type == TokenType.EXTENSION:
                token = Token(match.group(4) + '[EXPANZE]', match.start(), match.end() - match.start(), token_type)
            else:
                token = Token(match.group(), match.start(), match.end() - match.start(), token_type)
            tokens.append(token)
        return tokens


if __name__ == '__main__':
    document = 'příliš žluťoučký kůň úpěl ďábelské ódy. 20.25 https//www.google.com a konec'
    tokenizer = SplitTokenizer(" ")
    tokens = tokenizer.tokenize(document)
    print(tokens)
    tokenizer = RegexMatchTokenizer()
    tokens = tokenizer.tokenize(document)
    print(tokens)
