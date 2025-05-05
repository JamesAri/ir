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
