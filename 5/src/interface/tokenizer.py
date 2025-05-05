from abc import ABC, abstractmethod
from model.token import Token


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, document: str) -> list[Token]:
        raise NotImplementedError()
