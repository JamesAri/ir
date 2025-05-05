from abc import ABC, abstractmethod
from model.document import Document


class Parser(ABC):
    @staticmethod
    @abstractmethod
    def parse(document: str) -> Document:
        raise NotImplementedError()
