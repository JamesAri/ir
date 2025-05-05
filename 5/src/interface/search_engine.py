from abc import ABC, abstractmethod
from model.document import Document

from lemmatizer import lemmatize


class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> list[Document]:
        raise NotImplementedError()

    def prepare_query(self, query: Document):
        lemmatize(query)
        query.tokenize().preprocess()
