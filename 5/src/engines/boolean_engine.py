from engines.boolean_parser import BooleanParser
from interface.search_engine import SearchEngine
from model.positional_index import PositionalIndex
from model.document import Document


class BooleanSearchEngine(SearchEngine):
    def __init__(self, index: PositionalIndex):
        print("Initializing Boolean search engine")
        self.index = index
        self.all_docs_ids = set(self.index.documents_dict.keys())

    def search(self, query: str, k: int) -> list[Document]:
        parser = BooleanParser(query)
        ast = parser.parse()
        ids = ast.evaluate(self.index, self.all_docs_ids)
        docs = [self.index.documents_dict[doc_id] for doc_id in ids]
        return docs[:k]
