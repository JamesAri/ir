from collections import defaultdict
import json
from typing import Iterable
from document import Document
from dataclasses import dataclass


@dataclass
class Posting:
    doc_id: int
    positions: list[int]

    @property
    def tf(self):
        return len(self.positions)


class PositionalIndex:

    index: defaultdict[str, defaultdict[int, list[int]]]

    def __init__(self, documents=None):
        self.index = defaultdict(lambda: defaultdict(list))
        if documents:
            self.add_documents(documents)

    def add_documents(self, documents: Iterable[Document]):
        [self.add_document(doc) for doc in documents]

    def add_document(self, doc: Document):
        for token in doc.tokens:
            self.index[token.processed_form][doc.doc_id].append(token.position)

    def get_index(self):
        return self.index

    def get_document_frequency(self, term):
        return len(self.index[term])

    def get_term_frequency(self, term, doc_id):
        return len(self.index[term][doc_id])

    def get_postings(self, term):
        return self.index[term]

    def get_positions(self, term, doc_id):
        return self.index[term][doc_id]

    def get_document_count(self):
        return len(self.index)

    def __repr__(self):
        return json.dumps(
            {
                term: {
                    doc_id: {
                        "tf": len(positions),
                        "positions": positions
                    }
                    for doc_id, positions in postings.items()
                } for term, postings in self.index.items()
            },
            indent=2,
            ensure_ascii=False
        )
