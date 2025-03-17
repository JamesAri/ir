from collections import defaultdict
import json
from typing import Iterable
from document import Document
import tqdm


class PositionalIndex:

    index: defaultdict[str, defaultdict[int, list[int]]]

    documents_dict: dict[int, Document]
    """doc_id -> Document"""

    def __init__(self, documents: Iterable[Document], show_progress: bool = False):
        self.index = defaultdict(self._index_posting_factory)
        self.documents_dict = {doc.doc_id: doc for doc in documents}
        if show_progress:
            self._add_documents(tqdm.tqdm(self.get_documents(), desc="Indexing documents"))
        else:
            self._add_documents(self.get_documents())

    def _add_documents(self, documents: Iterable[Document]):
        for doc in documents:
            self._add_document(doc)

    def _add_document(self, doc: Document):
        for token in doc.tokens:
            self.index[token.processed_form][doc.doc_id].append(token.position)
            
    def _index_posting_factory(self) -> defaultdict[int, list[int]]:
        """
        Factory method for creating new postings lists.
        - key -> doc_id
        - value -> list of positions
        """
        # we don't want to use lambda so we can pickle the object
        return defaultdict(list)

    def get_document_frequency(self, term: str):
        return len(self.index[term].keys()) if term in self.index else 0

    def get_term_frequency(self, term: str, doc_id: int):
        positions = self.get_positions(term, doc_id)
        return len(positions) if positions else 0
    
    def get_positions(self, term: str, doc_id: int):
        postings = self.get_postings(term)
        return postings[doc_id] if postings and doc_id in postings else None

    def get_document_length(self, doc_id: int):
        return len(self.documents_dict[doc_id].tokens)

    def get_documents(self):
        return list(self.documents_dict.values())

    def get_documents_count(self):
        return len(self.documents_dict)

    def get_documents_dict(self):
        return self.documents_dict

    def get_unique_terms(self, doc_id: int | None = None):
        if doc_id is None:
            return set(self.index.keys())
        else:
            return self.documents_dict[doc_id].get_unique_terms()

    def get_avg_document_length(self):
        doc_ids = self.documents_dict.keys()
        return sum([self.get_document_length(doc_id) for doc_id in doc_ids]) / self.get_documents_count()
    
    def get_postings(self, term: str):
        """
        Returns the postings list for a given term.

        The postings list is a dictionary where the keys are document IDs
        and the values are lists of positions at which the term occurs in the document
        """
        return self.index[term] if term in self.index else None

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
