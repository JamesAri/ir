import json
import os
import pickle

from model.positional_index import PositionalIndex
from model.document import Document
from interface.parser import Parser
from lemmatizer import bulk_lemmatize
from utils.documents import preprocess_documents, tokenize_documents
import config


class Dataset:
    index: PositionalIndex
    pickle_file: str
    json_file: str
    parser: Parser

    def __init__(self, json_file, pickle_file, parser, tag: str = ""):
        self.tag = tag
        self.json_file = json_file
        self.pickle_file = pickle_file
        self.parser = parser

        # if pickle file exists, load it
        if os.path.exists(self.pickle_file):
            self.index = self._load_index_from_pickle_file()
            Document._doc_id_counter += self.index.get_documents_count()
            return

        self.create_index()

    def create_index(self):
        documents = self._load_documents_from_json_file()
        bulk_lemmatize(documents)
        tokenize_documents(documents)
        preprocess_documents(documents=documents)
        self.index = PositionalIndex(documents=documents, show_progress=True)
        self.save_index()

    def save_index(self):
        if not config.SAVE_TO_DISK:
            print("Saving to disk is disabled, not saving index")
            return
        if self.index is None:
            raise ValueError("Index is None, cannot save to file")

        with open(self.pickle_file, "wb") as f:
            print(f"Saving positional index to file {self.pickle_file}")
            pickle.dump(self.index, f)
            print(f"Saved index with {self.index.get_documents_count()} documents")

    def _load_index_from_pickle_file(self) -> PositionalIndex:
        with open(self.pickle_file, "rb") as f:
            print(f"Loading positional index from file {self.pickle_file}")
            index: PositionalIndex = pickle.load(f)
            print(f"Loaded index with {index.get_documents_count()} documents")
            return index

    def _load_documents_from_json_file(self) -> list[Document]:
        with open(self.json_file, "r", encoding="utf-8") as f:
            print(f"Loading documents from file {self.json_file}")
            documents = []
            for doc in json.load(f):
                documents.append(self.parser.parse(doc))
            print(f"Loaded {len(documents)} documents")
            return documents
