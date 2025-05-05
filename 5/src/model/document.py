from collections import Counter
from typing import Iterable

from bs4 import BeautifulSoup
from tokenizeers import RegexMatchTokenizer
from interface.tokenizer import Tokenizer
from model.pipeline import PreprocessingPipeline
import config

SHOW_CHARS = 70


class Document:
    _doc_id_counter = 0

    def __init__(self, text: str, title: str = ""):
        self.title = title
        self.text = text
        self.content = self.title + " " + self.text
        self.tokens = []
        self.vocab = None
        self.doc_id = Document._doc_id_counter
        Document._doc_id_counter += 1

        self.strip_html()

    def strip_html(self):
        self.content = BeautifulSoup(self.content, "html.parser").get_text()

    def tokenize(self, tokenizer: Tokenizer = None):
        tokenizer = tokenizer or RegexMatchTokenizer()
        self.tokens = tokenizer.tokenize(self.content)
        self.content = None  # let the garbage collector free up memory
        return self

    def preprocess(self, pipeline: PreprocessingPipeline = config.PIPELINE):
        self.tokens = pipeline.preprocess(self.tokens, self.text)
        return self

    def get_unique_terms(self):
        return list(set(token.processed_form for token in self.tokens))

    def __repr__(self):
        content = self.title + " " + self.text
        show_more = (
            f"... [{len(content) - SHOW_CHARS} characters hidden]"
            if len(content) > SHOW_CHARS
            else ""
        )
        return f"Document {self.doc_id}: {content[:SHOW_CHARS]}{show_more}"

    @staticmethod
    def build_vocabulary(documents: Iterable["Document"]):
        vocab = Counter()
        for doc in documents:
            vocab.update((token.processed_form for token in doc.tokens))
        return vocab
