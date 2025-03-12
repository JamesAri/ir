from collections import Counter
from typing import Iterable
import preprocess as pre
from tokenizer import RegexMatchTokenizer, Tokenizer


class Document:
    _doc_id_counter = 0

    def __init__(self, text: str):
        self.text = text
        self.tokens = None
        self.vocab = None
        self.doc_id = Document._doc_id_counter
        Document._doc_id_counter += 1

    def tokenize(self, tokenizer: Tokenizer = None):
        tokenizer = tokenizer or RegexMatchTokenizer()
        self.tokens = tokenizer.tokenize(self.text)
        return self

    def preprocess(self, preprocessing_pipeline: pre.PreprocessingPipeline):
        self.tokens = preprocessing_pipeline.preprocess(self.tokens, self.text)
        return self

    @staticmethod
    def build_vocabulary(documents: Iterable["Document"]):
        vocab = Counter()
        for doc in documents:
            vocab.update((token.processed_form for token in doc.tokens))
        return vocab

    @staticmethod
    def write_weighted_vocab(vocab, file):
        for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            file.write(f"{key} {value}\n")
