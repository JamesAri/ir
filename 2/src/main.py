import os
import json
from typing import Iterable

from tqdm import tqdm

from tokenizer import RegexMatchTokenizer, TokenType, Tokenizer
import preprocess as pre
from collections import defaultdict, Counter
from utils import load_stopwords

cs_stop_words = load_stopwords(os.path.join(os.path.dirname(__file__), "..", "stopwords", "stopwords-cs.txt"))

MIN_LENGTH_TOKEN = 2


class Document:
    def __init__(self, text: str):
        self.text = text
        self.tokens = None
        self.vocab = None

    def tokenize(self, tokenizer: Tokenizer = None):
        tokenizer = tokenizer or RegexMatchTokenizer()
        self.tokens = tokenizer.tokenize(self.text)
        return self

    def preprocess(self, preprocessing_pipeline: pre.PreprocessingPipeline):
        self.tokens = preprocessing_pipeline.preprocess(self.tokens, self.text)
        return self


def build_vocabulary(documents: Iterable[Document]):
    vocab = Counter()
    for doc in documents:
        vocab.update((token.processed_form for token in doc.tokens))
    return vocab


def write_weighted_vocab(vocab, file):
    for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
        file.write(f"{key} {value}\n")


def parseDocText(doc):
    return " ".join([doc["Prodavane_predmety"], doc["Popisek"]])
    # return " ".join([doc["title"] or "", doc["text"] or ""])


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "sample.json")
    out_dir = os.path.join(script_dir, "..", "out")
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, "vocab.txt")

    documents = []
    with open(input_file, 'r', encoding="utf-8") as f:
        documents = [Document(parseDocText(doc)).tokenize() for doc in json.load(f)]

    vocab = build_vocabulary(documents)
    print(len(vocab))
    out_file = os.path.normpath(os.path.join(out_dir, "vocab1.txt"))
    with open(out_file, "w", encoding="utf-8") as f:
        write_weighted_vocab(vocab, f)

    pipeline = pre.PreprocessingPipeline([
        pre.TokenFilterPreprocessor([TokenType.URL, TokenType.PUNCT]),
        pre.StopWordsPreprocessor(cs_stop_words),
        pre.LowercasePreprocessor(),
        pre.HtmlStripPreprocessor(),
        pre.WhitespaceStripPreprocessor(),
        pre.NumberPreprocessor(),
        pre.TokenLengthPreprocessor(MIN_LENGTH_TOKEN),
        # pre.LemmatizationPreprocessor(),
        pre.UnidecodePreprocessor(),
    ])

    tqdm_documents = tqdm(documents)
    for i, doc in enumerate(tqdm_documents, 1):
        tqdm_documents.set_description(f"[>] Processing document {i}/{len(documents)}")
        doc.preprocess(pipeline)

    vocab = build_vocabulary(documents)
    print(len(vocab))
    out_file_preprocessed = os.path.normpath(os.path.join(out_dir, "vocab2.txt"))
    with open(out_file_preprocessed, "w", encoding="utf-8") as f:
        write_weighted_vocab(vocab, f)

    print(f"Vocabulary written to {out_file} (without preprocessing)")
    print(f"Vocabulary written to {out_file_preprocessed} (preprocessed)")
