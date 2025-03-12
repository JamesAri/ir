import os
import json
from tqdm import tqdm

from document import Document
from tokenizer import TokenType
import preprocess as pre
from utils import CosineSimilarityWithPivot, load_stopwords
from positional_index import PositionalIndex

cs_stop_words = load_stopwords(os.path.join(os.path.dirname(__file__), "..", "stopwords", "stopwords-cs.txt"))

MIN_LENGTH_TOKEN = 2

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


def parseDocText(doc):
    return " ".join([doc["Prodavane_predmety"], doc["Popisek"]])
    # return " ".join([doc["title"] or "", doc["text"] or ""])


def write_vocabulary(documents, file):
    vocab = Document.build_vocabulary(documents)
    print(len(vocab))
    with open(file, "w", encoding="utf-8") as f:
        Document.write_weighted_vocab(vocab, f)


def write_index(index, file):
    with open(file, "w", encoding="utf-8") as f:
        f.write(str(index))


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "sample.json")
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "out"))
    os.makedirs(out_dir, exist_ok=True)

    documents = []
    with open(input_file, 'r', encoding="utf-8") as f:
        documents = [Document(parseDocText(doc)).tokenize() for doc in json.load(f)]

    ###### Without Preprocessing ######

    vocab_1_file = os.path.join(out_dir, "vocab1.txt")
    write_vocabulary(documents, vocab_1_file)

    index_1_file = os.path.join(out_dir, "index1.json")
    write_index(PositionalIndex(documents), index_1_file)

    ###### Preprocessing ######

    tqdm_documents = tqdm(documents)
    for i, doc in enumerate(tqdm_documents, 1):
        tqdm_documents.set_description(f"[>] Processing document {i}/{len(documents)}")
        doc.preprocess(pipeline)

    vocab_2_file = os.path.join(out_dir, "vocab2.txt")
    write_vocabulary(documents, vocab_2_file)

    index_2_file = os.path.join(out_dir, "index2.json")
    write_index(PositionalIndex(documents), index_2_file)

    ###### Output ######

    print(f"Vocabulary written to {vocab_1_file} (without preprocessing)")
    print(f"Vocabulary written to {vocab_2_file} (preprocessed)")
    print(f"Positional index written to {index_1_file} (without preprocessing)")
    print(f"Positional index written to {index_2_file} (preprocessed)")
