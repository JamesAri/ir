import os
import json
import pickle
from tqdm import tqdm

from model.document import Document
import preprocess as pre
from engines.tf_idf_engine import TfIdfSearchEngine
from engines.st_engine import SentenceTransformersSearchEngine
from engines.lsa_engine import LSASearchEngine
from src.model.positional_index import PositionalIndex
from utils.stopwords import load_stopwords

cs_stop_words = load_stopwords(
    os.path.join(os.path.dirname(__file__), "..", "stopwords", "stopwords-cs.txt")
)

pipeline = pre.PreprocessingPipeline(
    [
        pre.StopWordsPreprocessor(cs_stop_words),
        pre.LowercasePreprocessor(),
        # pre.HtmlStripPreprocessor(),
        pre.WhitespaceStripPreprocessor(),
        # pre.LemmatizationPreprocessor(), # turned off so you can test search engine faster
        pre.UnidecodePreprocessor(),
    ]
)


def parse_zatrolene_hry_document_text(doc):
    return " ".join([doc["Prodavane_predmety"], doc["Popisek"]])


def parse_courseware_cz_demo_document_text(doc):
    return " ".join([doc["title"] or "", doc["text"] or ""])


def print_top_k(iterable: list, k: int):
    width = len(str(k))
    for i, item in enumerate(iterable, 1):
        print(f"{str(i).rjust(width)}. {item}")
        if i >= k:
            break


def save_index(index, file):
    with open(file, "wb") as f:
        pickle.dump(index, f)


def load_index(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def get_zh_index():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "sample.json")

    documents = []
    with open(input_file, "r", encoding="utf-8") as f:
        tqdm_json = tqdm(json.load(f), desc="Tokenizing documents")
        for doc in tqdm_json:
            documents.append(
                Document(parse_zatrolene_hry_document_text(doc)).tokenize()
            )

    ###### Preprocessing ######

    tqdm_documents = tqdm(documents, desc="Preprocessing documents")
    for doc in tqdm_documents:
        doc.preprocess(pipeline)

    ###### Positional Index ######

    return PositionalIndex(documents, show_progress=True)


def get_cw_index():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "documents.json")
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "out"))
    os.makedirs(out_dir, exist_ok=True)

    index_file = os.path.join(out_dir, "index.pkl")

    if os.path.exists(index_file):
        print(f"Loading positional index from file {index_file}")
        index: PositionalIndex = load_index(index_file)
        print(f"Loaded positional index with {index.get_documents_count()} documents")
    else:
        documents = []
        with open(input_file, "r", encoding="utf-8") as f:
            tqdm_json = tqdm(json.load(f), desc="Tokenizing documents")
            for doc in tqdm_json:
                documents.append(
                    Document(parse_courseware_cz_demo_document_text(doc)).tokenize()
                )

        ###### Preprocessing ######

        tqdm_documents = tqdm(documents, desc="Preprocessing documents")
        for doc in tqdm_documents:
            doc.preprocess(pipeline)

        ###### Positional Index ######

        index = PositionalIndex(documents, show_progress=True)

        print("Saving positional index to file (pickle)")
        print("Note: should be around 1,1GB for the courseware data")
        save_index(index, index_file)
        print(f"Saved positional index to file {index_file}")

    return index


if __name__ == "__main__":
    index = get_zh_index()
    # index = get_cw_index()

    search_engines = [
        TfIdfSearchEngine(index, pipeline),
        LSASearchEngine(index, pipeline),
        SentenceTransformersSearchEngine(index.get_documents()),
    ]

    k = 15 // len(search_engines)

    print(f"Looking for top {k * len(search_engines)} results.")

    for engine in search_engines:
        print(f"\nSearching with {engine.__class__.__name__}")
        result = engine.search(query="Horror hra", k=k)
        print_top_k(result, k=k)
