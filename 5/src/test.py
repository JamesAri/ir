import os

from model.dataset import Dataset
from model.document import Document
from engines.boolean_engine import BooleanSearchEngine
from engines.tf_idf_engine import TfIdfSearchEngine
from lemmatizer import bulk_lemmatize
import parsers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "out"))

os.makedirs(OUT_DIR, exist_ok=True)

dataset = Dataset(
    json_file=os.path.join(BASE_DIR, "..", "data", "zh.json"),
    pickle_file=os.path.join(OUT_DIR, "zh_index.pkl"),
    parser=parsers.ZHParser,
    tag="Zatrolene hry",
)


def boolean():

    engine = BooleanSearchEngine(index=dataset.index)

    res = engine.search(
        query="Karetní AND Prodám",
        k=10,
    )

    print(res)


def tfidf():

    engine = TfIdfSearchEngine(index=dataset.index)

    res = engine.search(
        query="Jo uz to facha",
        k=10,
    )

    print(res)


def pipeline():
    doc1 = Document(
        text="Karetní 1x hry <div>hry</div> 44 karet",
        title="Prodám karetní hry",
    )
    doc2 = Document(
        text="Velký pátek je svátek",
        title="A budeme se na to podívat",
    )
    print("==========================")

    print(doc1.content)
    print(doc2.content)

    bulk_lemmatize([doc1, doc2])
    print(doc1.content)
    print(doc2.content)

    doc1.tokenize()
    doc2.tokenize()

    print(doc1.tokens)
    print(doc2.tokens)

    doc1.preprocess()
    doc2.preprocess()

    print(doc1.tokens)
    print(doc2.tokens)

    print("==========================")


if __name__ == "__main__":
    # boolean()
    # tfidf()
    # tokenization_preprocessing()
    pipeline()
