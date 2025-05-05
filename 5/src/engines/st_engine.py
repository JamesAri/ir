import numpy as np
from sentence_transformers import SentenceTransformer

from interface.search_engine import SearchEngine
from model.document import Document
from utils.heap import HeapEntry, MinHeap


class SentenceTransformersSearchEngine(SearchEngine):

    def __init__(self, documents: list[Document]):
        print("Initializing Sentence Transformers search engine")

        self.documents = documents

        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        sentences = [doc.title + " " + doc.text for doc in self.documents]

        self.embeddings = self.model.encode(sentences, show_progress_bar=True)

        self.similarities = self.model.similarity(self.embeddings, self.embeddings)
        print("Loaded Sentence Transformers model")

    def search(self, query: str, k: int) -> list[Document]:
        heap = MinHeap(max_size=k)

        query_embedding = np.array(self.model.encode(query))
        scores = self.model.similarity(query_embedding, self.embeddings)[0]

        for doc, score in zip(self.documents, scores):
            heap.push(HeapEntry(score=score, document=doc))

        return heap.sort().get_documents()
