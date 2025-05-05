import engines
from model.dataset import Dataset


class EngineCollection:
    dataset: Dataset

    lsa_se: engines.LSASearchEngine
    boolean_se: engines.BooleanSearchEngine
    sent_trans_se: engines.SentenceTransformersSearchEngine
    tfidf_se: engines.TfIdfSearchEngine

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        print("\nCreating search engine collection for dataset:", dataset.tag)
        self.load_engines()
        print("Search engine collection for dataset", dataset.tag, "created")

    def load_engines(self):
        index = self.dataset.index
        documents = list(index.documents_dict.values())

        self.lsa_se = engines.LSASearchEngine(index=index)
        self.boolean_se = engines.BooleanSearchEngine(index=index)
        self.sent_trans_se = engines.SentenceTransformersSearchEngine(documents)
        self.tfidf_se = engines.TfIdfSearchEngine(index=index)

    def refresh_engines(self):
        print("Refreshing search engine collection for dataset:", self.dataset.tag)
        self.load_engines()
        print("Search engine collection for dataset", self.dataset.tag, "refreshed")
