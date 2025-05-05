from tqdm import tqdm
from model.document import Document


def tokenize_documents(documents: list[Document]):
    tqdm_docs = tqdm(documents, desc="Tokenizing documents")
    for doc in tqdm_docs:
        doc.tokenize()


def preprocess_documents(documents: list[Document]):
    tqdm_docs = tqdm(documents, desc="Preprocessing documents")
    for doc in tqdm_docs:
        doc.preprocess()
