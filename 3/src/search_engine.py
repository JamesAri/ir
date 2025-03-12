from document import Document
from positional_index import PositionalIndex


class SearchEngine:
    def __init__(self, index: PositionalIndex, pipeline):
        self.index = index
        self.pipeline = pipeline

    def search(self, query, k=10):
        query_doc = Document(query).tokenize()
        query_doc.preprocess(self.pipeline)

        query_vector = self.get_query_vector(query_doc)
        return self.rank_documents(query_vector, k)

    def get_document_vector(self, doc_id):
        pass

    def get_query_vector(self, query):
        pass

    def get_relevant_documents(self, doc_query: Document):
        tokens = doc_query.tokens
        relevant_docs = {}
        for token in tokens:
            token_postings = self.index.get_postings(token.processed_form)

            if not token_postings:
                continue

            for doc_id, positions in token_postings.keys():
                if doc_id not in relevant_docs:
                    relevant_docs[doc_id] = {}

    def rank_documents(self, query, k=10):
        pass

    def get_top_k_documents(self, query, k=10):
        pass

    def get_top_k_documents_with_scores(self, query, k=10):
        pass
