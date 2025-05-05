import scipy.sparse as ss
import scipy.sparse.linalg as sl
import numpy as np
from tqdm import tqdm
from interface.search_engine import SearchEngine
from model.positional_index import PositionalIndex
from model.document import Document
from utils.heap import HeapEntry, MinHeap
from utils.tfidf import TfIdf
from utils.vector import VectorUtils


class LSASearchEngine(SearchEngine):
    def __init__(self, index: PositionalIndex):
        print("Initializing LSA search engine")
        self.index = index
        self.avg_document_length = self.index.get_avg_document_length()
        self.collection_terms = self.index.get_unique_terms()
        self.df_vector = [
            self.index.get_document_frequency(term=term)
            for term in self.collection_terms
        ]
        self.df_vector_term_map = {
            term: i for i, term in enumerate(self.collection_terms)
        }

    def search(self, query: str, k: int) -> list[Document]:
        # prepare min heap for top k results
        heap = MinHeap(max_size=k)

        # process query
        query_doc = Document(query)
        super().prepare_query(query_doc)
        query_index = PositionalIndex([query_doc])
        query_unique_terms = query_index.get_unique_terms()
        query_tf_vector = [
            query_index.get_term_frequency(term=term, doc_id=query_doc.doc_id)
            for term in query_unique_terms
        ]
        query_df_vector = [
            self.index.get_document_frequency(term) for term in query_unique_terms
        ]
        query_tf_idf_vector = TfIdf.ltc_weighting(
            tf_vector=query_tf_vector,
            df_vector=query_df_vector,
            total_documents=self.index.get_documents_count(),
        )

        # get relevant documents - here we totally kill the principle of LSA, but it
        # would take several minutes to compute the SVD for all documents
        relevant_documents = self.get_relevant_documents(query_doc)
        print(f"Found {len(relevant_documents)} relevant documents")
        if not relevant_documents:
            return []

        relevant_documents_terms = set()
        for doc in relevant_documents:
            relevant_documents_terms.update(
                self.index.get_unique_terms(doc_id=doc.doc_id)
            )
        relevant_documents_terms = list(relevant_documents_terms)

        total_relevant_documents = len(relevant_documents)

        df_vector = [
            self.index.get_document_frequency(term=term)
            for term in relevant_documents_terms
        ]

        m, n = len(relevant_documents_terms), total_relevant_documents
        print(f"Preparing {m}x{n} sparse matrix with dtype=np.float64 for TF-IDF")
        A = ss.lil_array((m, n), dtype=np.float64)

        relevant_documents_dict = {
            doc.doc_id: i for i, doc in enumerate(relevant_documents)
        }

        for doc in tqdm(relevant_documents, desc="Computing sparse matrix for TF-IDF"):
            tf_idf_vector = TfIdf.ltc_weighting(
                tf_vector=[
                    self.index.get_term_frequency(term, doc.doc_id)
                    for term in relevant_documents_terms
                ],
                df_vector=df_vector,
                total_documents=total_relevant_documents,
            )
            A[:, relevant_documents_dict[doc.doc_id]] = tf_idf_vector

        U, S, Vt = sl.svds(A, k=min(m, n) // 4)
        S = np.diag(S)

        q = np.zeros((m, 1))
        for i, term in enumerate(relevant_documents_terms):
            if term in query_unique_terms:
                q[i] = query_tf_idf_vector[query_unique_terms.index(term)]

        doc_matrix = S @ Vt

        query_matrix = (np.linalg.inv(S) @ U.T) @ q

        query_vector = query_matrix.flatten()

        for i, doc in enumerate(relevant_documents):
            score = VectorUtils.calculate_cosine_similarity(
                query_vector, doc_matrix[:, i]
            )
            heap.push(HeapEntry(score=score, document=doc))

        return heap.sort().get_documents()

    def get_relevant_documents(self, doc_query: Document):
        tokens = doc_query.tokens
        relevant_docs_ids = set()
        for token in tokens:
            postings = self.index.get_postings(token.processed_form)
            if postings:
                relevant_docs_ids.update(postings.keys())
        return [self.index.documents_dict[doc_id] for doc_id in relevant_docs_ids]
