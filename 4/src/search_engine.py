from typing import Callable, List
from tqdm import tqdm
from document import Document
from positional_index import PositionalIndex
import heapq
from utils import VectorUtils, TfIdf, VectorSimilarity
import preprocess as pre

class HeapEntry:
    def __init__(self, score: float, document: Document):
        self.score = score
        self.document = document

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score

    def __gt__(self, other):
        return self.score > other.score

    def __repr__(self):
        return f"({self.score}) {self.document}"


class MinHeap:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.heap = []

    def push(self, val: HeapEntry):
        heapq.heappush(self.heap, val)

        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)

    def get_array(self):
        return self.heap


class SearchEngine:
    def __init__(self, index: PositionalIndex, pipeline=pre.PreprocessingPipeline([])):
        self.documents_index = index
        self.documents_dict = index.get_documents_dict()
        self.pipeline = pipeline

        # precompute collection specific values
        self.total_documents = self.documents_index.get_documents_count()
        self.avg_document_length = self.documents_index.get_avg_document_length()
        self.collection_terms = self.documents_index.get_unique_terms()
        self.df_vector = [self.documents_index.get_document_frequency(term=term) for term in self.collection_terms]
        self.df_vector_term_map = {term: i for i, term in enumerate(self.collection_terms)}

    def search(self, query: str, k=10, method="ltc.ltc"):
        """ defaults to ltc.ltc_search """

        print(f"Search method: {method}")
        print("Query:", query)

        search_methods = {
            "ltc.ltc": self.ltc_ltc_search,
            "ltu.ltc": self.ltu_ltc_search
        }

        if method in search_methods:
            return search_methods[method](
                query=query,
                k=k,
            )
        else:
            raise ValueError(f"Invalid search method: {method}")

    def _search(
        self,
        query: str,
        k: int,
        get_query_tf_idf_vector: Callable[[List[float], List[int]], List[float]],
        get_doc_tf_idf_vector: Callable[[Document, List[float], List[int]], List[float]],
        get_score: Callable[[List[float], List[float]], float],
    ):
        # prepare min heap for top k results
        heap = MinHeap(max_size=k)
        
        # process query
        query_doc = Document(query).tokenize().preprocess(self.pipeline)
        query_index = PositionalIndex([query_doc])
        query_unique_terms = query_index.get_unique_terms()
        query_tf_vector = [query_index.get_term_frequency(term=term, doc_id=query_doc.doc_id) for term in query_unique_terms]
        query_df_vector = [self.documents_index.get_document_frequency(term) for term in query_unique_terms]
        query_tf_idf_vector = get_query_tf_idf_vector(tf_vector=query_tf_vector, df_vector=query_df_vector)

        # get relevant documents
        relevant_documents = self.get_relevant_documents(query_doc)
        print(f"Found {len(relevant_documents)} relevant documents")

        # get top k results
        for doc in tqdm(relevant_documents, desc='Searching...'):
            # optimization - skip collection terms that are not in the document
            doc_unique_terms = self.documents_index.get_unique_terms(doc_id=doc.doc_id)
            query_term_to_doc_vector_entry_map = {term: i for i, term in enumerate(doc_unique_terms) if term in query_unique_terms}
            tf_vector_trimmed = [self.documents_index.get_term_frequency(term=term, doc_id=doc.doc_id) for term in doc_unique_terms]
            df_vector_trimmed = [self.df_vector[self.df_vector_term_map[term]] for term in doc_unique_terms]

            doc_tf_idf_vector = get_doc_tf_idf_vector(doc=doc, tf_vector=tf_vector_trimmed, df_vector=df_vector_trimmed)

            # optimization - calculate only non-zero dot products
            doc_tf_idf_vector_mapped_to_query_vector = self.map_doc_vector_to_query_vector(
                doc_vector=doc_tf_idf_vector,
                query_unique_terms=query_unique_terms,
                query_term_to_doc_vector_entry_map=query_term_to_doc_vector_entry_map
            )
            score = get_score(query_vector=query_tf_idf_vector, doc_vector=doc_tf_idf_vector_mapped_to_query_vector)
            heap.push(HeapEntry(score=score, document=doc))

        return sorted(heap.get_array(), reverse=True)

    def ltc_ltc_search(self, query: str, k: int):
        def get_query_tf_idf_vector(tf_vector: List[float], df_vector: List[int]):
            return  TfIdf.ltn_weighting(
            tf_vector=tf_vector,
            df_vector=df_vector,
            total_documents=self.total_documents
        )

        def get_doc_tf_idf_vector(doc: Document, tf_vector: List[float], df_vector: List[int]):
            return TfIdf.ltn_weighting(
                tf_vector=tf_vector,
                df_vector=df_vector,
                total_documents=self.total_documents,
            )

        def get_score(query_vector: List[float], doc_vector: List[float]):
            return VectorSimilarity.calculate_cosine_similarity_with_norm(
                vector_a=query_vector,
                vector_b=doc_vector,
                norm=VectorUtils.compute_magnitude(query_vector) * VectorUtils.compute_magnitude(doc_vector),
            )
        
        return self._search(
            query=query,
            k=k,
            get_doc_tf_idf_vector=get_doc_tf_idf_vector,
            get_query_tf_idf_vector=get_query_tf_idf_vector,
            get_score=get_score,
        )

    def ltu_ltc_search(self, query: str, k: int):

        def get_query_tf_idf_vector(tf_vector: List[float], df_vector: List[int]):
            return TfIdf.ltc_weighting(
            tf_vector=tf_vector,
            df_vector=df_vector,
            total_documents=self.total_documents
        )

        def get_doc_tf_idf_vector(doc: Document, tf_vector: List[float], df_vector: List[int]):
            return TfIdf.ltu_weighting(
                tf_vector=tf_vector,
                df_vector=df_vector,
                total_documents=self.total_documents,
                document_length=self.documents_index.get_document_length(doc.doc_id),
                avg_document_length=self.avg_document_length,
                slope=0.95
            )

        def get_score(query_vector: List[float], doc_vector: List[float]):
            return VectorSimilarity.calculate_dot_product_score(
                vector_a=query_vector,
                vector_b=doc_vector,
            )

        return self._search(
            query=query,
            k=k,
            get_query_tf_idf_vector=get_query_tf_idf_vector,
            get_doc_tf_idf_vector=get_doc_tf_idf_vector,
            get_score=get_score
        )

    def map_doc_vector_to_query_vector(
        self,
        doc_vector: list[float],
        query_unique_terms: list[str],
        query_term_to_doc_vector_entry_map: dict[str, int]
    ):
        """
        Returns a mapped document vector containing only the terms that are present in the query.
        Padded with zeros for query terms that are not present in the document.
        """
        return [
            doc_vector[query_term_to_doc_vector_entry_map[term]] if term in query_term_to_doc_vector_entry_map else 0
            for term in query_unique_terms
        ]

    def get_relevant_documents(self, doc_query: Document):
        tokens = doc_query.tokens
        relevant_docs_ids = set()
        for token in tokens:
            postings = self.documents_index.get_postings(token.processed_form)
            if postings:
                relevant_docs_ids.update(postings.keys())
        return [self.documents_dict[doc_id] for doc_id in relevant_docs_ids]
