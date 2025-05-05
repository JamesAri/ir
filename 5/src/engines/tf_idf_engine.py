from typing import Callable, List
from tqdm import tqdm
from interface.search_engine import SearchEngine
from model.document import Document
from model.positional_index import PositionalIndex
from utils.heap import HeapEntry, MinHeap
from utils.vector import VectorUtils
from utils.tfidf import TfIdf
import config


class TfIdfSearchEngine(SearchEngine):
    def __init__(self, index: PositionalIndex):
        print("Initializing TF-IDF search engine")
        self.index = index
        self.method = config.DEFAULT_TF_IDF_METHOD

        # precompute collection specific values
        self.total_documents = self.index.get_documents_count()
        self.avg_document_length = self.index.get_avg_document_length()
        self.collection_terms = self.index.get_unique_terms()
        self.df_vector = [
            self.index.get_document_frequency(term=term)
            for term in self.collection_terms
        ]
        self.df_vector_term_map = {
            term: i for i, term in enumerate(self.collection_terms)
        }

    def set_method(self, method: str):
        self.method = method

    def search(self, query: str, k: int) -> list[Document]:
        """defaults to ltc.ltc_search"""

        print(f"Search method: {self.method}")
        print("Query:", query)

        search_methods = {
            "ltc.ltc": self.ltc_ltc_search,
            "ltu.ltc": self.ltu_ltc_search,
        }

        if self.method in search_methods:
            return search_methods[self.method](
                query=query,
                k=k,
            )
        else:
            raise ValueError(f"Invalid search method: {self.method}")

    def _search(
        self,
        query: str,
        k: int,
        get_query_tf_idf_vector: Callable[[List[float], List[int]], List[float]],
        get_doc_tf_idf_vector: Callable[
            [Document, List[float], List[int]], List[float]
        ],
        get_score: Callable[[List[float], List[float], List[float]], float],
    ) -> list[Document]:
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
        query_tf_idf_vector = get_query_tf_idf_vector(
            tf_vector=query_tf_vector, df_vector=query_df_vector
        )

        # get relevant documents
        relevant_documents = self.get_relevant_documents(query_doc)
        print(f"Found {len(relevant_documents)} relevant documents")
        if not relevant_documents:
            return []

        # get top k results
        for doc in tqdm(relevant_documents, desc="Searching..."):
            # optimization - skip collection terms that are not in the document
            doc_unique_terms = self.index.get_unique_terms(doc_id=doc.doc_id)
            query_term_to_doc_vector_entry_map = {
                term: i
                for i, term in enumerate(doc_unique_terms)
                if term in query_unique_terms
            }
            tf_vector_trimmed = [
                self.index.get_term_frequency(term=term, doc_id=doc.doc_id)
                for term in doc_unique_terms
            ]
            df_vector_trimmed = [
                self.df_vector[self.df_vector_term_map[term]]
                for term in doc_unique_terms
            ]

            doc_tf_idf_vector = get_doc_tf_idf_vector(
                doc=doc, tf_vector=tf_vector_trimmed, df_vector=df_vector_trimmed
            )

            # optimization - calculate only non-zero dot products
            doc_tf_idf_vector_mapped_to_query_vector = self.map_doc_vector_to_query_vector(
                doc_vector=doc_tf_idf_vector,
                query_unique_terms=query_unique_terms,
                query_term_to_doc_vector_entry_map=query_term_to_doc_vector_entry_map,
            )

            score = get_score(
                query_vector=query_tf_idf_vector,
                doc_vector=doc_tf_idf_vector,
                doc_vector_mapped=doc_tf_idf_vector_mapped_to_query_vector,
            )
            heap.push(HeapEntry(score=score, document=doc))

        return heap.sort().get_documents()

    def ltc_ltc_search(self, query: str, k: int) -> list[Document]:
        return self._search(
            query=query,
            k=k,
            get_query_tf_idf_vector=lambda tf_vector, df_vector: TfIdf.ltc_weighting(
                tf_vector=tf_vector,
                df_vector=df_vector,
                total_documents=self.total_documents,
            ),
            get_doc_tf_idf_vector=lambda doc, tf_vector, df_vector: TfIdf.ltc_weighting(
                tf_vector=tf_vector,
                df_vector=df_vector,
                total_documents=self.total_documents,
            ),
            get_score=lambda query_vector, doc_vector, doc_vector_mapped: VectorUtils.calculate_cosine_similarity_with_norm(
                vector_a=query_vector,
                vector_b=doc_vector_mapped,
                norm=VectorUtils.compute_magnitude(query_vector)
                * VectorUtils.compute_magnitude(doc_vector),
            ),
        )

    def ltu_ltc_search(self, query: str, k: int) -> list[Document]:
        return self._search(
            query=query,
            k=k,
            get_query_tf_idf_vector=lambda tf_vector, df_vector: TfIdf.ltc_weighting(
                tf_vector=tf_vector,
                df_vector=df_vector,
                total_documents=self.total_documents,
            ),
            get_doc_tf_idf_vector=lambda doc, tf_vector, df_vector: TfIdf.ltu_weighting(
                tf_vector=tf_vector,
                df_vector=df_vector,
                total_documents=self.total_documents,
                document_length=self.index.get_document_length(doc.doc_id),
                avg_document_length=self.avg_document_length,
                slope=0.75,
            ),
            get_score=lambda query_vector, doc_vector, doc_vector_mapped: VectorUtils.calculate_dot_product_score(
                vector_a=query_vector,
                vector_b=doc_vector_mapped,
            ),
        )

    def map_doc_vector_to_query_vector(
        self,
        doc_vector: list[float],
        query_unique_terms: list[str],
        query_term_to_doc_vector_entry_map: dict[str, float],
    ):
        """
        Returns a mapped document vector containing only the terms that are present in the query.
        Padded with zeros for query terms that are not present in the document.
        """
        return [
            (
                doc_vector[query_term_to_doc_vector_entry_map[term]]
                if term in query_term_to_doc_vector_entry_map
                else 0.0
            )
            for term in query_unique_terms
        ]

    def get_relevant_documents(self, doc_query: Document):
        tokens = doc_query.tokens
        relevant_docs_ids = set()
        for token in tokens:
            postings = self.index.get_postings(token.processed_form)
            if postings:
                relevant_docs_ids.update(postings.keys())
        return [self.index.documents_dict[doc_id] for doc_id in relevant_docs_ids]
