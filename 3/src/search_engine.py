from document import Document
from positional_index import PositionalIndex
import heapq
from utils import VectorUtils, TfIdf, VectorSimilarity
import preprocess as pre


class HeapEntry:
    def __init__(self, similarity: float, document: Document):
        self.similarity = similarity
        self.document = document

    def __lt__(self, other):
        return self.similarity < other.similarity

    def __eq__(self, other):
        return self.similarity == other.similarity

    def __gt__(self, other):
        return self.similarity > other.similarity

    def __repr__(self):
        return f"({self.similarity}) {self.document}"


class MinHeap:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.heap = []

    def push(self, val: HeapEntry):
        heapq.heappush(self.heap, val)

        if len(self.heap) > self.max_size:
            heapq.heappop(self.heap)

    def get_heap(self):
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

    def search(self, query, k=10, method="ltc.ltc"):
        """ defaults to ltc.ltc_search """
        
        print(f"Search method: {method}")
        
        # process query
        query_doc = Document(query).tokenize().preprocess(self.pipeline)
        query_index = PositionalIndex([query_doc])
        query_unique_terms = query_index.get_unique_terms()
        
        # get mapping from query term to index in document vector - optimization
        query_term_to_doc_vector_index_mapping = {term: i for i, term in enumerate(self.collection_terms) if term in query_unique_terms}

        # get relevant documents
        relevant_documents = self.get_relevant_documents(query_doc)

        # prepare min heap
        heap = MinHeap(max_size=k)

        search_methods = {
            "ltc.ltc": self.ltc_ltc_search,
            "ltu.ltc": self.ltu_ltc_search
        }

        if method in search_methods:
            search_methods[method](
                heap=heap,
                query_index=query_index,
                query_doc=query_doc,
                query_unique_terms=query_unique_terms,
                relevant_documents=relevant_documents,
                query_term_to_doc_vector_index_mapping=query_term_to_doc_vector_index_mapping
            )
        else:
            raise ValueError(f"Invalid search method: {method}")

        return sorted(heap.get_heap(), reverse=True)

    def ltc_ltc_search(
        self,
        heap: MinHeap,
        query_index: PositionalIndex,
        query_doc: Document,
        query_unique_terms: list[str],
        relevant_documents: list[Document],
        query_term_to_doc_vector_index_mapping: dict[str, int],
    ):
        # get query vector
        query_tf_idf_vector = TfIdf.ltc_weighting(
            tf_vector=[query_index.get_term_frequency(term=term, doc_id=query_doc.doc_id) for term in query_unique_terms],
            df_vector=[self.documents_index.get_document_frequency(term) for term in query_unique_terms],
            total_documents=self.total_documents
        )

        # get top k documents
        for doc in relevant_documents:
            doc_tf_idf_vector = TfIdf.ltc_weighting(
                tf_vector=[self.documents_index.get_term_frequency(term=term, doc_id=doc.doc_id) for term in self.collection_terms],
                df_vector=self.df_vector,
                total_documents=self.total_documents,
            )

            # optimiziaiton - calculate only non-zero dot products - denominator still must use the whole document vector
            doc_tf_idf_vector_trimmed = self.get_trimmed_doc_vector(
                doc_vector=doc_tf_idf_vector,
                query_unique_terms=query_unique_terms,
                query_term_to_doc_vector_index_mapping=query_term_to_doc_vector_index_mapping
            )

            similarity = VectorSimilarity.calculate_cosine_similarity_with_norm(
                vector_a=query_tf_idf_vector,
                vector_b=doc_tf_idf_vector_trimmed,
                norm=VectorUtils.compute_magnitude(query_tf_idf_vector) * VectorUtils.compute_magnitude(doc_tf_idf_vector),
            )
            heap.push(HeapEntry(similarity=similarity, document=doc))
    
    def ltu_ltc_search(
        self,
        heap: MinHeap,
        query_index: PositionalIndex,
        query_doc: Document,
        query_unique_terms: list[str],
        relevant_documents: list[Document],
        query_term_to_doc_vector_index_mapping: dict[str, int]
    ):
        # get query vector
        query_tf_idf_vector = TfIdf.ltc_weighting(
            tf_vector=[query_index.get_term_frequency(term=term, doc_id=query_doc.doc_id) for term in query_unique_terms],
            df_vector=[self.documents_index.get_document_frequency(term) for term in query_unique_terms],
            total_documents=self.total_documents
        )

        # get top k documents
        for doc in relevant_documents:
            doc_tf_idf_vector = TfIdf.ltu_weighting(
                tf_vector=[self.documents_index.get_term_frequency(term=term, doc_id=doc.doc_id) for term in self.collection_terms],
                df_vector=self.df_vector,
                total_documents=self.total_documents,
                document_length=self.documents_index.get_document_length(doc.doc_id),
                avg_document_length=self.avg_document_length,
                slope=0.5
            )

            # optimiziaiton - calculate only non-zero dot products
            doc_tf_idf_vector_trimmed = self.get_trimmed_doc_vector(
                doc_vector=doc_tf_idf_vector,
                query_unique_terms=query_unique_terms,
                query_term_to_doc_vector_index_mapping=query_term_to_doc_vector_index_mapping
            )

            similarity = VectorSimilarity.calculate_dot_product_score(
                vector_a=query_tf_idf_vector,
                vector_b=doc_tf_idf_vector_trimmed,
            )
            heap.push(HeapEntry(similarity=similarity, document=doc))

    def get_trimmed_doc_vector(
        self,
        doc_vector: list[float],
        query_unique_terms: list[str],
        query_term_to_doc_vector_index_mapping: dict[str, int]
    ):
        """
        Returns a trimmed document vector containing only the terms that are present in the query.
        """
        return [
            doc_vector[query_term_to_doc_vector_index_mapping[term]] if term in query_term_to_doc_vector_index_mapping else 0
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
