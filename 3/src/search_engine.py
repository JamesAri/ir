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
        return f"({self.similarity}, {self.document})"


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
    def __init__(self, index: PositionalIndex, pipeline: pre.PreprocessingPipeline):
        self.documents_index = index
        self.documents_dict = index.get_documents_dict()
        self.pipeline = pipeline

    def search(self, query, k=10):
        # process query
        query_doc = Document(query).tokenize().preprocess(self.pipeline)

        # get relevant documents
        relevant_documents = self.get_relevant_documents(query_doc)

        # prepare min heap
        heap = MinHeap(max_size=k)

        # prepare query vector
        query_index = PositionalIndex([query_doc])
        query_unique_terms = query_index.get_unique_terms()
        print(query_unique_terms)

        # collection metrics
        total_documents = self.documents_index.get_documents_count()
        avg_document_length = self.documents_index.get_avg_document_length()

        # get query vector
        query_tf_idf_vector = TfIdf.ltc_weighting(
            tf_vector=[query_index.get_term_frequency(term=term, doc_id=query_doc.doc_id) for term in query_unique_terms],
            df_vector=[self.documents_index.get_document_frequency(term) for term in query_unique_terms],
            total_documents=total_documents
        )
        print('query_tf_idf_vector', query_tf_idf_vector)

        # find top k documents
        collection_terms = self.documents_index.get_unique_terms()
        print(collection_terms)

        for doc in relevant_documents:
            print('processing', doc)
            query_term_to_doc_vector_mapping = dict()
            tf_vector = []
            df_vector = []
            for i, term in enumerate(collection_terms):
                tf = self.documents_index.get_term_frequency(term=term, doc_id=doc.doc_id)
                df = self.documents_index.get_document_frequency(term=term)
                if tf and df:
                    tf_vector.append(tf)
                    df_vector.append(df)
                    if term in query_unique_terms:
                        query_term_to_doc_vector_mapping[term] = i

            print(df_vector)
            doc_tf_idf_vector = TfIdf.ltu_weighting(
                tf_vector=tf_vector,
                df_vector=df_vector,
                total_documents=total_documents,
                document_length=self.documents_index.get_document_length(doc.doc_id),
                avg_document_length=avg_document_length,
                slope=0.5
            )
            print('doc_tf_idf_vector', doc_tf_idf_vector)

            # optimiziaiton - calculate only non-zero dot products
            doc_tf_idf_vector_mapped = [doc_tf_idf_vector[query_term_to_doc_vector_mapping[term]] for term in query_unique_terms]
            similarity = VectorSimilarity.calculate_dot_product_score(
                vector_a=query_tf_idf_vector,
                vector_b=doc_tf_idf_vector_mapped,
                # norm=VectorUtils.compute_magnitude(query_tf_idf_vector) * VectorUtils.compute_magnitude(doc_tf_idf_vector),
            )
            heap.push(HeapEntry(similarity=similarity, document=doc))

        return heap.get_heap()

    def get_relevant_documents(self, doc_query: Document):
        tokens = doc_query.tokens
        relevant_docs_ids = set()
        for token in tokens:
            postings = self.documents_index.get_postings(token.processed_form)
            if postings:
                relevant_docs_ids.update(postings.keys())
        return [self.documents_dict[doc_id] for doc_id in relevant_docs_ids]
