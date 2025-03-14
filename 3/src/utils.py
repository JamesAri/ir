from typing import List

import math


def load_stopwords(file_path):
    with open(file_path, encoding="utf-8") as f:
        return set(f.read().splitlines())


class VectorUtils:
    @staticmethod
    def compute_dot_product(vector_a: List[float], vector_b: List[float]) -> float:
        """Compute the dot product of two vectors."""
        return sum(a * b for a, b in zip(vector_a, vector_b, strict=True))

    @staticmethod
    def compute_magnitude(vector: List[float]) -> float:
        """Compute the magnitude (Euclidean norm) of a vector."""
        return math.sqrt(sum(x ** 2 for x in vector))


class TfIdf:

    @staticmethod
    def tf_log(tf_vector: List[float]) -> List[float]:
        """Compute the term-frequency vector using logarithmic scaling."""
        return [1 + math.log(tf, 10) if tf > 0 else 0 for tf in tf_vector]

    @staticmethod
    def tf_idf(tf_vector: List[float], df_vector: List[int], total_documents: int) -> List[float]:
        """
        Compute the TF-IDF vector from term-frequency and ddocument-frequency vectors.
        The term-frequency vector won't be modified - normalization/transformation should be applied beforehand.
        """
        return [
            tf * math.log(total_documents / df, 10) if df > 0 else 0
            for tf, df in zip(tf_vector, df_vector, strict=True)
        ]

    @staticmethod
    def ltu_weighting(
        tf_vector: List[float],
        df_vector: List[int],
        total_documents: int,
        document_length: int,
        avg_document_length: int,
        slope: float = 0.5,
    ) -> List[float]:
        """
        - term-frequency: logarithmic
        - document-frequency: logarithmic (idf)
        - normalization: pivoted cosine normalization
        """
        tf_idf_vect = TfIdf.tf_log(tf_vector)
        tf_idf_vect = TfIdf.tf_idf(tf_idf_vect, df_vector, total_documents)
        # aprox pivot... should be cosine normalization value at which the two curves intersect
        pivot = document_length / avg_document_length
        pivot_norm = (1 - slope) * pivot + VectorUtils.compute_magnitude(tf_idf_vect) * slope
        # pivot_norm = (1 - slope) + pivot * slope
        tf_idf_vect = [tf_idf / pivot_norm for tf_idf in tf_idf_vect]
        return tf_idf_vect

    @staticmethod
    def ltc_weighting(
        tf_vector: List[float],
        df_vector: List[int],
        total_documents: int,
    ) -> List[float]:
        """
        - term-frequency: logarithmic
        - document-frequency: logarithmic (idf)
        - normalization: cosine normalization (creates unit vectors)
        """
        tf_idf_vect = TfIdf.tf_log(tf_vector)
        tf_idf_vect = TfIdf.tf_idf(tf_idf_vect, df_vector, total_documents)
        l2_norm = VectorUtils.compute_magnitude(tf_idf_vect)
        tf_idf_vect = [tf_idf / l2_norm for tf_idf in tf_idf_vect] if l2_norm else tf_idf_vect
        print('l2_norm:', l2_norm)
        return tf_idf_vect

    @staticmethod
    def ltn_weighting(
        tf_vector: List[float],
        df_vector: List[int],
        total_documents: int,
    ) -> List[float]:
        """ 
        - term-frequency: logarithmic
        - document-frequency: logarithmic (idf)
        - normalization: none
        """
        tf_idf_vect = TfIdf.tf_log(tf_vector)
        tf_idf_vect = TfIdf.tf_idf(tf_idf_vect, df_vector, total_documents)
        return tf_idf_vect


class VectorSimilarity:
    @staticmethod
    def calculate_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        """
        Calculate the standard cosine similarity between two vectors using standard dot product and division 
        by magnitudes of the vectors.
        """
        norm = VectorUtils.compute_magnitude(vector_a) * VectorUtils.compute_magnitude(vector_b)
        return VectorSimilarity.calculate_cosine_similarity_with_norm(vector_a, vector_b, norm)

    @staticmethod
    def calculate_cosine_similarity_with_norm(vector_a: List[float], vector_b: List[float], norm: float) -> float:
        """
        Calculate the cosine similarity between two vectors using standard dot product and specified 
        normalization (denominator).
        """
        return (VectorUtils.compute_dot_product(vector_a, vector_b) / norm) if norm else 0

    @staticmethod
    def calculate_cosine_similarity_unit(vector_a: List[float], vector_b: List[float]) -> float:
        """Calculate the standard cosine similarity between two L2 normalized (unit) vectors."""
        return VectorUtils.compute_dot_product(vector_a, vector_b)

    @staticmethod
    def calculate_dot_product_score(vector_a: List[float], vector_b: List[float]) -> float:
        """Calculate the similarity score as a dot product."""
        return VectorUtils.compute_dot_product(vector_a, vector_b)


if __name__ == "__main__":
    # Example usage
    vectorA = [5, 3, 2]
    vectorB = [2, 1, 0]
    df_vector = [1, 1, 1]
    pivot_value = 0.5
    document_lengthA = sum(vectorA)
    document_lengthB = sum(vectorB)
    total_documents = 10
    avg_document_length = 7

    vectorA_ltc = TfIdf.ltc_weighting(vectorA, df_vector, total_documents)
    vectorB_ltc = TfIdf.ltc_weighting(vectorB, df_vector, total_documents)
    vectorA_ltu = TfIdf.ltu_weighting(vectorA, df_vector, total_documents, document_lengthA, avg_document_length, pivot_value)
    vectorB_ltu = TfIdf.ltu_weighting(vectorB, df_vector, total_documents, document_lengthB, avg_document_length, pivot_value)

    print('[ltc.ltc] (optimized):')
    similarity = VectorSimilarity.calculate_cosine_similarity_unit(vectorA_ltc, vectorB_ltc)
    print(f"Cos Similarity: {similarity:.4f}\n")

    print('[ltc.ltc]:')
    similarity = VectorSimilarity.calculate_cosine_similarity(vectorA_ltc, vectorB_ltc)
    print(f"Cos Similarity: {similarity:.4f}\n")

    print('[ltc.ltu]:')
    similarity = VectorSimilarity.calculate_dot_product_score(vectorA_ltc, vectorB_ltu)
    print(f"Cos Similarity: {similarity:.4f}\n")

    print('[ltu.ltc]:')
    similarity = VectorSimilarity.calculate_dot_product_score(vectorA_ltu, vectorB_ltc)
    print(f"Cos Similarity: {similarity:.4f}\n")

    print('[ltu.ltu]:')
    similarity = VectorSimilarity.calculate_dot_product_score(vectorA_ltu, vectorB_ltu)
    print(f"Cos Similarity: {similarity:.4f}\n")
