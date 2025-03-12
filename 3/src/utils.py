import numpy as np
from sklearn.preprocessing import normalize
import math
from typing import List


def load_stopwords(file_path):
    with open(file_path, encoding="utf-8") as f:
        return set(f.read().splitlines())


class CosineSimilarityWithPivot:
    @staticmethod
    def compute_dot_product(vectorA: List[float], vectorB: List[float]) -> float:
        """Compute the dot product of two vectors."""
        return sum(a * b for a, b in zip(vectorA, vectorB))

    @staticmethod
    def compute_magnitude(vector: List[float]) -> float:
        """Compute the magnitude (Euclidean norm) of a vector."""
        return math.sqrt(sum(x ** 2 for x in vector))

    @staticmethod
    def pivot_normalize(vector: List[float], pivot: float) -> List[float]:
        """Apply pivot normalization to the vector."""
        magnitude = CosineSimilarityWithPivot.compute_magnitude(vector)
        pivoted_magnitude = max(magnitude, pivot)  # Prevents excessive penalization of long vectors
        return [x / pivoted_magnitude for x in vector] if pivoted_magnitude > 0 else vector

    @staticmethod
    def calculate_cosine_similarity(vectorA: List[float], vectorB: List[float], pivot: float) -> float:
        """Calculate the cosine similarity between two vectors with pivot normalization."""
        # Normalize vectors using pivot normalization
        norm_vectorA = CosineSimilarityWithPivot.pivot_normalize(vectorA, pivot)
        norm_vectorB = CosineSimilarityWithPivot.pivot_normalize(vectorB, pivot)

        # Compute cosine similarity on normalized vectors
        dot_product = CosineSimilarityWithPivot.compute_dot_product(norm_vectorA, norm_vectorB)
        magnitudeA = CosineSimilarityWithPivot.compute_magnitude(norm_vectorA)
        magnitudeB = CosineSimilarityWithPivot.compute_magnitude(norm_vectorB)

        # Handle zero vectors
        if magnitudeA == 0 or magnitudeB == 0:
            return 0  # If either vector has zero magnitude, similarity is undefined or 0

        return dot_product / (magnitudeA * magnitudeB)


def pivoted_cosine_normalization(term_doc_matrix, pivot=0.2):
    """
    Applies Pivoted Cosine Normalization to a term-document matrix.

    Parameters:
    - term_doc_matrix (numpy array): A term-document matrix where rows represent terms and columns represent documents.
    - pivot (float): The pivot parameter for normalization, typically between 0 and 1.

    Returns:
    - normalized_matrix (numpy array): The pivoted cosine normalized matrix.
    """
    # Compute document lengths (L_d)
    doc_lengths = np.sum(term_doc_matrix, axis=0)

    # Compute the average document length (L_avg)
    avg_doc_length = np.mean(doc_lengths)

    # Compute normalization factor (1 / (1 - pivot + pivot * (L_d / L_avg)))
    norm_factors = 1 / (1 - pivot + pivot * (doc_lengths / avg_doc_length))

    # Apply normalization to the term-document matrix
    normalized_matrix = term_doc_matrix * norm_factors

    # Apply cosine normalization (L2 normalization)
    normalized_matrix = normalize(normalized_matrix, norm='l2', axis=0)

    return normalized_matrix


if __name__ == "__main__":
    # Example usage
    vectorA = [1, 2, 3]
    vectorB = [4, 5, 6]
    pivot_value = 3.0

    similarity = CosineSimilarityWithPivot.calculate_cosine_similarity(vectorA, vectorB, pivot_value)
    print(f"Cosine Similarity with Pivot Normalization: {similarity:.4f}")

    # Example usage
    term_doc_matrix = np.array([
        [3, 0, 2],
        [1, 5, 0],
        [0, 2, 4]
    ])

    normalized_matrix = pivoted_cosine_normalization(term_doc_matrix)
    print("Pivoted Cosine Normalized Matrix:\n", normalized_matrix)
