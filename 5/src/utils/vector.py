import math
from typing import List


class VectorUtils:
    @staticmethod
    def compute_dot_product(vector_a: List[float], vector_b: List[float]) -> float:
        """Compute the dot product of two vectors."""
        return sum(a * b for a, b in zip(vector_a, vector_b, strict=True))

    @staticmethod
    def compute_magnitude(vector: List[float]) -> float:
        """Compute the magnitude (Euclidean norm) of a vector."""
        return math.sqrt(sum(x ** 2 for x in vector))
    
    @staticmethod
    def calculate_cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        """
        Calculate the standard cosine similarity between two vectors using standard dot product and division 
        by magnitudes of the vectors.
        """
        norm = VectorUtils.compute_magnitude(vector_a) * VectorUtils.compute_magnitude(vector_b)
        return VectorUtils.calculate_cosine_similarity_with_norm(vector_a, vector_b, norm)

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
    