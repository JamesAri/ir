from typing import List

import math

from utils.vector import VectorUtils


class TfIdf:

    @staticmethod
    def tf_log(tf_vector: List[float]) -> List[float]:
        """Compute the term-frequency vector using logarithmic scaling."""
        return [1 + math.log(tf, 10) if tf > 0 else 0 for tf in tf_vector]

    @staticmethod
    def tf_idf(
        tf_vector: List[float], df_vector: List[int], total_documents: int
    ) -> List[float]:
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
        pivot_norm = (1 - slope) * pivot + VectorUtils.compute_magnitude(
            tf_idf_vect
        ) * slope
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
        tf_idf_vect = (
            [tf_idf / l2_norm for tf_idf in tf_idf_vect] if l2_norm else tf_idf_vect
        )
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
