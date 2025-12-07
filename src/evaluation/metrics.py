"""Evaluation metrics"""
from typing import List, Set
import numpy as np

class MetricsCalculator:
    """Calculate recommendation metrics"""

    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """Precision@K metric"""
        if not recommended:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for job in top_k if job in relevant)
        return hits / k

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """Recall@K metric"""
        if not relevant:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for job in top_k if job in relevant)
        return hits / len(relevant)

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """F1 Score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mean_reciprocal_rank(recommended: List[str], relevant: Set[str]) -> float:
        """MRR metric"""
        for i, job in enumerate(recommended, 1):
            if job in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K"""
        dcg = sum([1.0 / np.log2(i + 2) for i, job in enumerate(recommended[:k]) if job in relevant])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(relevant)))])
        return dcg / idcg if idcg > 0 else 0.0
