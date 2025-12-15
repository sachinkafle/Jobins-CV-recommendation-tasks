"""Evaluation pipeline"""
import asyncio
import time
import random
from typing import List, Dict, Set
from dataclasses import dataclass
import pandas as pd

from .metrics import MetricsCalculator
from ..parsers.models import ResumeSchema
from ..recommendation.matcher import JobMatcher
from ..utils import logger


@dataclass
class EvalResult:
    """Single evaluation result"""
    candidate_id: str
    latency: float
    precision: float
    recall: float
    f1: float
    ndcg: float


class SyntheticDataGenerator:
    """Generate synthetic ground truth"""
    
    @staticmethod
    def generate_realistic_ground_truth(num_candidates: int, num_jobs: int) -> Dict[str, Set[str]]:
        """Create broader ground truth"""
        ground_truth = {}
        
        for i in range(num_candidates):
            cand_id = f"candidate_{i}"
            profile_type = i % 5  # 0-4 for your 5 job types
            
            # Mark ALL jobs of matching type as relevant (not just top 5)
            relevant = set()
            for j in range(num_jobs):
                if j % 5 == profile_type:  # Every 5th job matches
                    relevant.add(f"job_{str(j+1).zfill(3)}")
            
            # Now relevant set has ~200 jobs for 1000 job database
            ground_truth[cand_id] = relevant
        
        return ground_truth




class RecommendationEvaluator:
    """Evaluate recommendation system"""

    def __init__(self, matcher: JobMatcher):
        self.matcher = matcher
        self.results: List[EvalResult] = []

    async def evaluate_single(
        self, 
        candidate_id: str, 
        resume: ResumeSchema,
        ground_truth: Set[str],
        k: int = 5
    ) -> EvalResult:
        """Evaluate single candidate"""
        start = time.time()

        try:
            # Get recommendations (returns RecommendationOutput object)
            recommendation_output = self.matcher.match(resume, candidate_id)
            
            # Extract job IDs from the recommendations list
            recommended_ids = [
                rec.job_id 
                for rec in recommendation_output.recommendations[:k]
            ]

            latency = time.time() - start

            # Calculate metrics
            calc = MetricsCalculator()
            precision = calc.precision_at_k(recommended_ids, ground_truth, k)
            recall = calc.recall_at_k(recommended_ids, ground_truth, k)
            f1 = calc.f1_score(precision, recall)
            ndcg = calc.ndcg_at_k(recommended_ids, ground_truth, k)

            return EvalResult(
                candidate_id=candidate_id,
                latency=latency,
                precision=precision,
                recall=recall,
                f1=f1,
                ndcg=ndcg
            )
        
        except Exception as e:
            logger.error(f"Error evaluating {candidate_id}: {e}")
            # Return zero metrics on error
            return EvalResult(
                candidate_id=candidate_id,
                latency=time.time() - start,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                ndcg=0.0
            )

    async def run_evaluation(
        self,
        test_data: List[tuple],  # List of (candidate_id, resume, ground_truth)
        k: int = 5,
        batch_size: int = 20
    ) -> Dict:
        """Run full evaluation"""
        logger.info(f"Starting evaluation on {len(test_data)} candidates")

        start_time = time.time()

        # Process in batches
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            tasks = [
                self.evaluate_single(cand_id, resume, truth, k)
                for cand_id, resume, truth in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            self.results.extend(batch_results)

            logger.info(f"Completed batch {i//batch_size + 1}/{(len(test_data)-1)//batch_size + 1}")

        total_time = time.time() - start_time

        return self.generate_report(total_time)

    def generate_report(self, total_time: float) -> Dict:
        """Generate evaluation report"""
        df = pd.DataFrame([vars(r) for r in self.results])

        return {
            "total_samples": len(df),
            "total_time_sec": round(total_time, 2),
            "quality_metrics": {
                "precision@5": {
                    "mean": round(df['precision'].mean(), 4),
                    "std": round(df['precision'].std(), 4)
                },
                "recall@5": {
                    "mean": round(df['recall'].mean(), 4),
                    "std": round(df['recall'].std(), 4)
                },
                "f1_score": {
                    "mean": round(df['f1'].mean(), 4),
                    "std": round(df['f1'].std(), 4)
                },
                "ndcg@5": {
                    "mean": round(df['ndcg'].mean(), 4),
                    "std": round(df['ndcg'].std(), 4)
                }
            },
            "performance_metrics": {
                "avg_latency_sec": round(df['latency'].mean(), 4),
                "p50_latency_sec": round(df['latency'].quantile(0.50), 4),
                "p95_latency_sec": round(df['latency'].quantile(0.95), 4),
                "throughput_req_per_sec": round(len(df) / total_time, 2)
            }
        }
