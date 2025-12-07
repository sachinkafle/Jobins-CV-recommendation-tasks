#!/usr/bin/env python3
"""Run evaluation pipeline"""
import asyncio
import argparse
import json
from src.evaluation import RecommendationEvaluator, SyntheticDataGenerator
from src.recommendation import JobMatcher
from src.parsers.models import ResumeSchema, Basics
from src.utils import logger

def create_mock_resume(candidate_id: str) -> ResumeSchema:
    """Create a mock resume for testing"""
    return ResumeSchema(
        basics=Basics(name=f"Test Candidate {candidate_id}", email=f"{candidate_id}@test.com"),
        work=[], education=[], skills=[], languages=[]
    )

async def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    logger.info(f"Running evaluation with {args.num_candidates} candidates")

    generator = SyntheticDataGenerator()
    ground_truth = generator.generate_ground_truth(args.num_candidates, num_jobs=100)

    test_data = [
        (cand_id, create_mock_resume(cand_id), truth)
        for cand_id, truth in ground_truth.items()
    ]

    matcher = JobMatcher()
    evaluator = RecommendationEvaluator(matcher)
    report = await evaluator.run_evaluation(test_data, k=5, batch_size=20)

    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"Total Samples: {report['total_samples']}")
    print(f"Total Time: {report['total_time_sec']}s")
    print("\nQuality Metrics:")
    for metric, values in report['quality_metrics'].items():
        print(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})")
    print("\nPerformance Metrics:")
    for metric, value in report['performance_metrics'].items():
        print(f"  {metric}: {value}")
    print("="*50)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"✓ Saved report to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
