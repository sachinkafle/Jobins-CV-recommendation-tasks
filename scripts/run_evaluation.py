#!/usr/bin/env python3
"""Run evaluation pipeline"""
import asyncio
import argparse
import json
from src.evaluation import RecommendationEvaluator, SyntheticDataGenerator
from src.recommendation import JobMatcher
from src.parsers.models import ResumeSchema, Basics, Work, Education, Skill
from src.utils import logger


def create_realistic_resume(candidate_id: str, profile_type: str) -> ResumeSchema:
    """Create realistic mock resumes with actual content"""
    
    profiles = {
        "ml_engineer": {
            "name": f"ML Engineer {candidate_id}",
            "summary": "Machine Learning Engineer with 5 years experience in Python, TensorFlow, and AWS",
            "skills": [
                Skill(name="Python", keywords=["Python", "Pandas", "NumPy"]),
                Skill(name="Machine Learning", keywords=["TensorFlow", "PyTorch", "scikit-learn"]),
                Skill(name="Cloud", keywords=["AWS", "Docker", "Kubernetes"])
            ],
            "work": [
                Work(
                    name="Tech Corp",
                    position="Senior ML Engineer",
                    summary="Built ML models for production systems",
                    startDate="2019-01-01",
                    endDate="Present"
                )
            ],
            "education": [
                Education(
                    institution="State University",
                    area="Computer Science",
                    studyType="Master's"
                )
            ]
        },
        "full_stack": {
            "name": f"Full Stack Developer {candidate_id}",
            "summary": "Full Stack Developer with 3 years experience in React, Node.js, and MongoDB",
            "skills": [
                Skill(name="JavaScript", keywords=["JavaScript", "React", "Node.js"]),
                Skill(name="Databases", keywords=["MongoDB", "PostgreSQL"]),
                Skill(name="Web Development", keywords=["REST APIs", "HTML", "CSS"])
            ],
            "work": [
                Work(
                    name="StartupXYZ",
                    position="Full Stack Developer",
                    summary="Developed modern web applications",
                    startDate="2020-06-01",
                    endDate="Present"
                )
            ],
            "education": [
                Education(
                    institution="Tech Institute",
                    area="Software Engineering",
                    studyType="Bachelor's"
                )
            ]
        },
        "data_scientist": {
            "name": f"Data Scientist {candidate_id}",
            "summary": "Data Scientist with expertise in Python, SQL, and statistical analysis",
            "skills": [
                Skill(name="Python", keywords=["Python", "Pandas", "SQL"]),
                Skill(name="Statistics", keywords=["Machine Learning", "Statistics", "Data Analysis"]),
                Skill(name="Visualization", keywords=["Tableau", "Matplotlib"])
            ],
            "work": [
                Work(
                    name="Analytics Corp",
                    position="Data Scientist",
                    summary="Analyzed large datasets and built predictive models",
                    startDate="2018-03-01",
                    endDate="Present"
                )
            ],
            "education": [
                Education(
                    institution="Data University",
                    area="Data Science",
                    studyType="Master's"
                )
            ]
        },
        "devops": {
            "name": f"DevOps Engineer {candidate_id}",
            "summary": "DevOps Engineer specializing in AWS, Kubernetes, and CI/CD",
            "skills": [
                Skill(name="Cloud", keywords=["AWS", "GCP", "Azure"]),
                Skill(name="Containers", keywords=["Docker", "Kubernetes"]),
                Skill(name="CI/CD", keywords=["Jenkins", "Terraform", "Ansible"])
            ],
            "work": [
                Work(
                    name="Cloud Solutions Ltd",
                    position="DevOps Engineer",
                    summary="Managed cloud infrastructure and deployment pipelines",
                    startDate="2017-09-01",
                    endDate="Present"
                )
            ],
            "education": [
                Education(
                    institution="Engineering College",
                    area="Information Technology",
                    studyType="Bachelor's"
                )
            ]
        },
        "backend": {
            "name": f"Backend Developer {candidate_id}",
            "summary": "Backend Developer with Java, Spring Boot, and PostgreSQL experience",
            "skills": [
                Skill(name="Java", keywords=["Java", "Spring Boot"]),
                Skill(name="Databases", keywords=["PostgreSQL", "Redis"]),
                Skill(name="APIs", keywords=["REST APIs", "Microservices"])
            ],
            "work": [
                Work(
                    name="Enterprise Systems",
                    position="Backend Developer",
                    summary="Designed scalable backend services",
                    startDate="2019-04-01",
                    endDate="Present"
                )
            ],
            "education": [
                Education(
                    institution="Tech University",
                    area="Computer Engineering",
                    studyType="Bachelor's"
                )
            ]
        }
    }
    
    # Cycle through profile types
    profile_types = list(profiles.keys())
    profile_type = profile_types[int(candidate_id.split('_')[-1]) % len(profile_types)]
    profile = profiles[profile_type]
    
    return ResumeSchema(
        basics=Basics(
            name=profile["name"],
            email=f"{candidate_id}@test.com",
            summary=profile["summary"]
        ),
        work=profile["work"],
        education=profile["education"],
        skills=profile["skills"],
        languages=[]
    )


async def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--output", help="Output JSON file", default="output/evaluation_report.json")
    args = parser.parse_args()
    
    logger.info(f"Running evaluation with {args.num_candidates} candidates")
    
    # Generate realistic test data
    generator = SyntheticDataGenerator()
    
    # First, let's see what jobs are actually in the database
    from src.recommendation.database import db_manager
    from sqlalchemy import text
    
    with db_manager.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding;"))
        num_jobs = result.scalar()
        logger.info(f"Database contains {num_jobs} jobs")
    
    # Generate ground truth based on actual job patterns
    # This creates relevance labels that match job types
    ground_truth = generator.generate_realistic_ground_truth(
        args.num_candidates, 
        num_jobs=num_jobs
    )
    
    # Create realistic test resumes
    test_data = [
        (cand_id, create_realistic_resume(cand_id, "auto"), truth)
        for cand_id, truth in ground_truth.items()
    ]
    
    logger.info(f"Generated {len(test_data)} test cases")
    logger.info(f"Sample ground truth: {list(ground_truth.values())[0]}")
    
    # Run evaluation
    matcher = JobMatcher()
    evaluator = RecommendationEvaluator(matcher)
    report = await evaluator.run_evaluation(test_data, k=5, batch_size=5)
    
    # Print report
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    print(f"Total Samples: {report['total_samples']}")
    print(f"Total Time: {report['total_time_sec']}s")
    print("\nQuality Metrics:")
    for metric, values in report['quality_metrics'].items():
        print(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})")
    print("\nPerformance Metrics:")
    for metric, value in report['performance_metrics'].items():
        print(f"  {metric}: {value}")
    print("="*70)
    
    # Save report
    import os
    os.makedirs("output", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"✓ Saved report to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
