#!/usr/bin/env python3
"""Initialize database with sample data"""
import random
from src.recommendation.database import db_manager
from src.utils import logger

def generate_sample_jobs(num_jobs: int = 100):
    """Generate sample job postings"""
    roles = ["Senior Engineer", "Data Scientist", "ML Engineer", "Backend Developer", 
             "Frontend Developer", "DevOps Engineer", "Product Manager", "AI Researcher"]
    companies = ["Tech Corp", "AI Innovations", "Data Systems Inc", "Cloud Solutions", 
                 "StartupXYZ", "Enterprise Tech", "Future Labs", "Digital Dynamics"]

    skills_pool = ["Python", "Java", "JavaScript", "React", "Node.js", "AWS", "Docker", 
                   "Kubernetes", "TensorFlow", "PyTorch", "SQL", "NoSQL", "Git", "CI/CD"]

    jobs = []
    for i in range(num_jobs):
        role = random.choice(roles)
        company = random.choice(companies)
        required_skills = random.sample(skills_pool, k=random.randint(3, 7))

        jobs.append({
            "job_id": f"job_{i+1:03d}",
            "title": role,
            "company": company,
            "description": f"We are looking for a talented {role} to join our team.",
            "requirements": f"Required: {', '.join(required_skills)}. {random.randint(2, 8)}+ years experience."
        })

    return jobs

def main():
    """Initialize database"""
    logger.info("Initializing database")
    db_manager.optimize_database()
    logger.info("Generating sample job postings")
    jobs = generate_sample_jobs(100)
    logger.info("Adding jobs to vector store")
    db_manager.add_jobs(jobs)
    logger.info(f"✓ Successfully added {len(jobs)} jobs to database")

if __name__ == "__main__":
    main()
