#!/usr/bin/env python3
"""Generate job recommendations"""
import argparse
import json
from src.parsers import CVParser
from src.recommendation import JobMatcher
from src.utils import logger

def main():
    parser = argparse.ArgumentParser(description="Generate job recommendations")
    parser.add_argument("--pdf", help="Path to CV PDF")
    parser.add_argument("--candidate-id", required=True, help="Candidate ID")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    if args.pdf:
        cv_parser = CVParser()
        logger.info(f"Parsing CV: {args.pdf}")
        resume = cv_parser.parse(args.pdf)
        if not resume:
            logger.error("Failed to parse CV")
            return
    else:
        logger.error("Please provide --pdf argument")
        return

    matcher = JobMatcher()
    logger.info("Generating recommendations")
    recommendations = matcher.match(resume, args.candidate_id)

    output = {
    "candidate_id": args.candidate_id,
    "candidate_name": resume.basics.name,
    "recommendations": [rec.model_dump() for rec in recommendations.recommendations]
    # Use .model_dump() on each Pydantic model
}

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"✓ Saved recommendations to {args.output}")
    else:
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
