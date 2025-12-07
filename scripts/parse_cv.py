#!/usr/bin/env python3
"""Parse a CV from PDF"""
import argparse
import json
from pathlib import Path
from src.parsers import CVParser
from src.utils import logger

def main():
    parser = argparse.ArgumentParser(description="Parse CV from PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output", help="Output JSON file path")
    args = parser.parse_args()

    cv_parser = CVParser()
    logger.info(f"Parsing CV: {args.pdf}")
    resume = cv_parser.parse(args.pdf)

    if resume:
        logger.info("✓ Parsing successful")
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(resume.dict(), f, indent=2)
            logger.info(f"Saved to {args.output}")
        else:
            print(json.dumps(resume.dict(), indent=2))
    else:
        logger.error("✗ Parsing failed")

if __name__ == "__main__":
    main()
