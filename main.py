#!/usr/bin/env python3
"""Main entry point for CV-Job Matching System"""
import asyncio
from pathlib import Path
from rich.console import Console
from rich.table import Table

from src.parsers import CVParser
from src.recommendation import JobMatcher, db_manager
from src.evaluation import RecommendationEvaluator
from src.utils import logger, config

console = Console()

def display_recommendations(candidate_name: str, recommendations):
    """Display recommendations in a table"""
    table = Table(title=f"Job Recommendations for {candidate_name}")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Job Title", style="magenta")
    table.add_column("Company", style="green")
    table.add_column("Match Score", style="yellow", width=12)
    table.add_column("Skills Match", style="blue", width=12)

    for i, rec in enumerate(recommendations, 1):
        table.add_row(
            str(i),
            rec.job_title,
            rec.company,
            f"{rec.match_score:.2f}",
            f"{rec.matching_factors.skills_match:.2f}"
        )

    console.print(table)

def main():
    """Main workflow"""
    console.print("[bold blue]CV-Job Matching System[/bold blue]\n")

    # Initialize components
    logger.info("Initializing system components")
    parser = CVParser()
    matcher = JobMatcher()

    # Optimize database
    console.print("[yellow]Optimizing database...[/yellow]")
    # db_manager.optimize_database()

    # Example: Parse a CV
    sample_cv_path = "data/sample_cvs/sample_resume.pdf"

    if Path(sample_cv_path).exists():
        console.print(f"\n[cyan]Parsing CV: {sample_cv_path}[/cyan]")
        resume = parser.parse(sample_cv_path)

        if resume:
            console.print(f"[green]✓[/green] Successfully parsed CV for {resume.basics.name}")

            # Generate recommendations
            console.print(f"\n[cyan]Generating job recommendations...[/cyan]")
            recommendations = matcher.match(resume, "demo_candidate")

                   # Print JSON to console
            json_output = recommendations.model_dump_json(indent=2)
            print(json_output)
            
            # Save to file
            # with open(f"output/{"demo_candidate"}_recommendations.json", "w") as f:
            #     f.write(json_output)

if __name__ == "__main__":
    main()
