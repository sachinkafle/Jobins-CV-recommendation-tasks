"""Batch CV processing with concurrency"""
import os
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json

from .parsers.cv_parser import CVParser
from .recommendation.matcher import JobMatcher
from .recommendation.database import db_manager
from .utils import logger


@dataclass
class ProcessingResult:
    """Result of processing a single CV"""
    candidate_id: str
    success: bool
    parsing_time: float
    matching_time: float
    num_recommendations: int
    error: str = None


class BatchCVProcessor:
    """Process multiple CVs concurrently"""
    
    def __init__(self, max_workers: int = 10):
        self.parser = CVParser()
        self.matcher = JobMatcher()
        self.max_workers = max_workers
    
    def process_single_cv(self, cv_path: str) -> ProcessingResult:
        """Process a single CV and generate recommendations"""
        candidate_id = Path(cv_path).stem
        
        try:
            # Parse CV
            parse_start = time.time()
            resume = self.parser.parse(cv_path)
            parse_time = time.time() - parse_start
            
            if not resume:
                return ProcessingResult(
                    candidate_id=candidate_id,
                    success=False,
                    parsing_time=parse_time,
                    matching_time=0,
                    num_recommendations=0,
                    error="Parsing failed"
                )
            
            db_manager.metrics.add_parsing_time(parse_time)
            
            # Match to jobs
            match_start = time.time()
            recommendations = self.matcher.match(resume, candidate_id)
            match_time = time.time() - match_start
            
            db_manager.metrics.add_recommendation_time(match_time)
            
            # Save recommendations
            output_dir = "output/batch"
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/{candidate_id}.json", "w") as f:
                f.write(recommendations.model_dump_json(indent=2))
            
            return ProcessingResult(
                candidate_id=candidate_id,
                success=True,
                parsing_time=parse_time,
                matching_time=match_time,
                num_recommendations=len(recommendations.recommendations)
            )
            
        except Exception as e:
            logger.error(f"Error processing {cv_path}: {e}")
            return ProcessingResult(
                candidate_id=candidate_id,
                success=False,
                parsing_time=0,
                matching_time=0,
                num_recommendations=0,
                error=str(e)
            )
    
    def process_batch(self, cv_paths: List[str]) -> Dict:
        """Process multiple CVs concurrently"""
        logger.info(f"Processing {len(cv_paths)} CVs with {self.max_workers} workers")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single_cv, path): path 
                for path in cv_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        logger.info(f"✓ {result.candidate_id}: "
                                  f"Parse={result.parsing_time:.2f}s, "
                                  f"Match={result.matching_time:.2f}s, "
                                  f"Recs={result.num_recommendations}")
                    else:
                        logger.error(f"✗ {result.candidate_id}: {result.error}")
                        
                except Exception as e:
                    logger.error(f"Exception for {path}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        avg_parse = sum(r.parsing_time for r in successful) / len(successful) if successful else 0
        avg_match = sum(r.matching_time for r in successful) / len(successful) if successful else 0
        throughput = len(successful) / (total_time / 60)  # CVs per minute
        
        summary = {
            "total_cvs": len(cv_paths),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_seconds": round(total_time, 2),
            "avg_parsing_time_seconds": round(avg_parse, 3),
            "avg_matching_time_seconds": round(avg_match, 3),
            "throughput_cvs_per_minute": round(throughput, 2),
            "results": [
                {
                    "candidate_id": r.candidate_id,
                    "success": r.success,
                    "parsing_time": round(r.parsing_time, 3),
                    "matching_time": round(r.matching_time, 3),
                    "num_recommendations": r.num_recommendations,
                    "error": r.error
                }
                for r in results
            ]
        }
        
        # Save summary
        with open("output/batch/processing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Processing Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total CVs: {len(cv_paths)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Throughput: {throughput:.2f} CVs/min")
        logger.info(f"Avg Parse Time: {avg_parse:.3f}s")
        logger.info(f"Avg Match Time: {avg_match:.3f}s")
        logger.info(f"{'='*60}\n")
        
        return summary
