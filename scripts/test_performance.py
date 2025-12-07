"""Performance testing script"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch_processor import BatchCVProcessor
from src.recommendation.database import db_manager
from src.utils import logger


def test_load_performance():
    """Test system performance under load"""
    
    # Test configuration
    test_cvs_dir = "data/sample_cvs"  # Directory with test CVs
    num_cvs = 100  # Test with 100 CVs
    max_workers = 10  # Concurrent workers
    
    logger.info(f"Performance Test Configuration:")
    logger.info(f"  CVs to process: {num_cvs}")
    logger.info(f"  Concurrent workers: {max_workers}")
    logger.info(f"  Test CVs directory: {test_cvs_dir}\n")
    
    # Get CV files
    cv_files = list(Path(test_cvs_dir).glob("*.pdf"))[:num_cvs]
    
    if len(cv_files) < num_cvs:
        logger.warning(f"Only found {len(cv_files)} CVs, expected {num_cvs}")
    
    # Reset metrics
    db_manager.reset_metrics()
    
    # Run batch processing
    processor = BatchCVProcessor(max_workers=max_workers)
    summary = processor.process_batch([str(p) for p in cv_files])
    
    # Get database performance metrics
    db_metrics = db_manager.get_performance_report()
    
    # Combined report
    report = {
        "test_configuration": {
            "target_cvs": num_cvs,
            "actual_cvs": len(cv_files),
            "max_workers": max_workers
        },
        "processing_performance": summary,
        "database_performance": db_metrics
    }
    
    # Print report
    print("\n" + "="*80)
    print("PERFORMANCE TEST REPORT")
    print("="*80)
    print(f"\nProcessing Performance:")
    print(f"  Total CVs Processed: {summary['successful']}/{summary['total_cvs']}")
    print(f"  Success Rate: {summary['successful']/summary['total_cvs']*100:.1f}%")
    print(f"  Total Time: {summary['total_time_seconds']:.2f}s")
    print(f"  Throughput: {summary['throughput_cvs_per_minute']:.2f} CVs/min")
    print(f"  Avg Parsing Time: {summary['avg_parsing_time_seconds']:.3f}s")
    print(f"  Avg Matching Time: {summary['avg_matching_time_seconds']:.3f}s")
    
    print(f"\nDatabase Performance:")
    print(f"  Avg Search Time: {db_metrics['avg_search_time_seconds']:.3f}s")
    print(f"  Cache Hit Rate: {db_metrics['cache_hit_rate']*100:.1f}%")
    print(f"  Total Searches: {db_metrics['total_searches']}")
    
    print("\n" + "="*80)
    
    # Save full report
    import json
    with open("output/performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to: output/performance_report.json")


if __name__ == "__main__":
    test_load_performance()
