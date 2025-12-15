
"""Test vector search performance"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendation.database import db_manager
from src.utils import logger


def test_search_performance(num_tests=1000):
    """Test vector search performance"""
    
    logger.info(f"Running {num_tests} search performance tests...")
    
    # Test queries
    test_queries = [
        "Python Machine Learning Engineer",
        "Full Stack Developer React Node.js",
        "Data Scientist SQL Python",
        "DevOps Engineer Kubernetes AWS",
        "Backend Developer Java Spring Boot"
    ]
    
    search_times = []
    
    for i in range(num_tests):
        query = test_queries[i % len(test_queries)]
        
        start = time.time()
        results = db_manager.search_similar_jobs(query, k=5)
        elapsed = time.time() - start
        
        search_times.append(elapsed)
        
        if i == 0:
            logger.info(f"First search returned {len(results)} results in {elapsed*1000:.2f}ms")
    
    # Calculate statistics
    avg_time = sum(search_times) / len(search_times)
    min_time = min(search_times)
    max_time = max(search_times)
    
    print("\n" + "="*70)
    print("VECTOR SEARCH PERFORMANCE TEST")
    print("="*70)
    print(f"Number of tests:      {num_tests}")
    print(f"Average search time:  {avg_time*1000:.2f} ms")
    print(f"Min search time:      {min_time*1000:.2f} ms")
    print(f"Max search time:      {max_time*1000:.2f} ms")
    print("="*70)
    
    # Interpret results
    print("\nPERFORMANCE INTERPRETATION:")
    if avg_time < 0.1:
        print("✓ EXCELLENT - Using HNSW index (<100ms)")
        performance = "HNSW Index"
    elif avg_time < 0.2:
        print("✓ GOOD - Using IVFFlat index (<200ms)")
        performance = "IVFFlat Index"
    elif avg_time < 0.5:
        print("⚠ MODERATE - Sequential scan on small dataset (<500ms)")
        performance = "Sequential Scan (acceptable for <1000 jobs)"
    else:
        print("✗ SLOW - Sequential scan on larger dataset (>500ms)")
        performance = "Sequential Scan (needs index!)"
    
    print(f"\nLikely using: {performance}")
    print("="*70)
    
    # Check database indexes
    print("\nDATABASE INDEX STATUS:")
    from sqlalchemy import text
    
    try:
        with db_manager.engine.connect() as conn:
            # Count embeddings
            result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding;"))
            count = result.scalar()
            print(f"Total embeddings: {count}")
            
            # Check indexes
            result = conn.execute(text("""
                SELECT 
                    indexname, 
                    indexdef 
                FROM pg_indexes 
                WHERE tablename = 'langchain_pg_embedding'
                ORDER BY indexname;
            """))
            indexes = result.fetchall()
            
            if indexes:
                print(f"\nIndexes found ({len(indexes)}):")
                for idx in indexes:
                    index_name = idx[0]
                    index_def = idx[1]
                    
                    if 'hnsw' in index_def.lower():
                        print(f"  ✓ {index_name} - HNSW (fastest)")
                    elif 'ivfflat' in index_def.lower():
                        print(f"  ✓ {index_name} - IVFFlat (fast)")
                    elif 'gin' in index_def.lower():
                        print(f"  ✓ {index_name} - GIN metadata index")
                    else:
                        print(f"  - {index_name}")
            else:
                print("\n⚠ NO VECTOR INDEXES FOUND")
                print("  → Using sequential scan (slower)")
                print("  → Run: python scripts/create_indexes_now.py")
    
    except Exception as e:
        logger.error(f"Error checking indexes: {e}")
    
    print("="*70)
    
    # Cache performance
    if db_manager.cache_enabled:
        cache_hit_rate = db_manager.metrics.cache_hit_rate
        print(f"\nCache hit rate: {cache_hit_rate*100:.1f}%")
        if cache_hit_rate > 0:
            print("✓ Redis cache is working")
    
    print()


if __name__ == "__main__":
    test_search_performance(num_tests=100)
