#!/usr/bin/env python3
"""Benchmark HNSW vs Sequential Scan"""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendation.database import db_manager
from src.utils import logger
from sqlalchemy import text

def benchmark(description, num_tests=100):
    """Run benchmark"""
    print(f"\n{description}")
    times = []
    
    # Verify index usage FIRST
    if num_tests == 100:  # Only on first call
        print("\nVerifying index usage...")
        usage = db_manager.verify_index_usage()
        if usage.get('using_hnsw'):
            print("  ✓ HNSW index is being used")
        else:
            print("  ✗ WARNING: HNSW index NOT being used!")
    
    for i in range(num_tests):
        start = time.time()
        results = db_manager.search_similar_jobs(
            "Python Machine Learning Engineer with AWS experience",
            k=10
        )
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"  Average: {avg:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
    return avg


def main():
    print("="*70)
    print("HNSW INDEX vs SEQUENTIAL SCAN COMPARISON")
    print("="*70)
    
    # Check database state
    with db_manager.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding;"))
        count = result.scalar()
        print(f"\nTotal embeddings: {count}")
        
        try:
            result = conn.execute(text("SHOW hnsw.ef_search;"))
            ef_search = result.scalar()
            print(f"HNSW ef_search: {ef_search}")
        except:
            print("HNSW ef_search: not set")
    
    # TEST 1: With HNSW index
    print("\n" + "="*70)
    print("TEST 1: WITH HNSW Index")
    print("="*70)
    time_with_hnsw = benchmark("Testing with HNSW index...", num_tests=100)
    
    # TEST 2: Without index
    print("\n" + "="*70)
    print("TEST 2: WITHOUT Index (Sequential Scan)")
    print("="*70)
    with db_manager.engine.connect() as conn:
        conn.execute(text("DROP INDEX IF EXISTS idx_job_embeddings_hnsw;"))
        conn.commit()
    print("HNSW index dropped")
    
    time_without_index = benchmark("Testing without index...", num_tests=100)
    
    # Recreate index
    print("\nRecreating HNSW index...")
    with db_manager.engine.connect() as conn:
        # For datasets:
        # < 10k: m=16, ef_construction=64
        # 10k-100k: m=16, ef_construction=100
        # > 100k: m=32, ef_construction=200
        if count < 10000:
            m, ef = 16, 64
        elif count < 100000:
            m, ef = 16, 100
        else:
            m, ef = 32, 200
            
        conn.execute(text(f"""
            CREATE INDEX idx_job_embeddings_hnsw 
            ON langchain_pg_embedding 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef});
        """))
        conn.execute(text("ANALYZE langchain_pg_embedding;"))
        conn.commit()
    print(f"HNSW index recreated (m={m}, ef_construction={ef})")
    
    # RESULTS
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"WITH HNSW Index:        {time_with_hnsw:.2f} ms")
    print(f"WITHOUT Index (SeqScan): {time_without_index:.2f} ms")
    print("-"*70)
    
    if time_with_hnsw < time_without_index:
        speedup = time_without_index / time_with_hnsw
        improvement = time_without_index - time_with_hnsw
        print(f"\n✓ HNSW Index Performance:")
        print(f"  Speedup: {speedup:.2f}x faster")
        print(f"  Time saved: {improvement:.2f} ms per query")
        print(f"  Efficiency: {((1 - time_with_hnsw/time_without_index) * 100):.1f}% faster")
    else:
        slowdown = time_with_hnsw / time_without_index
        print(f"\n✗ Index is {slowdown:.2f}x SLOWER")
        print("  This should not happen with HNSW!")
        print("  Check pgvector version and configuration")
    
    print("\n" + "="*70)
    print("PERFORMANCE TARGET:")
    print(f"  Dataset size: {count} embeddings")
    if count < 1000:
        print("  Expected speedup: 1.5-2x (small dataset)")
    elif count < 10000:
        print("  Expected speedup: 3-5x (medium dataset)")
    else:
        print("  Expected speedup: 10-50x (large dataset)")
    print("="*70)

if __name__ == "__main__":
    main()
