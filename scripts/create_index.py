#!/usr/bin/env python3
"""Manually create vector indexes on existing data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.recommendation.database import db_manager
from src.utils import logger


def create_vector_indexes():
    """Create vector indexes on existing embeddings"""
    
    logger.info("Creating vector indexes on existing data...")
    
    # Check data exists
    with db_manager.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding;"))
        count = result.scalar()
        
        if count == 0:
            logger.error("No embeddings found! Run init_database.py first")
            return
        
        logger.info(f"Found {count} embeddings")
    
    # Try IVFFlat first (more stable than HNSW)
    logger.info("Creating IVFFlat index...")
    try:
        with db_manager.engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_job_embeddings_ivfflat 
                ON langchain_pg_embedding 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            conn.commit()
            logger.info("✓ IVFFlat index created successfully!")
            return True
    except Exception as e:
        logger.error(f"IVFFlat index failed: {e}")
        return False


if __name__ == "__main__":
    success = create_vector_indexes()
    
    if success:
        print("\n" + "="*70)
        print("✓ Vector index created successfully!")
        print("  Run the performance test again to see improvement:")
        print("  python scripts/test_search_performance.py")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ Failed to create vector index")
        print("  Check the error messages above")
        print("="*70)
