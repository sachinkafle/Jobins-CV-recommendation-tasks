#!/usr/bin/env python3
"""Fix vector index by altering column type"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, create_engine
from src.utils import logger, config


def fix_vector_index():
    """Fix embedding column and create vector index"""
    engine = create_engine(config.database_url)
    
    logger.info("Fixing vector index...")
    
    try:
        with engine.connect() as conn:
            # Check current count
            result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding;"))
            count = result.scalar()
            logger.info(f"Found {count} embeddings")
            
            if count == 0:
                logger.error("No data in database!")
                return
        
        # Alter column to add explicit dimensions (in new transaction)
        logger.info("Setting embedding column to vector(1536)...")
        with engine.connect() as conn:
            conn.execute(text("""
                ALTER TABLE langchain_pg_embedding 
                ALTER COLUMN embedding TYPE vector(1536);
            """))
            conn.commit()
            logger.info("✓ Column type updated to vector(1536)")
        
        # Create IVFFlat index (in new transaction)
        logger.info("Creating IVFFlat index...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX idx_job_embeddings_ivfflat 
                ON langchain_pg_embedding 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            conn.commit()
            logger.info("✓ IVFFlat index created successfully!")
        
        # Verify index exists (in new transaction)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'langchain_pg_embedding' 
                AND indexname = 'idx_job_embeddings_ivfflat';
            """))
            
            if result.fetchone():
                logger.info("✓ Index verification successful!")
                logger.info("="*70)
                logger.info("SUCCESS! IVFFlat index is now active")
                logger.info("Run: python scripts/test_search_performance.py")
                logger.info("="*70)
            else:
                logger.error("Index not found after creation")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info("\nTrying manual SQL approach...")
        logger.info("Run these commands in psql:")
        logger.info("  psql -h 127.0.0.1 -p 5433 -U user -d vectordb")
        logger.info("  ALTER TABLE langchain_pg_embedding ALTER COLUMN embedding TYPE vector(1536);")
        logger.info("  CREATE INDEX idx_job_embeddings_ivfflat ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    
    finally:
        engine.dispose()


if __name__ == "__main__":
    fix_vector_index()
