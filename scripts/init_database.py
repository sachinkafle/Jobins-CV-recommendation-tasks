#!/usr/bin/env python3
"""Initialize database with sample job postings"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, create_engine
from src.utils import logger, config
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector


def clean_database():
    """Clean existing tables"""
    engine = create_engine(config.database_url)
    
    logger.info("Cleaning database...")
    
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;"))
        conn.commit()
        logger.info("✓ Database cleaned")
    
    engine.dispose()


def generate_sample_jobs(num_jobs: int = 10):
    """Generate sample job postings"""
    jobs = []
    
    job_templates = [
        {
            "title": "Senior Machine Learning Engineer",
            "company": "Tech Innovations Inc",
            "description": "Build and deploy ML models for production systems. Work with large datasets and optimize model performance.",
            "requirements": "Required: Python, TensorFlow/PyTorch, AWS, Docker. 5+ years ML experience. Strong math background."
        },
        {
            "title": "Full Stack Developer",
            "company": "StartupXYZ",
            "description": "Develop modern web applications using React and Node.js. Build RESTful APIs and responsive UIs.",
            "requirements": "Required: JavaScript, React, Node.js, MongoDB, Git. 3+ years experience."
        },
        {
            "title": "Data Scientist",
            "company": "Analytics Corp",
            "description": "Analyze large datasets and build predictive models. Create dashboards and reports for stakeholders.",
            "requirements": "Required: Python, Pandas, SQL, Machine Learning, Statistics. 2+ years experience."
        },
        {
            "title": "DevOps Engineer",
            "company": "Cloud Solutions Ltd",
            "description": "Manage cloud infrastructure and CI/CD pipelines. Ensure high availability and scalability.",
            "requirements": "Required: AWS/GCP, Kubernetes, Docker, Terraform, Jenkins. 3+ years experience."
        },
        {
            "title": "Backend Developer",
            "company": "Enterprise Systems",
            "description": "Design and implement scalable backend services. Work with microservices architecture.",
            "requirements": "Required: Java/Python, Spring Boot, PostgreSQL, Redis, REST APIs. 4+ years experience."
        }
    ]
    
    for i in range(num_jobs):
        template = job_templates[i % len(job_templates)]
        jobs.append({
            "job_id": f"job_{str(i+1).zfill(3)}",
            "title": template["title"],
            "company": f"{template['company']} (Office {i+1})",
            "description": template["description"],
            "requirements": template["requirements"]
        })
    
    return jobs


def add_jobs_to_vectorstore(jobs):
    """Add all jobs using from_texts (creates table correctly)"""
    logger.info(f"Adding {len(jobs)} jobs to vector store...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Prepare all texts and metadatas
    texts = [
        f"{job['title']}\n{job['description']}\n{job['requirements']}"
        for job in jobs
    ]
    
    metadatas = [
        {
            'job_id': job['job_id'],
            'title': job['title'],
            'company': job['company'],
            'requirements': job['requirements']
        }
        for job in jobs
    ]
    
    # Use from_texts to create vector store and add all data at once
    vector_store = PGVector.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="job_postings",
        connection_string=config.database_url,
        use_jsonb=True
    )
    
    logger.info(f"✓ Successfully added {len(jobs)} jobs")
    return vector_store


def fix_column_type_and_create_indexes():
    """Fix embedding column type and create indexes"""
    engine = create_engine(config.database_url)
    
    logger.info("Fixing vector column type and creating indexes...")
    
    try:
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding;"))
            count = result.scalar()
            logger.info(f"Found {count} embeddings")
        
        
        logger.info("Setting embedding column to vector(1536)...")
        with engine.connect() as conn:
            conn.execute(text("""
                ALTER TABLE langchain_pg_embedding 
                ALTER COLUMN embedding TYPE vector(1536);
            """))
            conn.commit()
            logger.info("Column type updated to vector(1536)")
        
        #  Create HNSW index (better for <100k rows)
        logger.info("Creating HNSW vector index...")
        with engine.connect() as conn:
            # HNSW parameters:
            # m = max connections per layer (16-64, default 16)
            # ef_construction = size of dynamic candidate list (default 64)
            conn.execute(text("""
                CREATE INDEX idx_job_embeddings_hnsw 
                ON langchain_pg_embedding 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """))
            conn.commit()
            logger.info("HNSW vector index created (m=16, ef_construction=64)")
        
        
        logger.info("Creating metadata index...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX idx_job_metadata 
                ON langchain_pg_embedding 
                USING gin (cmetadata jsonb_path_ops);
            """))
            conn.commit()
            logger.info("✓ Metadata index created")
        
        # Step 5: Verify indexes
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'langchain_pg_embedding'
                ORDER BY indexname;
            """))
            indexes = result.fetchall()
            logger.info(f"✓ Created {len(indexes)} indexes:")
            for idx in indexes:
                logger.info(f"  - {idx[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        return False
    
    finally:
        engine.dispose()


def configure_database_for_hnsw():
    """Configure database for HNSW index usage"""
    engine = create_engine(config.database_url)
    
    logger.info("Configuring database for HNSW index usage...")
    
    try:
        with engine.connect() as conn:
            # Set HNSW ef_search parameter (trade-off between speed and accuracy)
            # Higher = more accurate but slower (default 40)
            # For 2000 rows: 40-100 is good
            conn.execute(text("ALTER DATABASE vectordb SET hnsw.ef_search = 40;"))
            
            # Adjust cost parameters to favor index
            conn.execute(text("ALTER DATABASE vectordb SET random_page_cost = 1.1;"))
            conn.execute(text("ALTER DATABASE vectordb SET seq_page_cost = 1.0;"))
            
            # Update statistics
            conn.execute(text("ANALYZE langchain_pg_embedding;"))
            
            conn.commit()
            logger.info("Configured: hnsw.ef_search=40")
            
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
    
    finally:
        engine.dispose()


# def configure_database_for_ivfflat():
#     """Configure database to prefer IVFFlat index"""
#     engine = create_engine(config.database_url)
    
#     logger.info("Configuring database for IVFFlat index usage...")
    
#     try:
#         with engine.connect() as conn:
#             # Get row count to calculate optimal parameters
#             result = conn.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding;"))
#             count = result.scalar()
            
#             # Calculate optimal probes (lists/10 for < 1M rows)
#             lists = 10  # We created index with lists=10
#             probes = max(1, lists // 2)  # Use lists/2 for better recall
            
#             # Set ivfflat.probes
#             conn.execute(text(f"ALTER DATABASE vectordb SET ivfflat.probes = {probes};"))
            
#             # Adjust cost parameters to favor index
#             conn.execute(text("ALTER DATABASE vectordb SET random_page_cost = 1.1;"))
#             conn.execute(text("ALTER DATABASE vectordb SET seq_page_cost = 1.0;"))
#             conn.execute(text("ALTER DATABASE vectordb SET effective_cache_size = '2GB';"))
            
#             # Update statistics
#             conn.execute(text("ANALYZE langchain_pg_embedding;"))
            
#             conn.commit()
#             logger.info(f"✓ Configured: probes={probes}, optimized for {count} embeddings")
            
#     except Exception as e:
#         logger.error(f"Configuration failed: {e}")
    
#     finally:
#         engine.dispose()


def main():
    logger.info("="*70)
    logger.info("DATABASE INITIALIZATION")
    logger.info("="*70)
    
    
    clean_database()
    
    
    logger.info("Generating sample job postings...")
    jobs = generate_sample_jobs(num_jobs=1000)  
    logger.info(f"Generated {len(jobs)} job postings")
    
    
    add_jobs_to_vectorstore(jobs)
    
    
    success = fix_column_type_and_create_indexes()
    
    
    if success:
        configure_database_for_hnsw()
    
    logger.info("="*70)
    if success:
        logger.info("DATABASE INITIALIZATION COMPLETE WITH HNSW INDEX")
    else:
        logger.info("DATABASE INITIALIZATION COMPLETE (without vector index)")
    logger.info("="*70)



if __name__ == "__main__":
    main()
