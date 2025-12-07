"""Optimized database manager with caching and batch processing"""
import time
from typing import List, Dict, Optional
from functools import lru_cache
import hashlib
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, text, pool
from sqlalchemy.orm import sessionmaker
import redis
import pickle

from ..utils import config, logger


class PerformanceMetrics:
    """Track performance metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.parsing_times = []
        self.recommendation_times = []
        self.search_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def add_parsing_time(self, duration: float):
        self.parsing_times.append(duration)
    
    def add_recommendation_time(self, duration: float):
        self.recommendation_times.append(duration)
    
    def add_search_time(self, duration: float):
        self.search_times.append(duration)
    
    @property
    def avg_parsing_time(self) -> float:
        return sum(self.parsing_times) / len(self.parsing_times) if self.parsing_times else 0
    
    @property
    def avg_recommendation_time(self) -> float:
        return sum(self.recommendation_times) / len(self.recommendation_times) if self.recommendation_times else 0
    
    @property
    def avg_search_time(self) -> float:
        return sum(self.search_times) / len(self.search_times) if self.search_times else 0
    
    @property
    def throughput(self) -> float:
        """CVs processed per minute"""
        if not self.parsing_times:
            return 0
        total_time = sum(self.parsing_times) / 60  # Convert to minutes
        return len(self.parsing_times) / total_time if total_time > 0 else 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0
    
    def report(self) -> Dict:
        return {
            "avg_parsing_time_seconds": round(self.avg_parsing_time, 3),
            "avg_recommendation_time_seconds": round(self.avg_recommendation_time, 3),
            "avg_search_time_seconds": round(self.avg_search_time, 3),
            "throughput_cvs_per_minute": round(self.throughput, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "total_cvs_processed": len(self.parsing_times),
            "total_recommendations_generated": len(self.recommendation_times),
            "total_searches": len(self.search_times)
        }


class OptimizedDatabaseManager:
    """Database manager with performance optimizations"""
    
    def __init__(self):
        # Connection pooling for concurrent requests
        self.engine = create_engine(
            config.database_url,
            poolclass=pool.QueuePool,
            pool_size=20,  # Max 20 concurrent connections
            max_overflow=10,  # Allow 10 additional connections
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False
        )
        
        self.Session = sessionmaker(bind=self.engine)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize vector store with HNSW index for fast search
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="job_postings",
            connection=config.database_url,
            use_jsonb=True
        )
        
        # Redis cache for embeddings and search results
        try:
            self.cache = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=False,
                socket_timeout=5
            )
            self.cache.ping()
            self.cache_enabled = True
            logger.info("Redis cache connected")
        except:
            self.cache = None
            self.cache_enabled = False
            logger.warning("Redis not available, caching disabled")
        
        self.metrics = PerformanceMetrics()
        
        # Create optimized indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        with self.engine.connect() as conn:
            try:
                # First, ensure pgvector extension is installed
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
                
                # Check if the table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'langchain_pg_embedding'
                    );
                """))
                
                table_exists = result.scalar()
                
                if not table_exists:
                    logger.info("langchain_pg_embedding table doesn't exist yet, skipping index creation")
                    return
                
                # Check if embedding column has data
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM langchain_pg_embedding;
                """))
                
                count = result.scalar()
                
                if count == 0:
                    logger.info("No embeddings in database yet, skipping HNSW index")
                    return
                
                logger.info(f"Creating indexes on {count} embeddings...")
                
                # Try to create HNSW index (requires pgvector extension)
                try:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_job_embeddings_hnsw 
                        ON langchain_pg_embedding 
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64);
                    """))
                    logger.info("HNSW index created successfully")
                except Exception as e:
                    logger.warning(f"HNSW index creation failed (will use slower search): {e}")
                    # Fallback: try IVFFlat index
                    try:
                        conn.execute(text("""
                            CREATE INDEX IF NOT EXISTS idx_job_embeddings_ivfflat 
                            ON langchain_pg_embedding 
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                        """))
                        logger.info("IVFFlat index created as fallback")
                    except Exception as e2:
                        logger.warning(f"IVFFlat index also failed: {e2}")
                
                # B-tree index for metadata filtering (safe to create anytime)
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_job_metadata 
                    ON langchain_pg_embedding 
                    USING gin (cmetadata jsonb_path_ops);
                """))
                
                conn.commit()
                logger.info("Database indexes setup complete")
                
            except Exception as e:
                logger.error(f"Index creation error: {e}")
                conn.rollback()

    
    def _cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        if not self.cache_enabled:
            return None
        
        key = self._cache_key(text)
        try:
            cached = self.cache.get(key)
            if cached:
                self.metrics.cache_hits += 1
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        self.metrics.cache_misses += 1
        return None
    
    def _set_cached_embedding(self, text: str, embedding: List[float]):
        """Store embedding in cache"""
        if not self.cache_enabled:
            return
        
        key = self._cache_key(text)
        try:
            self.cache.setex(
                key,
                3600,  # 1 hour TTL
                pickle.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached:
            return cached
        
        # Generate new embedding
        embedding = self.embeddings.embed_query(text)
        
        # Cache it
        self._set_cached_embedding(text, embedding)
        
        return embedding
    
    def search_similar_jobs(
        self, 
        query: str, 
        k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Optimized vector similarity search"""
        start_time = time.time()
        
        try:
            # Use vector store's optimized search
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_metadata
            )
            
            # Convert to dict format
            jobs = []
            for doc, score in results:
                job_data = doc.metadata.copy()
                job_data['similarity_score'] = float(score)
                job_data['description'] = doc.page_content
                jobs.append(job_data)
            
            search_time = time.time() - start_time
            self.metrics.add_search_time(search_time)
            
            logger.info(f"Search completed in {search_time:.3f}s")
            return jobs
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def batch_add_jobs(self, jobs: List[Dict], batch_size: int = 100):
        """Add jobs in batches for efficiency"""
        logger.info(f"Adding {len(jobs)} jobs in batches of {batch_size}")
        
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            
            texts = [f"{job['title']}\n{job['description']}\n{job['requirements']}" 
                    for job in batch]
            
            metadatas = [{
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'requirements': job['requirements']
            } for job in batch]
            
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added batch {i//batch_size + 1}/{(len(jobs)-1)//batch_size + 1}")
    
    def batch_search(
        self, 
        queries: List[str], 
        k: int = 10
    ) -> List[List[Dict]]:
        """Batch search for multiple queries"""
        start_time = time.time()
        
        results = []
        for query in queries:
            jobs = self.search_similar_jobs(query, k=k)
            results.append(jobs)
        
        search_time = time.time() - start_time
        logger.info(f"Batch search of {len(queries)} queries in {search_time:.3f}s")
        
        return results
    
    def get_performance_report(self) -> Dict:
        """Get performance metrics report"""
        return self.metrics.report()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics.reset()


# Global instance
db_manager = OptimizedDatabaseManager()
