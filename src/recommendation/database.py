"""Optimized database manager with caching and batch processing"""
import time
from typing import List, Dict, Optional
from functools import lru_cache
import hashlib
from langchain_community.vectorstores.pgvector import PGVector
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
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        
        self.Session = sessionmaker(bind=self.engine)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # IMPORTANT: Don't initialize vector_store here!
        # It will be created on-demand to avoid schema conflicts
        self._vector_store = None
        
        # Redis cache
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
    
    # @property
    # def vector_store(self):
    #     """Lazy initialization of vector store"""
    #     if self._vector_store is None:
    #         self._vector_store = PGVector(
    #             embedding_function=self.embeddings,
    #             collection_name="job_postings",
    #             connection_string=config.database_url,
    #             use_jsonb=True
    #         )
    #     return self._vector_store
    
    @property
    def vector_store(self):
        """Lazy initialization of vector store"""
        if self._vector_store is None:
            self._vector_store = PGVector(
                embedding_function=self.embeddings,  
                collection_name="job_postings",
                connection_string=config.database_url,
                use_jsonb=True
            )
        return self._vector_store
    
    def search_similar_jobs(self, query: str, k: int = 10, filters: Dict = None) -> List[Dict]:
        """Search for similar jobs using direct SQL to ensure index usage"""
        try:
            start_time = time.time()
            
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Direct SQL query that WILL use the HNSW index
            sql = f"""
                SELECT 
                    uuid,
                    document,
                    cmetadata,
                    embedding <=> '{embedding_str}'::vector(1536) AS distance
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection 
                    WHERE name = 'job_postings' 
                    LIMIT 1
                )
                ORDER BY embedding <=> '{embedding_str}'::vector(1536)
                LIMIT {k};
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
            
            # Format results
            jobs = []
            for row in rows:
                # Check if metadata is already a dict or needs parsing
                metadata = row[2]
                if isinstance(metadata, str):
                    import json
                    metadata = json.loads(metadata)
                
                job = {
                    'job_id': metadata.get('job_id', 'unknown'),
                    'title': metadata.get('title', 'Unknown Title'),
                    'company': metadata.get('company', 'Unknown Company'),
                    'requirements': metadata.get('requirements', ''),
                    'content': row[1],  # document
                    'similarity_score': float(1 - row[3])  # 1 - distance
                }
                jobs.append(job)
            
            search_time = time.time() - start_time
            self.metrics.add_search_time(search_time)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []



    def verify_index_usage(self) -> Dict:
        """Verify HNSW index is being used"""
        try:
            # Generate test embedding
            test_embedding = self.embeddings.embed_query("Python Machine Learning")
            embedding_str = '[' + ','.join(map(str, test_embedding)) + ']'
            
            # Check query plan
            sql = f"""
                EXPLAIN (ANALYZE, BUFFERS)
                SELECT 
                    uuid,
                    embedding <=> '{embedding_str}'::vector(1536) AS distance
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection 
                    WHERE name = 'job_postings' 
                    LIMIT 1
                )
                ORDER BY embedding <=> '{embedding_str}'::vector(1536)
                LIMIT 10;
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                plan_lines = [row[0] for row in result]
            
            plan_text = '\n'.join(plan_lines)
            using_hnsw = 'idx_job_embeddings_hnsw' in plan_text.lower()
            using_seqscan = 'seq scan' in plan_text.lower()
            
            logger.info("="*70)
            logger.info("INDEX USAGE VERIFICATION")
            logger.info("="*70)
            
            if using_hnsw:
                logger.info("✓ HNSW index IS being used!")
            elif using_seqscan:
                logger.warning("✗ Sequential scan detected (index NOT used)")
            else:
                logger.info("? Unknown - check plan manually")
            
            logger.info("\nQuery Plan:")
            logger.info(plan_text)
            logger.info("="*70)
            
            return {
                'using_hnsw': using_hnsw,
                'using_seqscan': using_seqscan,
                'plan': plan_text
            }
            
        except Exception as e:
            logger.error(f"Error verifying index: {e}")
            return {'error': str(e)}



    # def check_index_usage(self) -> Dict:
    #     """Check if IVFFlat index is being used"""
    #     try:
    #         # Generate a test embedding
    #         test_query = "Python Machine Learning Engineer"
    #         query_embedding = self.embeddings.embed_query(test_query)
    #         embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
    #         with self.engine.connect() as conn:
    #             result = conn.execute(text(f"""
    #                 EXPLAIN 
    #                 SELECT uuid, 
    #                     embedding <=> '{embedding_str}'::vector(1536) AS distance
    #                 FROM langchain_pg_embedding
    #                 ORDER BY distance
    #                 LIMIT 5;
    #             """))
                
    #             using_ivfflat = False
    #             using_seqscan = False
    #             plan_lines = []
                
    #             for row in result:
    #                 plan_line = row[0]
    #                 plan_lines.append(plan_line)
                    
    #                 if 'idx_job_embeddings_ivfflat' in plan_line.lower():
    #                     using_ivfflat = True
    #                 if 'seq scan' in plan_line.lower():
    #                     using_seqscan = True
                
    #             return {
    #                 'using_ivfflat': using_ivfflat,
    #                 'using_seqscan': using_seqscan,
    #                 'plan': '\n'.join(plan_lines)
    #             }
                
    #     except Exception as e:
    #         logger.error(f"Error checking index: {e}")
    #         return {'error': str(e)}

    
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

    def optimize_database(self):
        """Run database optimization tasks"""
        logger.info("Optimizing database...")
        
        try:
            with self.engine.connect() as conn:
                # Analyze tables for query planner
                conn.execute(text("ANALYZE langchain_pg_embedding;"))
                conn.commit()
                logger.info("✓ Database statistics updated")
            
            # Create indexes if they don't exist yet
            self._create_indexes()
            
            logger.info("✓ Database optimization complete")
            
        except Exception as e:
            logger.warning(f"Database optimization warning: {e}")



# Global instance
db_manager = OptimizedDatabaseManager()
