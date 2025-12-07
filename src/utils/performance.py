"""Performance monitoring utilities"""
import time
from functools import wraps
from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class SystemMetrics:
    """System performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return np.mean(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0.0

    @property
    def throughput(self) -> float:
        total_time = sum(self.latencies)
        return (self.successful_requests / total_time * 60) if total_time > 0 else 0.0

    def add_request(self, latency: float, success: bool = True):
        """Record a request"""
        self.total_requests += 1
        self.latencies.append(latency)
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

class PerformanceMonitor:
    """Monitor system performance"""

    def __init__(self):
        self.metrics = SystemMetrics()

    def measure(self, func):
        """Decorator to measure execution time"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                self.metrics.add_request(time.time() - start, True)
                return result
            except Exception as e:
                self.metrics.add_request(time.time() - start, False)
                raise e

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                self.metrics.add_request(time.time() - start, True)
                return result
            except Exception as e:
                self.metrics.add_request(time.time() - start, False)
                raise e

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def get_report(self) -> dict:
        """Generate performance report"""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "avg_latency_sec": round(self.metrics.avg_latency, 4),
            "p95_latency_sec": round(self.metrics.p95_latency, 4),
            "throughput_per_min": round(self.metrics.throughput, 0)
        }

# Global monitor instance
monitor = PerformanceMonitor()
