from prometheus_client import Counter, Histogram, Info
import time
import asyncio
from functools import wraps
from typing import Callable, Any

# Metrics
REQUEST_COUNT = Counter(
    'app_request_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

BLENDER_PROCESSING_TIME = Histogram(
    'app_blender_processing_seconds',
    'Time spent processing Blender tasks',
    ['task_type']
)

S3_OPERATION_LATENCY = Histogram(
    'app_s3_operation_latency_seconds',
    'S3 operation latency in seconds',
    ['operation']
)

APP_INFO = Info('app_info', 'Application information')
APP_INFO.info({
    'version': '1.0.0',
    'name': 'Nestingale Blender API'
})

def track_time(metric: Histogram, labels: dict) -> Callable:
    """
    Decorator to track execution time of a function using a Prometheus histogram.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
