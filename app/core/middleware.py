import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: Status {response.status_code}, Time: {process_time:.2f}s")
        
        return response

class RequestValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            try:
                body = await request.json()
                logger.debug(f"Request body: {json.dumps(body)}")
            except json.JSONDecodeError:
                logger.error("Invalid JSON in request body")
        
        response = await call_next(request)
        return response
