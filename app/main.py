from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import get_settings
from app.core.middleware import LoggingMiddleware, RequestValidationMiddleware
from app.api.photo_realistic_view import router as photo_realistic_view_router
from app.api.product_2d_to_3d import router as product_2d_to_3d_router
from app.api.product_replacement import router as product_replacement_router
from app.api.usdz_to_glb_conversion import router as usdz_to_glb_router 

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    debug=settings.DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestValidationMiddleware)

# Add rate limiting
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests"}
    )

# Include routers with versioning
app.include_router(photo_realistic_view_router, prefix=f"{settings.API_V1_STR}/photo-realistic-view", tags=["Photo Realistic View"])
app.include_router(product_2d_to_3d_router, prefix=f"{settings.API_V1_STR}/product-2d-to-3d", tags=["2D to 3D"])
app.include_router(product_replacement_router, prefix=f"{settings.API_V1_STR}/product-replacement", tags=["Product Replacement"])
app.include_router(usdz_to_glb_router, prefix=f"{settings.API_V1_STR}/usdz-to-glb", tags=["USDZ to GLB Conversion"])

@app.get("/health", tags=["System"])
async def health_check():
    """
    Simple health check endpoint to verify the service is running
    """
    from app.core.health import get_health_status
    return await get_health_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )