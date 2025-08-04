from fastapi import FastAPI
from app.api.photo_realistic_view import router as photo_realistic_view_router
from app.api.product_2d_to_3d import router as product_2d_to_3d_router
from app.api.product_replacement import router as product_replacement_router

app = FastAPI()

# Include API routers
app.include_router(photo_realistic_view_router, prefix="/api/photo-realistic-view", tags=["Photo Realistic View"])
app.include_router(product_2d_to_3d_router, prefix="/api/product-2d-to-3d", tags=["2D to 3D"])
app.include_router(product_replacement_router, prefix="/api/product-replacement", tags=["Product Replacement"])