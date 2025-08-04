from fastapi import status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.core.config import get_settings

settings = get_settings()

class HealthResponse(BaseModel):
    status: str
    version: str
    port: int

async def get_health_status() -> JSONResponse:
    """
    Simple health check to verify the service is running
    """
    return JSONResponse(
        content=HealthResponse(
            status="up",
            version=settings.VERSION,
            port=settings.PORT
        ).dict(),
        status_code=status.HTTP_200_OK
    )
