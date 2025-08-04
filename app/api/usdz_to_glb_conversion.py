
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import json
import os
import logging
import mimetypes
from typing import Optional

from app.core.config import get_settings
from app.core.monitoring import track_time, BLENDER_PROCESSING_TIME
from app.services.s3_service import S3Service, S3ServiceError
from app.services.sqs_service import SQSService
from app.services.blender_service import process_blender_request_async, BlenderError, OutputFile

settings = get_settings()
logger = logging.getLogger(__name__)

s3_service = S3Service(settings.S3_BUCKET_NAME, settings.AWS_REGION)
sqs_service = SQSService(settings.SQS_QUEUE_URL)
router = APIRouter(prefix="/api/v1/conversion", tags=["File Conversion"])

class UsdzToGlbRequest(BaseModel):
    input_file_key: str = Field(..., description="S3 key for the input USDZ file")
    output_file_key: str = Field(..., description="S3 key for the output GLB file")

    @validator('input_file_key')
    def validate_input_file(cls, v):
        if not v.lower().endswith('.usdz'):
            raise ValueError("Input file must be a USDZ file")
        return v

    @validator('output_file_key')
    def validate_output_file(cls, v):
        if not v.lower().endswith('.glb'):
            raise ValueError("Output file must be a GLB file")
        return v

    class Config:
        schema_extra = {
            "example": {
                "input_file_key": "models/input_model.usdz",
                "output_file_key": "converted/output_model.glb"
            }
        }



@router.get('/health')
@track_time(BLENDER_PROCESSING_TIME, {"task_type": "health_check"})
async def health_check():
    """
    Check the health status of the conversion service.
    """
    try:
        # Test connection to S3
        await s3_service.check_connection()
        # Test connection to SQS
        await sqs_service.check_connection()
        
        return JSONResponse(
            content={
                "status": "healthy",
                "message": "Service is running correctly",
                "components": {
                    "s3": "connected",
                    "sqs": "connected",
                    "blender": "available"
                }
            },
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "message": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@router.post('/convert', status_code=status.HTTP_200_OK)
@track_time(BLENDER_PROCESSING_TIME, {"task_type": "usdz_to_glb"})
async def convert_usdz_to_glb(request: UsdzToGlbRequest):
    """
    Convert a USDZ file to GLB format.
    The conversion is processed asynchronously and returns a task ID for tracking progress.
    """
    try:
        logger.info(
            "Starting USDZ to GLB conversion",
            extra={
                "input_file": request.input_file_key,
                "output_file": request.output_file_key
            }
        )

        # Validate file existence in S3
        if not await s3_service.file_exists(request.input_file_key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Input file {request.input_file_key} not found in S3"
            )

        # Start async conversion
        task_id = await process_conversion_async(request)

        return ConversionResponse(
            task_id=task_id,
            status="processing",
            message="Conversion started successfully",
            input_file=request.input_file_key,
            output_file=request.output_file_key
        )

    except S3ServiceError as e:
        logger.error(f"S3 error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"S3 service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Conversion request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def process_conversion_async(request: UsdzToGlbRequest) -> str:
    """
    Asynchronously process the USDZ to GLB conversion request.
    Returns a task ID that can be used to track the conversion progress.
    """
    try:
        # Set up file paths using settings
        work_dir = os.path.join(settings.BLENDER_SCRIPTS_PATH, 'usdz_to_glb_conversion')
        input_file_path = os.path.join(work_dir, f'input_{os.path.basename(request.input_file_key)}')
        output_file_path = os.path.join(work_dir, f'output_{os.path.basename(request.output_file_key)}')
        blender_script_path = os.path.join(work_dir, 'blender_with_furniture.py')

        # Ensure work directory exists
        os.makedirs(work_dir, exist_ok=True)

        # Download input file
        await s3_service.download_file_async(
            request.input_file_key,
            input_file_path
        )
        logger.info(f"Downloaded input file to {input_file_path}")

        # Start async Blender processing
        task_id = await process_blender_request_async(
            script_path=blender_script_path,
            input_file=input_file_path,
            output_dir=work_dir,
            output_file_path=output_file_path
        )

        # Register completion callback
        async def on_complete(success: bool):
            try:
                if success:
                    # Upload converted file
                    await s3_service.upload_file_async(
                        output_file_path,
                        request.output_file_key
                    )
                    logger.info(f"Uploaded converted file to {request.output_file_key}")

                    message_body = {
                        "eventType": "usdzToGlbConversionCompleted",
                        "taskId": task_id,
                        "inputFileKey": request.input_file_key,
                        "outputFileKey": request.output_file_key
                    }
                else:
                    message_body = {
                        "eventType": "usdzToGlbConversionFailed",
                        "taskId": task_id,
                        "inputFileKey": request.input_file_key,
                        "outputFileKey": request.output_file_key,
                        "error": "Conversion failed"
                    }

                # Send status message
                await sqs_service.send_message_async(json.dumps(message_body))
                logger.info(f"Sent {message_body['eventType']} message to SQS")

                # Clean up
                for file_path in [input_file_path, output_file_path]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up {file_path}")

            except Exception as e:
                logger.error(f"Error in completion callback: {str(e)}")
                raise

        # Register the callback
        from app.services.blender_service import register_completion_callback
        register_completion_callback(task_id, on_complete)

        return task_id

    except Exception as e:
        logger.error(f"Error processing conversion: {str(e)}")
        raise

@router.get('/task/{task_id}', response_model=ConversionStatus)
@track_time(BLENDER_PROCESSING_TIME, {"task_type": "status_check"})
async def get_conversion_status(task_id: str):
    """
    Get the status of a conversion task.
    """
    try:
        from app.services.blender_service import celery_app
        task = celery_app.AsyncResult(task_id)
        
        if task.ready():
            if task.successful():
                return ConversionStatus(
                    task_id=task_id,
                    status="completed",
                    message="Conversion completed successfully",
                    progress=100.0
                )
            else:
                return ConversionStatus(
                    task_id=task_id,
                    status="failed",
                    message="Conversion failed",
                    error=str(task.result)
                )
        else:
            # Get progress from task metadata if available
            progress = None
            if hasattr(task, 'info') and task.info:
                progress = task.info.get('progress')

            return ConversionStatus(
                task_id=task_id,
                status="processing",
                message="Conversion in progress",
                progress=progress
            )

    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )