from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json
import os
import logging
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
router = APIRouter()


class Product2DTo3DRequest(BaseModel):
    product_type: str = Field(..., description="Type of product (rug or pillow)")
    product_image_s3_path: str = Field(..., description="S3 path to the main product image")
    product_image_s3_path2: Optional[str] = Field(None, description="S3 path to the secondary image (for pillows)")
    product_sku_id: str = Field(..., description="Unique identifier for the product")
    output_s3_file_key: str = Field(..., description="S3 key for the output GLB file")

    class Config:
        schema_extra = {
            "example": {
                "product_type": "rug",
                "product_image_s3_path": "inputs/product.png",
                "product_sku_id": "SKU123",
                "output_s3_file_key": "outputs/product.glb"
            }
        }

@router.post('/processGlb', status_code=status.HTTP_200_OK)
@track_time(BLENDER_PROCESSING_TIME, {"task_type": "2d_to_3d"})
async def process_glb(request: Product2DTo3DRequest):
    """
    Convert a 2D product image to a 3D GLB model.
    Supports rugs and pillows with different processing requirements.
    """
    try:
        logger.info(
            "Processing 2D to 3D conversion request",
            extra={
                "product_type": request.product_type,
                "sku_id": request.product_sku_id
            }
        )

        # Define working directory and paths
        working_dir = os.path.join(settings.BLENDER_SCRIPTS_PATH, 'product_2d_to_3d')
        script_path = os.path.join(working_dir, 'create_Rug_or_Pillow_GLB_public.py')

        # Configure input and output files
        input_files = [request.product_image_s3_path]
        if request.product_type == "pillow" and request.product_image_s3_path2:
            input_files.append(request.product_image_s3_path2)

        output_files = [
            OutputFile(
                local_path=os.path.join(working_dir, 'output.glb'),
                s3_key=request.output_s3_file_key,
                file_type='glb'
            )
        ]

        # Process the request
        processed_files = await process_blender_request_async(
            script_path=script_path,
            input_files=input_files,
            output_files=output_files,
            working_dir=working_dir,
            product_type=request.product_type
        )

        return {
            "status": "completed",
            "product_sku_id": request.product_sku_id,
            "files": [
                {
                    "type": file.file_type,
                    "s3_key": file.s3_key
                }
                for file in processed_files
            ]
        }

    except BlenderError as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    

async def process_product_async(request: Product2DTo3DRequest) -> str:
    """
    Asynchronously process the 2D to 3D conversion request.
    Returns the task ID for tracking.
    """
    try:
        # Set up file paths
        input_file_path = os.path.join(
            settings.BLENDER_SCRIPTS_PATH,
            'product_2d_to_3d',
            f'input_image_{request.product_sku_id}.png'
        )
        input_file_path2 = None
        if request.product_image_s3_path2:
            input_file_path2 = os.path.join(
                settings.BLENDER_SCRIPTS_PATH,
                'product_2d_to_3d',
                f'input_image2_{request.product_sku_id}.png'
            )

        blender_script_path = os.path.join(
            settings.BLENDER_SCRIPTS_PATH,
            'product_2d_to_3d',
            'create_Rug_or_Pillow_GLB_public.py'
        )
        output_dir = os.path.join(settings.BLENDER_SCRIPTS_PATH, 'product_2d_to_3d')

        # Download input files
        await s3_service.download_file_async(request.product_image_s3_path, input_file_path)
        logger.info(f"Downloaded main image to {input_file_path}")

        if request.product_type == "pillow" and request.product_image_s3_path2:
            await s3_service.download_file_async(request.product_image_s3_path2, input_file_path2)
            logger.info(f"Downloaded secondary image to {input_file_path2}")

        # Prepare Blender parameters
        blender_params = {
            "product_type": request.product_type,
            "output_dir": output_dir
        }
        if input_file_path2:
            blender_params["input_file_path2"] = input_file_path2

        # Start async Blender processing
        task_id = await process_blender_request_async(
            blender_script_path,
            input_file_path,
            output_dir,
            **blender_params
        )

        # Set up completion callback
        async def on_complete(success: bool):
            try:
                if success:
                    output_file_path = os.path.join(output_dir, 'output.glb')
                    await s3_service.upload_file_async(output_file_path, request.output_s3_file_key)
                    logger.info(f"Uploaded output file to {request.output_s3_file_key}")

                    message_body = {
                        "eventType": "twodToThreedFileCreated",
                        "projectId": request.product_sku_id,
                        "taskId": task_id
                    }
                else:
                    message_body = {
                        "eventType": "twodToThreedFileFailed",
                        "projectId": request.product_sku_id,
                        "taskId": task_id
                    }

                await sqs_service.send_message_async(json.dumps(message_body))
                logger.info(f"Sent completion status to SQS: {message_body['eventType']}")

            except Exception as e:
                logger.error(f"Error in completion callback: {str(e)}")
                raise

        # Register the callback
        from app.services.blender_service import register_completion_callback
        register_completion_callback(task_id, on_complete)

        return task_id

    except Exception as e:
        logger.error(f"Error processing request for SKU {request.product_sku_id}: {str(e)}")
        raise

@router.get('/task/{task_id}', response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a 2D to 3D conversion task.
    """
    try:
        from app.services.blender_service import celery_app
        task = celery_app.AsyncResult(task_id)
        
        if task.ready():
            if task.successful():
                return TaskResponse(
                    task_id=task_id,
                    status="completed",
                    message="Conversion completed successfully"
                )
            else:
                return TaskResponse(
                    task_id=task_id,
                    status="failed",
                    message=str(task.result)
                )
        else:
            return TaskResponse(
                task_id=task_id,
                status="processing",
                message="Task is still processing"
            )

    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
  
