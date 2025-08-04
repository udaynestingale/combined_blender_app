from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import Dict, Any
import json
import os
import logging
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

class CameraInfo(BaseModel):
    position: Dict[str, float]
    rotation: Dict[str, float]
    focal_length: float = Field(50.0, ge=0.0)

class LightingInfo(BaseModel):
    intensity: float = Field(..., ge=0.0)
    color: Dict[str, float]
    position: Dict[str, float]

class ReplaceProductData(BaseModel):
    object_name: str
    new_material_path: str
    scale: Dict[str, float] = Field(default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0})

class ProductReplacementRequest(BaseModel):
    product_sku_id: str = Field(..., description="Unique identifier for the product")
    glb_image_key: str = Field(..., description="S3 key for input GLB file")
    generated_2d_image_key: str = Field(..., description="S3 key for output 2D image")
    all_masks_key: str = Field(..., description="S3 key for all product masks")
    target_product_mask_key: str = Field(..., description="S3 key for target product mask")
    camera_info: CameraInfo
    lighting_info: LightingInfo
    replace_product_data: ReplaceProductData

    class Config:
        schema_extra = {
            "example": {
                "product_sku_id": "SKU123",
                "glb_image_key": "inputs/scene.glb",
                "generated_2d_image_key": "outputs/render.png",
                "all_masks_key": "outputs/all_masks.png",
                "target_product_mask_key": "outputs/target_mask.png",
                "camera_info": {
                    "position": {"x": 0, "y": 2, "z": -5},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "focal_length": 50.0
                },
                "lighting_info": {
                    "intensity": 1.0,
                    "color": {"r": 1.0, "g": 1.0, "b": 1.0},
                    "position": {"x": 0, "y": 5, "z": 0}
                },
                "replace_product_data": {
                    "object_name": "Product_1",
                    "new_material_path": "materials/new_material.png",
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0}
                }
            }
        }

@router.post("/replaceProduct", status_code=status.HTTP_200_OK)
@track_time(BLENDER_PROCESSING_TIME, {"task_type": "product_replacement"})
async def replace_product(request: ProductReplacementRequest):
    """
    Replace a product in a 3D scene and generate a new render with masks.
    """
    try:
        logger.info(
            "Processing product replacement request",
            extra={
                "sku_id": request.product_sku_id,
                "object_name": request.replace_product_data.object_name
            }
        )

        # Define working directory and paths
        working_dir = os.path.join(settings.BLENDER_SCRIPTS_PATH, 'product_replacement', 'generated_files')
        script_path = os.path.join(settings.BLENDER_SCRIPTS_PATH, 'product_replacement', 'blender_script_camera_public.py')

        # Configure output files
        output_files = [
            OutputFile(
                local_path=os.path.join(working_dir, 'render.png'),
                s3_key=request.generated_2d_image_key,
                file_type='png'
            ),
            OutputFile(
                local_path=os.path.join(working_dir, 'all_masks.png'),
                s3_key=request.all_masks_key,
                file_type='png'
            ),
            OutputFile(
                local_path=os.path.join(working_dir, 'target_mask.png'),
                s3_key=request.target_product_mask_key,
                file_type='png'
            )
        ]

        # Process the request
        processed_files = await process_blender_request_async(
            script_path=script_path,
            input_files=[request.glb_image_key],
            output_files=output_files,
            working_dir=working_dir,
            generate_mask=True,
            camera_json=request.camera_info.dict(),
            lighting_json=request.lighting_info.dict(),
            use_environment_map="studio.exr",
            use_existing_camera=True,
            replace_product=request.replace_product_data.dict()
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

async def process_replacement_async(request: ProductReplacementRequest) -> str:
    """
    Asynchronously process the product replacement request.
    Returns the task ID for tracking.
    """
    try:
        # Set up file paths
        input_file_path = os.path.join(
            settings.BLENDER_SCRIPTS_PATH,
            'product_replacement',
            'input_image.glb'
        )
        blender_script_path = os.path.join(
            settings.BLENDER_SCRIPTS_PATH,
            'product_replacement',
            'blender_script_camera_public.py'
        )
        output_dir = os.path.join(
            settings.BLENDER_SCRIPTS_PATH,
            'product_replacement',
            'generated_files'
        )

        # Download input GLB file
        await s3_service.download_file_async(request.glb_image_key, input_file_path)
        logger.info(f"Downloaded GLB file to {input_file_path}")

        # Prepare Blender parameters
        blender_params = {
            "generate_mask": True,
            "camera_json": request.camera_info.dict(),
            "lighting_json": request.lighting_info.dict(),
            "use_environment_map": "studio.exr",
            "use_existing_camera": True,
            "replace_product": request.replace_product_data.dict()
        }

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
                    # Upload generated files
                    output_files = {
                        "generated_2d_image_key": os.path.join(output_dir, "room_render.png"),
                        "all_masks_key": os.path.join(output_dir, "mask_all_products.png"),
                        "target_product_mask_key": os.path.join(output_dir, f"mask_{request.product_sku_id}.png")
                    }

                    for key, file_path in output_files.items():
                        await s3_service.upload_file_async(file_path, getattr(request, key))
                        logger.info(f"Uploaded {key} to S3")

                    message_body = {
                        "eventType": "ProductReplaced",
                        "productSkuId": request.product_sku_id,
                        "taskId": task_id
                    }
                else:
                    message_body = {
                        "eventType": "ProductReplacementFailed",
                        "productSkuId": request.product_sku_id,
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
    Get the status of a product replacement task.
    """
    try:
        from app.services.blender_service import celery_app
        task = celery_app.AsyncResult(task_id)
        
        if task.ready():
            if task.successful():
                return TaskResponse(
                    task_id=task_id,
                    status="completed",
                    message="Product replacement completed successfully"
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