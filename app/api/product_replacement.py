from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List
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
        
        # Create working directory if it doesn't exist
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(os.path.join(working_dir, 'input'), exist_ok=True)
        
        # Download input GLB file from S3
        input_file_local_path = os.path.join(working_dir, 'input', os.path.basename(request.glb_image_key))
        
        logger.info(f"Downloading input file from S3: {request.glb_image_key} to {input_file_local_path}")
        await s3_service.download_file_async(request.glb_image_key, input_file_local_path)
        logger.info(f"Successfully downloaded input file")

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
        # Construct Blender command as a list of arguments
        blender_path = settings.BLENDER_PATH if hasattr(settings, 'BLENDER_PATH') else 'blender'
        blender_command = [
            blender_path,
            "--background",
            "--python", script_path,
            "--",  # Argument separator
            input_file_local_path,  # Local path to input file instead of S3 key
            "-d", working_dir,  # Working directory
            f"--generate_mask", json.dumps(True),
            f"--camera_json", json.dumps(request.camera_info.dict()),
            f"--lighting_json", json.dumps(request.lighting_info.dict()),
            f"--use_environment_map", json.dumps("studio.exr"),
            f"--use_existing_camera", json.dumps(True),
            f"--replace_product", json.dumps(request.replace_product_data.dict())
        ]

        # Process the request with the new approach
        processed_files = await process_blender_request_async(
            working_dir=working_dir,
            blender_command=blender_command,
            output_files=output_files
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
    finally:
        # Clean up the downloaded input file and local output files
        try:
            # Clean up input file
            if os.path.exists(input_file_local_path):
                os.remove(input_file_local_path)
                logger.info(f"Cleaned up input file: {input_file_local_path}")
            
            # Clean up output files - they should be already uploaded to S3
            for output_file in output_files:
                if os.path.exists(output_file.local_path):
                    os.remove(output_file.local_path)
                    logger.info(f"Cleaned up output file: {output_file.local_path}")
                    
            # Clean up any other temporary files in the working directory
            input_dir = os.path.join(working_dir, 'input')
            if os.path.exists(input_dir) and os.path.isdir(input_dir):
                for filename in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up temporary file: {file_path}")
                        
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")


async def cleanup_files(*file_paths: str):
    """
    Clean up temporary files after processing.
    """
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")