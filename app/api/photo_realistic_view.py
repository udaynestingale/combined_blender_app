from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import json
import os
import logging
from app.core.config import get_settings
from app.services.s3_service import S3Service
from app.services.sqs_service import SQSService
from app.services.blender_service import process_blender_request_async, BlenderError, OutputFile

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize services
s3_service = S3Service(settings.S3_BUCKET_NAME, settings.AWS_REGION)
sqs_service = SQSService(settings.SQS_QUEUE_URL)

class CameraInfo(BaseModel):
    position: Dict[str, float]
    rotation: Dict[str, float]
    focal_length: Optional[float] = Field(default=50.0)

class LightingInfo(BaseModel):
    intensity: float = Field(ge=0.0)
    color: Dict[str, float]
    position: Dict[str, float]

class PhotoRealisticViewRequest(BaseModel):
    template_id: str = Field(..., description="Unique identifier for the template")
    glb_image_key: str = Field(..., description="S3 key for the GLB input file")
    generated_2d_image_key: str = Field(..., description="S3 key for the output 2D image")
    all_masks_key: str = Field(..., description="S3 key for the output masks")
    camera_info: CameraInfo
    lighting_info: LightingInfo

    class Config:
        schema_extra = {
            "example": {
                "template_id": "template123",
                "glb_image_key": "inputs/model.glb",
                "generated_2d_image_key": "outputs/render.png",
                "all_masks_key": "outputs/masks.png",
                "camera_info": {
                    "position": {"x": 0, "y": 2, "z": -5},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "focal_length": 50.0
                },
                "lighting_info": {
                    "intensity": 1.0,
                    "color": {"r": 1.0, "g": 1.0, "b": 1.0},
                    "position": {"x": 0, "y": 5, "z": 0}
                }
            }
        }

@router.post("/generatePhotoRealisticView", status_code=status.HTTP_200_OK)
async def generate_photo_realistic_view(
    request: PhotoRealisticViewRequest
):
    """
    Generate a photo-realistic view of a 3D model with the specified camera and lighting settings.
    Returns the S3 locations of the generated files.
    """
    try:
        logger.info(f"Processing request for template_id: {request.template_id}")

        # Define working directory and paths
        working_dir = os.path.join(settings.BLENDER_SCRIPTS_PATH, 'photo_realistic_view', 'generated_files')
        script_path = os.path.join(settings.BLENDER_SCRIPTS_PATH, 'photo_realistic_view', 'blender_script.py')

        # Configure output files
        output_files = [
            OutputFile(
                local_path=os.path.join(working_dir, 'render.png'),
                s3_key=request.generated_2d_image_key,
                file_type='png'
            ),
            OutputFile(
                local_path=os.path.join(working_dir, 'masks.png'),
                s3_key=request.all_masks_key,
                file_type='png'
            )
        ]

        # Process the request
        processed_files = await process_blender_request_async(
            script_path=script_path,
            input_files=[request.glb_image_key],
            output_files=output_files,
            working_dir=working_dir,
            camera_json=request.camera_info.dict(),
            lighting_json=request.lighting_info.dict(),
            generate_mask=True,
            combined_mask_only=True,
            r=1920
        )

        return {
            "status": "completed",
            "template_id": request.template_id,
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
  
