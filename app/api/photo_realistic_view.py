from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import json
import os
import logging
from app.core.config import get_settings
from app.services.s3_service import S3Service
from app.services.sqs_service import SQSService
from app.services.blender_service import process_blender_request_async, BlenderError, OutputFile
from app.utils.file_utils import cleanup_processing_files

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
        
        # Create working directory if it doesn't exist
        os.makedirs(working_dir, exist_ok=True)
        
        # Download input GLB file from S3
        input_file_local_path = os.path.join(working_dir, 'input', os.path.basename(request.glb_image_key))
        os.makedirs(os.path.dirname(input_file_local_path), exist_ok=True)
        
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
                local_path=os.path.join(working_dir, 'masks.png'),
                s3_key=request.all_masks_key,
                file_type='png'
            )
        ]

        # Construct Blender command as a list of arguments
        blender_path = settings.BLENDER_PATH if hasattr(settings, 'BLENDER_PATH') else 'blender'
        blender_command = [
            blender_path,
            "--background",
            "--python", script_path,
            "--",  # Argument separator
            input_file_local_path,  # Local path to input file instead of S3 key
            "-d", working_dir,  # Working directory
            f"--camera_json", json.dumps(request.camera_info.dict()),
            f"--lighting_json", json.dumps(request.lighting_info.dict()),
            f"--generate_mask", json.dumps(True),
            f"--combined_mask_only", json.dumps(True),
            f"--r", json.dumps(1920)
        ]

        # Process the request with the new approach
        processed_files = await process_blender_request_async(
            working_dir=working_dir,
            blender_command=blender_command,
            output_files=output_files
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
    finally:
        # Clean up the downloaded input file and local output files
        await cleanup_processing_files(
            input_files=input_file_local_path, 
            output_files=output_files,
            working_dir=working_dir
        )

# The cleanup_files function is now in utils.file_utils module
