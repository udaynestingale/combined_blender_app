from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import json
import os
import logging
from typing import Optional, List

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
        
        # Create working directory if it doesn't exist
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(os.path.join(working_dir, 'input'), exist_ok=True)

        # Download input files from S3
        input_files_s3 = [request.product_image_s3_path]
        if request.product_type == "pillow" and request.product_image_s3_path2:
            input_files_s3.append(request.product_image_s3_path2)
            
        local_input_files = []
        for i, s3_path in enumerate(input_files_s3):
            file_name = f"input_{i}_{os.path.basename(s3_path)}"
            local_path = os.path.join(working_dir, 'input', file_name)
            
            logger.info(f"Downloading input file from S3: {s3_path} to {local_path}")
            await s3_service.download_file_async(s3_path, local_path)
            local_input_files.append(local_path)
            logger.info(f"Successfully downloaded input file {i+1}")

        # Configure output files
        output_files = [
            OutputFile(
                local_path=os.path.join(working_dir, 'output.glb'),
                s3_key=request.output_s3_file_key,
                file_type='glb'
            )
        ]

        # Construct Blender command as a list of arguments
        blender_path = settings.BLENDER_PATH if hasattr(settings, 'BLENDER_PATH') else 'blender'
        blender_command = [
            blender_path,
            "--background",
            "--python", script_path,
            "--",  # Argument separator
        ]
        
        # Add local input files to command
        blender_command.extend(local_input_files)
        
        # Add working directory
        blender_command.extend(["-d", working_dir])
        
        # Add additional parameters
        blender_command.extend([f"--product_type", json.dumps(request.product_type)])

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
        # Clean up the downloaded input files and local output files
        try:
            # Clean up input files
            for local_file in local_input_files:
                if os.path.exists(local_file):
                    os.remove(local_file)
                    logger.info(f"Cleaned up input file: {local_file}")
            
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
  
