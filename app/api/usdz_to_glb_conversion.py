
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import json
import os
import logging
from typing import List

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


@router.post('/convert', status_code=status.HTTP_200_OK)
@track_time(BLENDER_PROCESSING_TIME, {"task_type": "usdz_to_glb"})
async def convert_usdz_to_glb(request: UsdzToGlbRequest):
    """
    Convert a USDZ file to GLB format.
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

        # Construct Blender command
        blender_path = settings.BLENDER_PATH if hasattr(settings, 'BLENDER_PATH') else 'blender'
        blender_command = [
            blender_path,
            "--background",
            "--python", blender_script_path,
            "--",  # Argument separator
            input_file_path,  # Input file
            "-d", work_dir,  # Working directory
            f"--output_file_path", json.dumps(output_file_path)
        ]
        
        # Configure output files
        output_files = [
            OutputFile(
                local_path=output_file_path,
                s3_key=request.output_file_key,
                file_type='glb'
            )
        ]
        
        # Process request
        processed_files = await process_blender_request_async(
            working_dir=work_dir,
            blender_command=blender_command,
            output_files=output_files
        )

        return {
            "status": "completed",
            "input_file": request.input_file_key,
            "output_file": request.output_file_key,
            "files": [
                {
                    "type": file.file_type,
                    "s3_key": file.s3_key
                }
                for file in processed_files
            ]
        }

    except S3ServiceError as e:
        logger.error(f"S3 error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"S3 service error: {str(e)}"
        )
    except BlenderError as e:
        logger.error(f"Blender error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Conversion request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        # Clean up the downloaded input file and local output files
        try:
            # Clean up input and output files
            for file_path in [input_file_path, output_file_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up {file_path}")
            
            # Clean up any other temporary files in the working directory
            if os.path.exists(work_dir) and os.path.isdir(work_dir):
                for filename in os.listdir(work_dir):
                    if filename.startswith('input_') or filename.startswith('output_'):
                        file_path = os.path.join(work_dir, filename)
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