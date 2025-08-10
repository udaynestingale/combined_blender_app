import os
import json
import asyncio
import subprocess
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

from app.core.config import get_settings
from app.services.s3_service import S3Service, S3ServiceError

settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize S3 service
s3_service = S3Service(bucket_name=settings.S3_BUCKET_NAME, region_name=settings.AWS_REGION)

@dataclass
class OutputFile:
    local_path: str
    s3_key: str
    file_type: str

@dataclass
class BlenderConfig:
    script_path: str
    input_files: List[str]
    output_files: List[OutputFile]
    working_dir: str
    additional_params: Dict[str, Any]

class BlenderError(Exception):
    """Custom exception for Blender-related errors"""
    pass

async def run_blender_script(config: BlenderConfig) -> List[OutputFile]:
    """
    Execute a Blender script with the given parameters and handle file uploads.
    Returns a list of processed output files.
    """
    try:
        # Download input files if they are S3 keys
        local_input_files = []
        for input_file in config.input_files:
            if input_file.startswith('s3://') or not os.path.exists(input_file):
                local_path = os.path.join(config.working_dir, 'input', os.path.basename(input_file))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                try:
                    await s3_service.download_file_async(input_file, local_path)
                    local_input_files.append(local_path)
                except S3ServiceError as e:
                    raise BlenderError(f"Failed to download input file {input_file}: {str(e)}")
            else:
                local_input_files.append(input_file)

        # Ensure output directory exists
        os.makedirs(config.working_dir, exist_ok=True)

        # Construct Blender command as a list of arguments
        blender_path = settings.BLENDER_PATH if hasattr(settings, 'BLENDER_PATH') else 'blender'
        command = [
            blender_path,
            "--background",
            "--python", config.script_path,
            "--",  # Argument separator
        ]
        
        # Add input files
        command.extend(local_input_files)
        
        # Add working directory
        command.extend(["-d", config.working_dir])
        
        # Add additional parameters
        for key, value in config.additional_params.items():
            command.extend([f"--{key}", json.dumps(value)])

        logger.info(f"Running Blender command: {' '.join(command)}")

        # Execute Blender command with retry logic
        for attempt in range(3):
            try:
                # Set a timeout that is slightly less than typical cloud service timeouts (e.g., 60 minutes)
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3500  # 58.3 minutes in seconds
                )
                
                logger.info(f"Blender processing complete")
                break
            
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else f"Process failed with code {e.returncode}"
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise BlenderError(f"All retry attempts failed: {error_msg}")
            
            except subprocess.TimeoutExpired as e:
                error_msg = f"Blender process timed out after {e.timeout} seconds"
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise BlenderError(f"All retry attempts failed: {error_msg}")
            
            except Exception as e:
                if attempt == 2:
                    raise BlenderError(f"Failed to execute Blender: {str(e)}")
                await asyncio.sleep(2 ** attempt)

        # Upload output files to S3
        processed_files = []
        for output_file in config.output_files:
            if os.path.exists(output_file.local_path):
                try:
                    await s3_service.upload_file_async(
                        output_file.local_path, 
                        output_file.s3_key
                    )
                    processed_files.append(output_file)
                    logger.info(f"Uploaded {output_file.local_path} to {output_file.s3_key}")
                except S3ServiceError as e:
                    raise BlenderError(f"Failed to upload output file {output_file.local_path}: {str(e)}")
            else:
                logger.warning(f"Output file {output_file.local_path} does not exist. Skipping upload.")

        return processed_files

    finally:
        # Cleanup temporary files
        try:
            # Only clean up downloaded input files
            for input_file in local_input_files:
                if input_file.startswith(config.working_dir) and os.path.exists(input_file):
                    logger.debug(f"Cleaning up temporary input file: {input_file}")
                    os.remove(input_file)
            
            # Clean up output files if they were successfully uploaded to S3
            for output_file in config.output_files:
                if os.path.exists(output_file.local_path):
                    logger.debug(f"Cleaning up temporary output file: {output_file.local_path}")
                    os.remove(output_file.local_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

async def process_blender_request_async(
    script_path: str,
    input_files: List[str],
    output_files: List[OutputFile],
    working_dir: str,
    **kwargs: Dict[str, Any]
) -> List[OutputFile]:
    """
    Process a Blender request, handling file downloads, processing, and uploads.
    Returns a list of processed output files with their S3 locations.
    """
    try:
        # Validate parameters
        if not script_path:
            raise ValueError("Script path is required")
        if not os.path.exists(script_path):
            raise ValueError(f"Script file not found: {script_path}")
        if not input_files:
            raise ValueError("At least one input file is required")
        if not output_files:
            raise ValueError("At least one output file configuration is required")
        if not working_dir:
            raise ValueError("Working directory is required")

        # Create working directory if it doesn't exist
        os.makedirs(working_dir, exist_ok=True)

        # Create Blender configuration
        config = BlenderConfig(
            script_path=script_path,
            input_files=input_files,
            output_files=output_files,
            working_dir=working_dir,
            additional_params=kwargs
        )

        # Process the request and handle files
        return await run_blender_script(config)

    except ValueError as e:
        logger.error(f"Invalid parameters for Blender request: {str(e)}")
        raise BlenderError(str(e))
    except Exception as e:
        logger.error(f"Error processing Blender request: {str(e)}", exc_info=True)
        raise BlenderError(str(e))