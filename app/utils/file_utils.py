import os
import logging
from typing import List, Union, Optional
from app.services.blender_service import OutputFile

logger = logging.getLogger(__name__)

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

async def cleanup_processing_files(
    input_files: Union[str, List[str]], 
    output_files: Optional[List[OutputFile]] = None,
    working_dir: Optional[str] = None
):
    """
    Clean up all files related to a processing request.
    
    Args:
        input_files: Single input file path or list of input file paths
        output_files: List of OutputFile objects
        working_dir: Working directory containing temporary files
    """
    try:
        # Convert single input file to list for consistent handling
        if isinstance(input_files, str):
            input_files = [input_files]
        
        # Clean up input files
        for input_file in input_files:
            if os.path.exists(input_file):
                os.remove(input_file)
                logger.info(f"Cleaned up input file: {input_file}")
        
        # Clean up output files if provided
        if output_files:
            for output_file in output_files:
                if os.path.exists(output_file.local_path):
                    os.remove(output_file.local_path)
                    logger.info(f"Cleaned up output file: {output_file.local_path}")
        
        # Clean up any other temporary files in the working directory
        if working_dir:
            input_dir = os.path.join(working_dir, 'input')
            if os.path.exists(input_dir) and os.path.isdir(input_dir):
                for filename in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up temporary file: {file_path}")
                        
            # Check for any other temp files in the working directory with common prefixes
            if os.path.exists(working_dir) and os.path.isdir(working_dir):
                for filename in os.listdir(working_dir):
                    if filename.startswith(('input_', 'output_', 'temp_')):
                        file_path = os.path.join(working_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")
