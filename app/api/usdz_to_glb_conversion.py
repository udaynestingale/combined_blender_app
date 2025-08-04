
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
from threading import Thread
from app.services.s3_service import S3Service
from app.services.sqs_service import SQSService

router = APIRouter()

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'nestingale-dev-product-3d-assets')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')
QUEUE_URL = os.getenv('SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/311504593279/EmailMarketing')

s3_service = S3Service(S3_BUCKET_NAME, S3_REGION)
sqs_service = SQSService(QUEUE_URL)

class UsdzToGlbRequest(BaseModel):
    input_file_key: str
    output_file_key: str

@router.get('/healthCheck')
async def health_check():
    return {"message": "Success"}

@router.post('/convertUsdzToGlb')
async def convert_usdz_to_glb(request: UsdzToGlbRequest):
    try:
        print(f"Received request: input_file_key={request.input_file_key}, output_file_key={request.output_file_key}")
        process_thread = Thread(target=process_request, args=(request.input_file_key, request.output_file_key))
        process_thread.start()
        return {"message": "Processing started"}, 202
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_request(input_file_key, output_file_key):
    try:
        input_file_path = '../scripts/usdz_to_glb_conversion/input_image.usdz'
        blender_script_path = '../scripts/usdz_to_glb_conversion/blender_with_furniture.py'
        output_file_path = '../scripts/usdz_to_glb_conversion/output_image.glb'

        # Download input file from S3
        s3_service.download_file(input_file_key, input_file_path)
        print(f"File {input_file_key} downloaded to {input_file_path}")

        # Run Blender Python script
        from app.services.blender_service import run_blender_script
        run_blender_script(
            blender_script_path,
            input_file_path,
            None,
            output_file_path=output_file_path
        )
        print("File conversion completed")

        # Upload output file to S3
        s3_service.upload_file(output_file_path, output_file_key)
        print(f"File {output_file_path} uploaded to {output_file_key}")

        # Clean up local files
        os.remove(input_file_path)
        os.remove(output_file_path)

        message_body = {
            "eventType": "usdzToGlbConversionCompleted",
            "inputFileKey": input_file_key,
            "outputFileKey": output_file_key,
        }
        sqs_service.send_message(json.dumps(message_body))
        print("Message sent to SQS")

    except Exception as e:
        print(f"Error processing request for input_file_key {input_file_key}: {str(e)}")
        message_body = {
            "eventType": "usdzToGlbConversionFailed",
            "inputFileKey": input_file_key,
            "outputFileKey": output_file_key,
        }
        sqs_service.send_message(json.dumps(message_body))
        print("Failure message sent to SQS")