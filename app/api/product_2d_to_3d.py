from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import os
from threading import Thread
from app.services.s3_service import S3Service
from app.services.sqs_service import SQSService
from app.services.blender_service import run_blender_script


S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'nestingale-dev-digital-assets')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')
QUEUE_URL = os.getenv('SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/311504593279/EmailMarketing')

s3_service = S3Service(S3_BUCKET_NAME, S3_REGION)
sqs_service = SQSService(QUEUE_URL)
router = APIRouter()


class Product2DTo3DRequest(BaseModel):
    product_type: str
    product_image_s3_path: str
    product_image_s3_path2: str = None
    product_sku_id: str
    output_s3_file_key: str


@router.post('/processGlb')
async def process_glb(request: Product2DTo3DRequest):
    try:
        print(f"JSON data, product_type: {request.product_type}, product_sku_id: {request.product_sku_id}, product_image_s3_path: {request.product_image_s3_path}, product_image_s3_path2: {request.product_image_s3_path2}")
        process_thread = Thread(target=process_request, args=(
            request.product_type,
            request.product_sku_id,
            request.output_s3_file_key,
            request.product_image_s3_path,
            request.product_image_s3_path2
        ))
        process_thread.start()
        return JSONResponse(content={"message": "Processing started"}, status_code=202)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

def process_request(product_type, product_sku_id, output_s3_file_key, product_image_s3_path, product_image_s3_path2):
    try:
        input_file_path = f'../scripts/product_2d_to_3d/input_image_{product_sku_id}.png'
        blender_script_path = '../scripts/product_2d_to_3d/create_Rug_or_Pillow_GLB_public.py'

        # Download input image from S3 using service
        s3_service.download_file(product_image_s3_path, input_file_path)
        print(f"File {product_image_s3_path} downloaded to {input_file_path}")

        if product_type == "pillow":
            input_file_path2 = f'../scripts/product_2d_to_3d/input_image2_{product_sku_id}.png'
            s3_service.download_file(product_image_s3_path2, input_file_path2)
            print(f"File {product_image_s3_path2} downloaded to {input_file_path2}")

        # Run Blender Python script using service
        if product_type == "rug":
            run_blender_script(
                blender_script_path,
                input_file_path,
                None,
                product_type=product_type
            )
        elif product_type == "pillow":
            run_blender_script(
                blender_script_path,
                input_file_path,
                None,
                product_type=product_type,
                input_file_path2=input_file_path2
            )
        else:
            print(f"Unsupported product type: {product_type}")

        print(f"Blender processing complete for {input_file_path}")

        output_file_path = f'../scripts/product_2d_to_3d/output.glb'
        # Upload output file back to S3 using service
        s3_service.upload_file(output_file_path, output_s3_file_key)
        print(f"File {output_file_path} uploaded to {output_s3_file_key}")

        # Clean up local files after processing
        os.remove(input_file_path)
        os.remove(output_file_path)

        message_body = {
            "eventType": "twodToThreedFileCreated",
            "projectId": product_sku_id,
        }
        sqs_service.send_message(json.dumps(message_body))
        print("Message sent to SQS")

    except Exception as e:
        print(f"Error processing request for SkuId :  {product_sku_id}: {str(e)}")
        message_body = {
            "eventType": "twodToThreedFileFailed",
            "projectId": product_sku_id,
        }
        sqs_service.send_message(json.dumps(message_body))
        print("Message sent to SQS")
  
