from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import boto3
import os
from threading import Thread

router = APIRouter()

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'nestingale-dev-digital-assets')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')

s3_client = boto3.client('s3', region_name=S3_REGION)
sqs = boto3.client('sqs', region_name=S3_REGION)
queue_url = os.getenv('SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/311504593279/EmailMarketing')

class ProductReplacementRequest(BaseModel):
    product_sku_id: str
    glb_image_key: str
    generated_2d_image_key: str
    all_masks_key: str
    target_product_mask_key: str
    camera_info: dict
    lighting_info: dict
    replace_product_data: dict

@router.post("/replaceProduct")
async def replace_product(request: ProductReplacementRequest):
    try:
        json_data = request.dict()
        print("Received request to replace product:", json_data)

        # Run the process in a new thread to handle parallel requests
        process_thread = Thread(target=process_request, args=(json_data,))
        process_thread.start()

        return {"message": "Processing started"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_request(data):
    try:
        input_file_path = f'../scripts/product_replacement/input_image.glb'
        blender_script_path = f'../scripts/product_replacement/blender_script_camera_public.py'
        output_dir = '../scripts/product_replacement/generated_files'
        generated_2d_image_local_path = f'{output_dir}/room_render.png'
        all_masks_local_path = f'{output_dir}/mask_all_products.png'
        target_product_mask_local_path = f'{output_dir}/mask_{data["product_sku_id"]}.png'
        
        camera_info = json.dumps(data['camera_info'])
        lighting_info = json.dumps(data['lighting_info'])
        replace_product_data = json.dumps(data['replace_product_data'])
        
        # Download input image from S3
        s3_client.download_file(S3_BUCKET_NAME, data['glb_image_key'], input_file_path)
        print(f"File {data['glb_image_key']} downloaded to {input_file_path}")

        from app.services.blender_service import run_blender_script
        run_blender_script(
            blender_script_path,
            input_file_path,
            output_dir,
            generate_mask=True,
            camera_json=json.loads(camera_info),
            lighting_json=json.loads(lighting_info),
            use_environment_map="studio.exr",
            use_existing_camera=True,
            replace_product=json.loads(replace_product_data)
        )
        print("2D image and masks are generated")

        s3_client.upload_file(generated_2d_image_local_path, S3_BUCKET_NAME, data['generated_2d_image_key'])
        s3_client.upload_file(all_masks_local_path, S3_BUCKET_NAME, data['all_masks_key'])
        s3_client.upload_file(target_product_mask_local_path, S3_BUCKET_NAME, data['target_product_mask_key'])
        print("Files uploaded to S3")

        # Clean up local files after processing
        os.remove(generated_2d_image_local_path)
        os.remove(all_masks_local_path)
        os.remove(target_product_mask_local_path)

        message_body = {
            "eventType": "ProductReplaced",
            "productSkuId": data['product_sku_id'],
        }

        # Send message to SQS queue
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message_body))
        print("Message sent to SQS")

    except Exception as e:
        print(f"Error processing request for SKU ID {data['product_sku_id']}: {str(e)}")
        message_body = {
            "eventType": "ProductReplacementFailed",
            "productSkuId": data['product_sku_id'],
        }
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message_body))
        print("Failure message sent to SQS")