from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
from threading import Thread
from app.services.s3_service import S3Service
from app.services.sqs_service import SQSService
from app.services.blender_service import run_blender_script

router = APIRouter()

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'nestingale-dev-digital-assets')
S3_REGION = os.getenv('S3_REGION', 'us-east-1')
QUEUE_URL = os.getenv('SQS_QUEUE_URL', 'https://sqs.us-east-1.amazonaws.com/311504593279/EmailMarketing')

s3_service = S3Service(S3_BUCKET_NAME, S3_REGION)
sqs_service = SQSService(QUEUE_URL)

class PhotoRealisticViewRequest(BaseModel):
    template_id: str
    glb_image_key: str
    generated_2d_image_key: str
    all_masks_key: str
    camera_info: dict
    lighting_info: dict

@router.post("/generatePhotoRealisticView")
async def generate_photo_realistic_view(request: PhotoRealisticViewRequest):
    try:
        json_data = request.dict()
        template_id = json_data['template_id']
        glb_image_key = json_data['glb_image_key']
        generated_2d_image_key = json_data['generated_2d_image_key']
        all_masks_key = json_data['all_masks_key']
        camera_info = json.dumps(json_data['camera_info'])
        lighting_info = json.dumps(json_data['lighting_info'])

        print(f"Processing request for template_id: {template_id}")

        process_thread = Thread(target=process_request, args=(template_id, glb_image_key, generated_2d_image_key, all_masks_key, camera_info, lighting_info))
        process_thread.start()

        return {"message": "Processing started"}, 202

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_request(template_id, glb_image_key, generated_2d_image_key, all_masks_key, camera_info, lighting_info):
    try:
        input_file_path = '../scripts/photo_realistic_view/input_image.glb'
        blender_script_path = '../scripts/photo_realistic_view/blender_script.py'
        output_dir = '../scripts/photo_realistic_view/generated_files'
        generated_2d_image_local_path = f'{output_dir}/room_render.png'
        all_masks_local_path = f'{output_dir}/mask_all_products.png'

        # Download input image from S3 using service
        s3_service.download_file(glb_image_key, input_file_path)
        print(f"File {glb_image_key} downloaded to {input_file_path}")

        # Run Blender script using service
        run_blender_script(
            blender_script_path,
            input_file_path,
            output_dir,
            camera_json=json.loads(camera_info),
            lighting_json=json.loads(lighting_info),
            generate_mask=True,
            combined_mask_only=True,
            r=1920
        )
        print("2d image and masks are generated")

        # Upload results to S3 using service
        s3_service.upload_file(generated_2d_image_local_path, generated_2d_image_key)
        s3_service.upload_file(all_masks_local_path, all_masks_key)
        print("2d image and masks upload completed")

        # Clean up local files
        os.remove(generated_2d_image_local_path)
        os.remove(all_masks_local_path)

        message_body = {
            "eventType": "photoRealisticViewGenerated",
            "templateId": template_id,
        }
        sqs_service.send_message(json.dumps(message_body))
        print("Message sent to SQS")

    except Exception as e:
        print(f"Error processing request {glb_image_key}: {str(e)}")
        message_body = {
            "eventType": "photoRealisticViewFailed",
            "templateId": template_id,
        }
        sqs_service.send_message(json.dumps(message_body))
        print("Failure message sent to SQS")
  
