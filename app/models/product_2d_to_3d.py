from pydantic import BaseModel
from typing import List, Optional

class CameraInfo(BaseModel):
    position: List[float]
    rotation: List[float]

class LightingInfo(BaseModel):
    intensity: float
    color: List[float]

class Product2DTo3DRequest(BaseModel):
    product_type: str
    product_image_s3_path: str
    product_image_s3_path2: Optional[str] = None
    product_sku_id: str
    output_s3_file_key: str

class Product2DTo3DResponse(BaseModel):
    message: str
    product_sku_id: str
    output_s3_file_key: str

class ReplaceProductRequest(BaseModel):
    product_sku_id: str
    glb_image_key: str
    generated_2d_image_key: str
    all_masks_key: str
    target_product_mask_key: str
    camera_info: CameraInfo
    lighting_info: LightingInfo
    replace_product_data: dict

class ReplaceProductResponse(BaseModel):
    message: str
    product_sku_id: str
    target_product_mask_key: str