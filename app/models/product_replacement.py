from pydantic import BaseModel
from typing import List, Optional

class CameraInfo(BaseModel):
    position: List[float]
    rotation: List[float]

class LightingInfo(BaseModel):
    intensity: float
    color: List[float]

class ReplaceProductData(BaseModel):
    product_id: str
    new_product_id: str
    mask_key: str

class ProductReplacementRequest(BaseModel):
    product_sku_id: str
    glb_image_key: str
    generated_2d_image_key: str
    all_masks_key: str
    target_product_mask_key: str
    target_product_image_key: str
    camera_info: CameraInfo
    lighting_info: LightingInfo
    replace_product_data: ReplaceProductData

class ProductReplacementResponse(BaseModel):
    message: str
    product_sku_id: str
    status: str