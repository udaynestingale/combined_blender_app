from pydantic import BaseModel
from typing import Any, List, Optional

class CameraInfo(BaseModel):
    position: List[float]
    rotation: List[float]
    fov: float

class LightingInfo(BaseModel):
    intensity: float
    color: List[float]

class PhotoRealisticViewRequest(BaseModel):
    template_id: str
    glb_image_key: str
    generated_2d_image_key: str
    all_masks_key: str
    camera_info: CameraInfo
    lighting_info: LightingInfo

class PhotoRealisticViewResponse(BaseModel):
    message: str
    generated_2d_image_key: Optional[str] = None
    all_masks_key: Optional[str] = None
    error: Optional[str] = None