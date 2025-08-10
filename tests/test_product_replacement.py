from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_replace_product_success():
    response = client.post("/replaceProduct", json={
        "product_sku_id": "12345",
        "glb_image_key": "path/to/glb_image.glb",
        "generated_2d_image_key": "path/to/generated_2d_image.png",
        "all_masks_key": "path/to/all_masks.png",
        "target_product_mask_key": "path/to/target_mask.png",
        "target_product_image_key": "path/to/target_image.png",
        "camera_info": {"field_of_view": 90},
        "lighting_info": {"intensity": 1.0},
        "replace_product_data": {"new_product_id": "67890"}
    })
    assert response.status_code == 202
    assert response.json() == {"message": "Processing started"}

def test_replace_product_failure():
    response = client.post("/replaceProduct", json={})
    assert response.status_code == 422  # Unprocessable Entity for missing required fields

def test_replace_product_invalid_data():
    response = client.post("/replaceProduct", json={
        "product_sku_id": "12345",
        "glb_image_key": "path/to/glb_image.glb",
        "generated_2d_image_key": "path/to/generated_2d_image.png",
        "all_masks_key": "path/to/all_masks.png",
        "target_product_mask_key": "path/to/target_mask.png",
        "target_product_image_key": "path/to/target_image.png",
        "camera_info": {"field_of_view": 90},
        "lighting_info": {"intensity": 1.0},
        "replace_product_data": "invalid_data"  # Invalid data type
    })
    assert response.status_code == 422  # Unprocessable Entity for invalid data type