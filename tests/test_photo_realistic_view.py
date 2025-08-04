from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/healthCheck")
    assert response.status_code == 200
    assert response.json() == {"message": "Success"}

def test_generate_photo_realistic_view():
    response = client.post("/generatePhotoRealisticView", json={
        "template_id": "test_template",
        "glb_image_key": "test_glb_image_key",
        "generated_2d_image_key": "test_generated_2d_image_key",
        "all_masks_key": "test_all_masks_key",
        "camera_info": {"field_of_view": 90},
        "lighting_info": {"intensity": 1.0}
    })
    assert response.status_code == 202
    assert response.json() == {"message": "Processing started"}

def test_invalid_request():
    response = client.post("/generatePhotoRealisticView", json={})
    assert response.status_code == 500
    assert "error" in response.json()