from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_process_2d_to_3d():
    response = client.post("/processGlb", json={
        "product_type": "rug",
        "product_image_s3_path": "path/to/image1.png",
        "product_image_s3_path2": "path/to/image2.png",
        "product_sku_id": "sku123",
        "output_s3_file_key": "output/path/output.glb"
    })
    assert response.status_code == 202
    assert response.json() == {"message": "Processing started"}

def test_process_2d_to_3d_invalid_type():
    response = client.post("/processGlb", json={
        "product_type": "invalid_type",
        "product_image_s3_path": "path/to/image1.png",
        "product_image_s3_path2": "path/to/image2.png",
        "product_sku_id": "sku123",
        "output_s3_file_key": "output/path/output.glb"
    })
    assert response.status_code == 202  # Processing should still start
    # Additional checks can be added to verify the handling of unsupported types

def test_process_2d_to_3d_missing_fields():
    response = client.post("/processGlb", json={
        "product_type": "rug",
        "product_sku_id": "sku123"
    })
    assert response.status_code == 422  # Unprocessable Entity due to missing fields

def test_health_check():
    response = client.get("/healthCheck")
    assert response.status_code == 200
    assert response.json() == {"message": "Success"}