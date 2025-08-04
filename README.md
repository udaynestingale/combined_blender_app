# Photo Product API

## Overview
The Photo Product API is a FastAPI application that provides endpoints for generating photo-realistic views, processing 2D images to 3D models, and replacing products in images. This application leverages AWS services such as S3 and SQS for file management and messaging.

## Features
- Generate photo-realistic views from 3D models.
- Convert 2D product images into 3D models.
- Replace products in existing images with new ones.
- Asynchronous processing using background tasks.

## Project Structure
```
photo_product_api
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── photo_realistic_view.py
│   │   ├── product_2d_to_3d.py
│   │   └── product_replacement.py
│   ├── services
│   │   ├── __init__.py
│   │   ├── s3_service.py
│   │   ├── sqs_service.py
│   │   └── blender_service.py
│   └── models
│       ├── __init__.py
│       ├── photo_realistic_view.py
│       ├── product_2d_to_3d.py
│       └── product_replacement.py
├── tests
│   ├── __init__.py
│   ├── test_photo_realistic_view.py
│   ├── test_product_2d_to_3d.py
│   └── test_product_replacement.py
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Docker (for containerization)
- AWS account with S3 and SQS configured

### Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd photo_product_api
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in the required values.

### Running the Application
To run the application locally, use:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment
To build and run the application using Docker:
1. Build the Docker image:
   ```
   docker build -t photo_product_api .
   ```

2. Run the Docker container:
   ```
   docker run -d -p 8000:8000 photo_product_api
   ```

### API Endpoints
- **Health Check**: `GET /healthCheck`
- **Generate Photo Realistic View**: `POST /generatePhotoRealisticView`
- **Process 2D to 3D**: `POST /processGlb`
- **Replace Product**: `POST /replaceProduct`

## Testing
To run the tests, use:
```
pytest tests/
```

## License
This project is licensed under the MIT License. See the LICENSE file for details."# combined_blender_app" 
