# Nestingale Blender API

This application provides a FastAPI service for processing various Blender-based operations including photo-realistic view, 2D to 3D conversion, product replacement, and USDZ to GLB conversion.

## Requirements

- Python 3.10+
- Blender (for running without Docker)
- Docker (optional, for containerized execution)

## Running without Docker

### Setting up the environment

1. Clone the repository:

   ```
   git clone https://github.com/udaynestingale/combined_blender_app.git
   cd combined_blender_app
   ```

2. Create a virtual environment and install dependencies:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your AWS and Blender configurations:

   ```
   S3_BUCKET_NAME=your_s3_bucket_name
   AWS_REGION=your_aws_region
   SQS_QUEUE_URL=your_sqs_queue_url
   AWS_ACCESS_KEY_ID=your_aws_access_key_id
   AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
   BLENDER_PATH=path_to_blender_executable
   ```

4. Start the application:

   ```
   uvicorn app.main:app --reload
   ```

5. Open your browser and navigate to `http://localhost:8000/docs` to see the API documentation.

## Running with Docker

### Build and run using Docker

1. Build the Docker image:

   ```
   docker build -t nestingale-blender-app .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 \
     -e AWS_ACCESS_KEY_ID=your_aws_access_key \
     -e AWS_SECRET_ACCESS_KEY=your_aws_secret_key \
     -e AWS_REGION=your_aws_region \
     -e S3_BUCKET_NAME=your_s3_bucket \
     -e SQS_QUEUE_URL=your_sqs_url \
     -e BLENDER_PATH=blender \
     -e LOG_LEVEL=INFO \
     -v "./app:/app/app" \
     -v "./tests:/app/tests" \
     -v "./blender_working_data:/app/blender_working_data" \
     nestingale-blender-app
   ```

### Using the convenience scripts

We provide convenience scripts for building and running the Docker container:

- On Linux/macOS:

  ```
  chmod +x run_docker.sh
  ./run_docker.sh
  ```

- On Windows:
  ```
  .\run_docker.ps1
  ```

Remember to edit these scripts to provide your actual AWS credentials and other configuration values.

## API Endpoints

The application provides the following endpoints:

- `/api/v1/photo-realistic-view` - For photo-realistic view operations
- `/api/v1/product-2d-to-3d` - For converting 2D product images to 3D models
- `/api/v1/product-replacement` - For product replacement operations
- `/api/v1/usdz-to-glb` - For converting USDZ files to GLB format

For detailed API documentation, visit `/docs` when the application is running.

## Testing

Run the tests with pytest:

```
pytest
```

## License

[Include license information here]
