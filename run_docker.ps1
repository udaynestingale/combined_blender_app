# Build the Docker image
docker build -t nestingale-blender-app .

# Run the Docker container
docker run -p 8000:8000 `
  -e AWS_ACCESS_KEY_ID=your_aws_access_key `
  -e AWS_SECRET_ACCESS_KEY=your_aws_secret_key `
  -e AWS_REGION=your_aws_region `
  -e S3_BUCKET_NAME=your_s3_bucket `
  -e SQS_QUEUE_URL=your_sqs_url `
  -e BLENDER_PATH=blender `
  -e LOG_LEVEL=INFO `
  -v "${PWD}/app:/app/app" `
  -v "${PWD}/tests:/app/tests" `
  -v "${PWD}/blender_working_data:/app/blender_working_data" `
  nestingale-blender-app
