#!/bin/bash

# --- CONFIGURATION ---
# 1. Replace with the OAuth 2.0 Client ID you used as the "Audience" in the AWS IAM Identity Provider.
GCP_AUDIENCE="101040847901631337904.apps.googleusercontent.com"

# 2. Replace with the ARN of the IAM Role you created in AWS.
AWS_ROLE_ARN="arn:aws:iam::311504593279:role/GCP_blender"

# 3. Set your target AWS region.
export AWS_DEFAULT_REGION="us-east-1" 
# --- END CONFIGURATION ---


# Define the path where the temporary token will be stored
export AWS_WEB_IDENTITY_TOKEN_FILE="/tmp/gcp_token"

# Fetch the OIDC token from the GCP metadata server
# The audience must match exactly what you configured in AWS IAM.
curl "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=${GCP_AUDIENCE}&format=full" \
-H "Metadata-Flavor: Google" -o "${AWS_WEB_IDENTITY_TOKEN_FILE}"

# Set the Role ARN for the AWS SDK to use
export AWS_ROLE_ARN

# --- Now, launch your Python application ---
# The AWS SDK will automatically read the environment variables and get temporary credentials.
echo "Starting Python application..."
# Replace the line below with the actual command to start your app
uvicorn main:app --host 0.0.0.0 --port 8000