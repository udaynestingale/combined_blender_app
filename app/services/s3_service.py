from botocore.exceptions import NoCredentialsError, ClientError
import boto3
import os

class S3Service:
    def __init__(self, bucket_name, region_name='us-east-1'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region_name)

    def upload_file(self, file_name, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_name)
        try:
            self.s3_client.upload_file(file_name, self.bucket_name, object_name)
            return True
        except FileNotFoundError:
            print(f"The file {file_name} was not found.")
            return False
        except NoCredentialsError:
            print("Credentials not available.")
            return False
        except ClientError as e:
            print(f"Failed to upload {file_name} to {self.bucket_name}/{object_name}: {e}")
            return False

    def download_file(self, object_name, file_name):
        try:
            self.s3_client.download_file(self.bucket_name, object_name, file_name)
            return True
        except ClientError as e:
            print(f"Failed to download {object_name} from {self.bucket_name}: {e}")
            return False

    def list_files(self, prefix=''):
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            print(f"Failed to list files in {self.bucket_name}: {e}")
            return []