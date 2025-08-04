from botocore.exceptions import ClientError
import boto3
import os

class SQSService:
    def __init__(self, queue_url):
        self.sqs_client = boto3.client('sqs')
        self.queue_url = queue_url

    def send_message(self, message_body):
        try:
            response = self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=message_body
            )
            return response
        except ClientError as e:
            print(f"Error sending message to SQS: {e}")
            return None

    def receive_messages(self, max_messages=10, wait_time_seconds=0):
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_seconds
            )
            return response.get('Messages', [])
        except ClientError as e:
            print(f"Error receiving messages from SQS: {e}")
            return []

    def delete_message(self, receipt_handle):
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
        except ClientError as e:
            print(f"Error deleting message from SQS: {e}")