import os
import io
import boto3
import json
import csv
import botocore
import sys
import uuid
from urllib.parse import unquote_plus

# model endpoint and result file location are stored in environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
RESULT_BUCKET = os.environ['RESULT_BUCKET']
RESULT_FILE_KEY = os.environ['RESULT_FILE_KEY']

runtime= boto3.client('runtime.sagemaker')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    # for the purposes of proving the idea, the classifications 
    # are logged in a file on S3 itelf.  This is not efficient 
    # with respect to S3 because every time a bird is detected I 
    # am doing a pull and put to read and update, but it is convenient
    # for seeing the results and for a few hundred images it is acceptable
    
    # download the results file from S3 or create it if it can't be downloaded.  
    # The bucket name is defined by the trigger that calls this (the trigger 
    # is on a specific bucket)
    result_filename = '/tmp/bird-classification-results.txt'
    try:
        s3_client.download_file(RESULT_BUCKET, RESULT_FILE_KEY, result_filename)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("Result file not found in result bucket.")
        else:
            raise    
    
    result_file = open(result_filename, 'at') 
        
    # for every file reported by the event, download the file, pass
    # the contents to the model, and append the results to the log file
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        print(bucket, key)
        
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
        s3_client.download_file(bucket, key, download_path)
        
        # open the downloaded file and pass contents to model
        with open(download_path, 'rb') as f:
            payload = f.read()
            payload = bytearray(payload)
    
        print(payload)
        
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                            ContentType='application/x-image',
                                            Body = payload)
        
        result = json.loads(response['Body'].read().decode())
        print(result)
        
        w = csv.writer(result_file)
        w.writerow(result)
    
    result_file.close()
    s3_client.upload_file(result_filename, RESULT_BUCKET, 
                                            'bird-classification-results.txt')