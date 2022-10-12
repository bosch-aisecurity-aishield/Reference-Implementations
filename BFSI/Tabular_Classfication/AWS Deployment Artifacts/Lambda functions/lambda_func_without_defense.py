import os
import io
import boto3
import json
import csv
import ast
from datetime import datetime
import base64
import hmac
import hashlib
import requests

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
dynamodb  = boto3.client('dynamodb')


def lambda_handler(event, context):
    data = json.loads(json.dumps(event['body']))
    payload = str(data['data'])
    header = json.loads(json.dumps(event['headers']))
    token = header['jwt_token']
    # for visualization purpose(in SIEM-Sentinel connectors). In actual scenario, actual host details would be captured
    if token == 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTIzNDU2Nzg5LCJuYW1lIjoiSm9zZXBoIn0.OpOSSw7e485LOP5PrzScxHb7SR6sAOMRckfFwi4rp2q':
        hostame = 'host1'
        IPAddr = '174.46.230.219'
    else:
        hostame = 'host2'
        IPAddr = '124.116.138.11'

    orig_mdl_response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='text/csv',
                                      Body=payload)

    original_model_out = orig_mdl_response['Body'].read().decode("utf-8")
    original_model_out = int(ast.literal_eval(original_model_out)[0])

    return {
        'result': {
            "original_model_output": original_model_out,
            }
    }
