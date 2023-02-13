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
DEF1_ENDPOINT_NAME = os.environ['DEF1_ENDPOINT_NAME']
DEF2_ENDPOINT_NAME = os.environ['DEF2_ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
dynamodb  = boto3.client('dynamodb')

"""
Config for sentinel - provide your own config
"""

azure_log_customer_id = 'xxx-xxx-xxx'
azure_log_shared_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
table_name = 'AIShield'

"""
Code for sentinel 
"""

def build_signature(customer_id:str, shared_key:str, date:str, content_length:int, method:str, content_type:str, resource:str):
        """
        Returns authorization header which will be used when sending data into Azure Log Analytics

        Parameters
        ----------
        customer_id : str
            DESCRIPTION.Workspace ID obtained from Advanced Settings
        shared_key : str
            DESCRIPTION.Authorization header, created using build_signature
        date : str
            DESCRIPTION. datetime
        content_length : int
            DESCRIPTION. lenth of msg
        method : str
            DESCRIPTION. post method
        content_type : str
            DESCRIPTION.
        resource : str
            DESCRIPTION.

        Returns
        -------
        authorization : TYPE
            DESCRIPTION.

        """
        x_headers = 'x-ms-date:' + date
        string_to_hash = method + "\n" + str(content_length) + "\n" + content_type + "\n" + x_headers + "\n" + resource
        bytes_to_hash = bytes(string_to_hash, 'UTF-8')
        decoded_key = base64.b64decode(shared_key)
        encoded_hash = base64.b64encode(hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()).decode('utf-8')
        authorization = "SharedKey {}:{}".format(customer_id,encoded_hash)
        return authorization
    
    
def post_data(customer_id:str, shared_key:str, body:json, log_type:str):
        """
        Sends payload to Azure Log Analytics Workspace
        
        Parameters
        ----------
        customer_id : str
            DESCRIPTION.Workspace ID obtained from Advanced Settings
        shared_key : str
            DESCRIPTION.Authorization header, created using build_signature
        body : json
            DESCRIPTION.payload to send to Azure Log Analytics
        log_type : str
            DESCRIPTION.Azure Log Analytics table name

        Returns
        -------
        None.

        """
        
        method = 'POST'
        content_type = 'application/json'
        resource = '/api/logs'
        rfc1123date = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        content_length = len(body)
        signature = build_signature(customer_id, shared_key, rfc1123date, content_length, method, content_type, resource)

        uri = 'https://' + customer_id + '.ods.opinsights.azure.com' + resource + '?api-version=2016-04-01'

        headers = {
            'content-type': content_type,
            'Authorization': signature,
            'Log-Type': log_type,
            'x-ms-date': rfc1123date
        }

        response = requests.post(uri,data=body, headers=headers)
        if 200 <= response.status_code <= 299:
            print('Accepted payload:' + body)
        else:
            print("Unable to Write: " + format(response.status_code))


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
    
    extraction_def_mdl_response = runtime.invoke_endpoint(EndpointName=DEF1_ENDPOINT_NAME,
                                      ContentType='text/csv',
                                      Body=payload)
    extraction_def_model_prob = float(extraction_def_mdl_response['Body'].read().decode("utf-8"))
    
    evasion_def_mdl_response = runtime.invoke_endpoint(EndpointName=DEF2_ENDPOINT_NAME,
                                      ContentType='text/csv',
                                      Body=payload)
    evasion_def_mdl_response = evasion_def_mdl_response['Body'].read().decode("utf-8")
    evasion_def_model_prob =  float(ast.literal_eval(evasion_def_mdl_response)['predictions'][0][0])
    
    mdl_extr_attack_flg = False
    mdl_evasion_attack_flg = False
    
    if extraction_def_model_prob >= 0.6:
        mdl_extr_attack_flg = True
        
    if evasion_def_model_prob >= 0.6:
        mdl_evasion_attack_flg = True
        

    if mdl_extr_attack_flg or mdl_evasion_attack_flg:
        # update db
        db_item = dynamodb.get_item(TableName='UserRequests', Key={'id':{'S':token}})
        new_attack_cnt = int(db_item['Item']['attack_cnt']['N']) + 1
        dynamodb.put_item(TableName='UserRequests', Item={'id':{'S':token},'attack_cnt':{'N':str(new_attack_cnt)}, 'if_allow':{'S':'True'}})
    
        
        sentinel_paylod = []
        if mdl_extr_attack_flg:
            extraction_payload_json = {
            				"RawMessage": "Tabular Classification AI Model Extraction Attack identified",
            				"service_name": "tabular_classification_extraction_defense_engine",
            				"asset_id" : hostame,
            				"source_name" : IPAddr,
            				"probablity": extraction_def_model_prob,
            				"attack_name": "model_attack",
                            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            				}
            sentinel_paylod.append(extraction_payload_json)
            
        if mdl_evasion_attack_flg:
            evasion_payload_json = {
            				"RawMessage": "Tabular Classification AI Model Evasion Attack identified",
            				"service_name": "tabular_classification_evasion_defense_engine",
            				"asset_id" : hostame,
            				"source_name" : IPAddr,
            				"probablity": evasion_def_model_prob,
            				"attack_name": "model_attack",
                            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            				}
            sentinel_paylod.append(evasion_payload_json)
            
        sentinel_paylod = json.dumps(sentinel_paylod)
        post_data(azure_log_customer_id, azure_log_shared_key, sentinel_paylod, table_name)

    return {
        'result': {
                        "original_model_output": original_model_out,
                        'model_extraction_attack_detected': mdl_extr_attack_flg,
                        'model_evasion_attack_detected':mdl_evasion_attack_flg,
                        'model_extraction_attack_probability': extraction_def_model_prob,
                        'model_evasion_attack_probability':evasion_def_model_prob
                            
                    }
    }
