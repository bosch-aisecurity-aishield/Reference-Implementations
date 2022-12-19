**Reference implementation of AIShield with MLflow on MNIST dataset.**
**MNIST DATA: ** It is a large database of handwritten digits commonly used for computer vision (CV) tasks.

**This document provides an overview of the use of the AIShield API with MLflow and logs artifact received from API.**

**To install packages**

 pip install -r requirements.txt

**Load Dataset** 

Download mnist dataset from tensorflow, preprocess it, and split the data set to train, test, and validation in 54k : 6k : 10K.
Visualize sample data. 

**Model Training**

Create model architecture and train it on training data. After training, 
validate on the validation dataset. Model accuracy expected ~ 99.3%

**Prepare artifact for AIShield API call**

Save a sample of data, its label, and model as a .zip file. The .zip file is needed during AIShield API call

**AIShield API call**

Requirement: Get AIShield API endpoint, x-API-key and ord_id from AIShield team.
1. Call model registration API. This API will return a unique model id and 
path to upload data, model, and label. It will also return a sample request payload for model analysis API. 
2. Upload the data, model, and label to the given url.
3. After successful upload, call model analysis API and pass the payload. In the response, you will get unique job_id and monitor link. Monitor link can be used to track the progress of the triggered job.
4. Monitor the progress of job id using get API, and after successful completion, download the artifacts and log it to MLflow as an artifact. The artifacts will contain vulnerability and defense reports, attack samples, defense artifact
5. Load AIShield provided defense model and pass sample of original data and attack data to get a prediction from the defense model. 
6. To integrate with SIEM solutions (Microsoft Sentinel : https://azuremarketplace.microsoft.com/en-us/marketplace/apps/rbei.bgsw_aishield_sentinel?tab=Overview and  Splunk : https://splunkbase.splunk.com/app/6506/) please follow the following instruction 
   
6.1. For the Microsoft Sentinel connector, please provide azure_log_customer_id and azure_log_shared_key while creating the AISDefenseModel object 

6.2. For Splunk connector, please provide splunk_url and splunk authorization while creating AISDefenseModel object















