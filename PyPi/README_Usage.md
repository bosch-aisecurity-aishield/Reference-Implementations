AIShield Package Example Usage
=============
![aishield](https://aisdocs.blob.core.windows.net/images/aishieldLogoPypi.PNG)

-------------
This document provides an overview of the use of the AIShield package. 

### Import SDK as:

    import aishield as ais

### Provide Subscription Details:
    
    API_URL = 'https://xxxxxxxxxxx/AIShieldAPI'
    API_KEY = 'xxxxx'
    ORG_ID =  'xxxxxxxxxxx'

### Provide Input and Output Paths:
    
    DATA_PATH = r'/data/original_data.zip'
    LABEL_PATH = r'/data/original_label.zip'
    MODEL_PATH = r'/model/mnist_model.zip'
    OUTPUT_PATH = r'/output'

### Define the Task and Analysis Type
    
    task_type = ais.get_type("task", "image_classification")
    analysis_type = ais.get_type("analysis", "extraction")

### Initialize AIShield SDK Client

    client = ais.AIShieldApi(api_url=API_URL, api_key=API_KEY, org_id=ORG_ID)

### Perform Model Registration and Upload the Input Artifacts

    status, job_details = client.register_model(task_type=task_type, analysis_type=analysis_type)
    model_id = job_details.model_id
    print('Model id: {} \nInput artifacts will be uploaded as:\n data_upload_uri: {}\n label_upload_uri: {}'
      '\n model_upload_uri: {}'.format(model_id, job_details.data_upload_uri, job_details.label_upload_uri,
                                       job_details.model_upload_uri))

    upload_status = client.upload_input_artifacts(job_details= job_details,
                                                  data_path=DATA_PATH,
                                                  label_path=LABEL_PATH,
                                                  model_path=MODEL_PATH)
    print('Upload status: {}'.format(', '.join(upload_status)))
    
    -------------
    Sample output
    -------------
    Model id: xxxx-xxx-xxx-xxx-xxxxx 
    Input artifacts will be uploaded as:
     data_upload_uri: https://xxx.xxx/xxx
     label_upload_uri: https://xxx.xxx/xxx
     model_upload_uri: https://xxx.xxx/xxx

    Upload status: data file upload successful, label file upload successful, model file upload successful

### Run Analysis:
    
    vuln_config = ais.VulnConfig(task_type=task_type, analysis_type=analysis_type, defense_generate=True)
    vuln_config.input_dimensions = (28,28,1)  # input dimension for mnist digit classification
    vuln_config.number_of_classes = 10  # number of classes for mnist digit classification
    vuln_config.encryption_strategy = 0  # value 0 (or) 1, if model is unencrypted or encrypted(pyc) respectively
    print('IC-Extraction parameters are: \n {} '.format(vuln_config.get_all_params()))

    my_status, job_details = client.vuln_analysis(model_id=model_id, vuln_config=vuln_config)
    
    my_job_id = job_details.job_id
    print('status: {}. Job_id: {} .job_monitor_uri: {}'.format(my_status, my_job_id, job_details.job_monitor_uri))
    
    -------------
    Sample output
    -------------
    IC-Extraction parameters are:
        {'attack': <Attack.EXTRACTION: 'extraction'>,
         'attack_type': 'blackbox',
         'defense_bestonly': 'no',
         'encryption_strategy': 0,
         'input_dimensions': '(28, 28, 1)',
         'model_api_details': '',
         'model_framework': 'tensorflow',
         'normalize_data': 'yes',
         'number_of_attack_queries': 200,
         'number_of_classes': 10,
         'task_type': <Task.IMAGE_CLASSIFICATION: 'image_classification'>,
         'use_model_api': 'no',
         'vulnerability_threshold': 0}


    status: success. Job_id: xxxx-xxxx. Please save this job_id for future reference
    job_monitor_uri: http://xxxx.xxx/?type=ImageClassification&jobid=xxxx-xxxx

### Monitor the Job Status:
    
    my_status = client.job_status (job_id = my_job_id)

    -------------
    Sample output
    -------------
    Job when running:
        Fetching job details for job id: xxxx.xxx  ...
        ModelExploration_Status : completed
        SanityCheck_Status : passed
        running...

    Job when completed:
        Fetching job details for job id: xxxx.xxx  ...
        ModelExploration_Status : completed
        SanityCheck_Status : passed
        QueryGenerator_Status : completed
        VunerabilityEngine_Status : completed
        DefenseReport_Status : completed
        job run completed

### Get Reports/Artifacts:

    if my_status == "success":
            output_conf = ais.OutputConf(report_type=ais.get_type("report", "defense"),
                                         file_format=ais.get_type("file_format", "pdf"),
                                         save_folder_path=OUTPUT_PATH)
        my_report = client.save_job_report(job_id=my_job_id, output_config=output_conf)
    
    -------------
    Sample output
    -------------
    defense_xxxx_xxxx.pdf is saved in {save_folder_path}