AIShield Package Example Usage
=============
![aishield](https://aisdocs.blob.core.windows.net/images/aishieldLogoPypi.PNG)

-------------
This document provides an overview of the use of the AIShield package. 

Import SDK as:

    import aishield as ais

Provide Subscription Details:
    
    API_URL = 'https://xxx.xxx.xxx/ic/ais/ImageClassification/VulnerabiltyReport'
    TOKEN = "xxxx"

Provide Input and Output Paths:
    
    DATA_PATH = r'/data/original_data.zip'
    LABEL_PATH = r'/data/original_label.zip'
    MODEL_PATH = r'/model/mnist_model.zip'
    OUTPUT_PATH = r'/output'

Run Analysis:
    
    client = ais.AIShieldApi(api_url=API_URL, auth_token=TOKEN)

    vuln_config = ais.VulnConfig(task_type=ais.get_type("task", "image_classification"),
                                 attack=ais.get_type("attack", "extraction"),
                                 defense_generate=True)
    vuln_config.input_dimensions = (28,28,1)  # input dimension for mnist digit classification
    vuln_config.number_of_classes = 10  # number of classes for mnist digit classification
    vuln_config.encryption_strategy = 0  # value 0 (or) 1, if model is unencrypted or encrypted(pyc) respectively
    print('IC-Extraction parameters are: \n {} '.format(vuln_config.get_all_params()))

    my_status, job_details = client.vuln_analysis(
        data_path=DATA_PATH,
        label_path=LABEL_PATH,
        model_path=MODEL_PATH,
        vuln_config=vuln_config
    )

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


    status: success. Job_id: gAAAAABjaNkYdCWZDekmBc2Gh-yvYgWjFVIQdTFN2GoVZhMEl3YhOj__5DCBnGWSkaXOHPAGeOFN1K_3PUaKp5Nr4pQeVwF7Jg==. Please save this job_id for future reference
    job_monitor_uri: https://xxx.xxx.net/?type=ImageClassification&jobid=gAAAAABjaNkYdCWZDekmBc2Gh-yvYgWjFVIQdTFN2GoVZhMEl3YhOj

Monitor the Job Status:
    
    my_status = client.job_status (job_id = my_job_id)

    -------------
    Sample output
    -------------
    Job when running:
        Fetching job details for job id: gAAAAABjaNkYdCWZDekmBc2Gh-yvYgWjFVIQdTFN2GoVZhMEl3YhOj__5DCBnGWSkaXOHPAGeOFN1K_3PUaKp5Nr4pQeVwF7Jg==  ...
        ModelExploration_Status : completed
        SanityCheck_Status : passed
        running...

    Job when completed:
        Fetching job details for job id: gAAAAABjaNkYdCWZDekmBc2Gh-yvYgWjFVIQdTFN2GoVZhMEl3YhOj__5DCBnGWSkaXOHPAGeOFN1K_3PUaKp5Nr4pQeVwF7Jg==  ...
        ModelExploration_Status : completed
        SanityCheck_Status : passed
        QueryGenerator_Status : completed
        VunerabilityEngine_Status : completed
        DefenseReport_Status : completed
        job run completed

Get Reports/Artifacts:

    if my_status == "success":
            output_conf = ais.OutputConf(report_type=ais.get_type("report", "defense"),
                                         file_format=ais.get_type("file_format", "pdf"),
                                         save_folder_path=OUTPUT_PATH)
        my_report = client.save_job_report(job_id=my_job_id, output_config=output_conf)
    
    -------------
    Sample output
    -------------
    defense_20221107_1545.pdf is saved in {save_folder_path}
