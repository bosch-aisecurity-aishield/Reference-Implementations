from clearml import PipelineDecorator


@PipelineDecorator.component(cache=True)
def create_dataset(source_url: str, project: str, dataset_name: str) -> str:
    print("starting create_dataset")
    from clearml import Dataset
    from urllib import request
    import pandas as pd
    import os
    import zipfile
    local_file = os.path.split(source_url)[-1]
    # Download remote and save locally
    request.urlretrieve(source_url, local_file)

    # unzip and load defense model
    def zip_extractor(file, extract_path=None, delete_zip=False):
        """
        extract zip file to the given path

        Parameters
        ----------
        file : path of zip file
        extract_path : path to extract zip file, default considered parent directory
        delete_zip: True, delete zip file after unzipping it

        Returns
        -------
        None.
        """
        if extract_path is None:
            extract_path = os.path.dirname(file)
        print("Extracting : {}".format(file))
        zf = zipfile.ZipFile(file=file, mode='r')
        zf.extractall(extract_path)
        zf.close()
        if delete_zip:
            os.remove(file)
            print("{} removed successfully.".format(file))

    zip_extractor(file="./{}".format(local_file), extract_path=None)

    df = pd.read_csv('bank-additional/bank-additional-full.csv', delimiter=";")

    df.to_csv(path_or_buf="./dataset.csv", index=False)
    dataset = Dataset.create(dataset_project=project, dataset_name=dataset_name)
    dataset.add_files("./dataset.csv")
    dataset.get_logger().report_table(title="sample", series="head", table_plot=df.head())
    dataset.finalize(auto_upload=True)
    print("done create_dataset")
    return dataset.id


@PipelineDecorator.component(cache=True)
def preprocess_dataset(dataset_id: str):
    print("starting preprocess dataset")
    from clearml import Dataset
    import pandas as pd
    import os
    from sklearn.preprocessing import LabelEncoder

    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    df = pd.read_csv(os.path.join(local_folder, "dataset.csv"))
    print("Data shape is : {}".format(df.shape))
    df.replace(['basic.6y', 'basic.4y', 'basic.9y'], 'basic', inplace=True)
    columns = ['day_of_week', 'month']
    df.drop(columns, axis=1, inplace=True)
    '''Description : Label Encoding the Categorical Columns'''

    le = LabelEncoder()
    df.job = le.fit_transform(df.job)
    df.marital = le.fit_transform(df.marital)
    df.education = le.fit_transform(df.education)
    df.housing = le.fit_transform(df.housing)
    df.loan = le.fit_transform(df.loan)
    df.poutcome = le.fit_transform(df.poutcome)
    df.contact = le.fit_transform(df.contact)
    df.default = le.fit_transform(df.default)
    df.y = le.fit_transform(df.y)

    df.to_csv(path_or_buf="./processed_dataset.csv", index=False)

    # store in a new dataset
    new_dataset = Dataset.create(
        dataset_project=dataset.project, dataset_name="{} v2".format(dataset.name),
        parent_datasets=[dataset]
    )
    new_dataset.add_files("./processed_dataset.csv")
    new_dataset.get_logger().report_table(title="sample", series="head", table_plot=df.head())
    new_dataset.finalize(auto_upload=True)

    print("done preprocess_dataset")
    return new_dataset.id


@PipelineDecorator.component()
def train_model(dataset_id: str, training_args: dict):
    print("starting train_model")
    from clearml import Dataset, Task
    import os
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import shutil
    import numpy as np
    import pickle

    task = Task.current_task()

    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    df = pd.read_csv(os.path.join(local_folder, "processed_dataset.csv"))
    output = 'y'

    X = df.loc[:, df.columns != output]
    y = df['y']

    '''
    Description : Splitting data for validation
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    '''
    Description : Check size of dataset
    '''
    print("shape of x_train: ", X_train.shape)
    print("shape of y_train: {}".format(y_train.shape))
    print(f'shape of x_test: {X_test.shape}')
    print(f'shape of y_test: {y_test.shape}')

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print('Accuracy score of the Decision Tree model is {}'.format(metrics.accuracy_score(y_test, pred)))
    print('F1 score of the Decision Tree model is {}'.format(metrics.f1_score(y_test, pred)))
    print('Confusion Matrix \n {}'.format(metrics.confusion_matrix(y_test, pred)))
    print('Classification Report \n {}'.format(metrics.classification_report(y_test, pred)))

    # saving model as artifact on clearml
    modelname = "xgboostmodel.pkl"

    # save sample data , model and label for aishield api
    # setting path
    data_path = "./data"
    label_path = "./label"
    model_path = "./model"
    zip_path = "./zip_file"

    for path in [data_path, label_path, model_path, zip_path]:
        if os.path.isdir(path):
            if os.path.isfile(path):
                os.remove(path=path)
            else:
                shutil.rmtree(path=path)

        os.mkdir(path=path)

    pickle.dump(model, open(model_path + "/" + modelname, 'wb'))

    # saving model as artifact on clearml
    # task.upload_artifact(name='xgboost_model', artifact_object=model)
    task.update_output_model(model_name="xgboost_model", model_path=model_path + "/" + modelname,
                             auto_delete_file=False)

    df1 = pd.DataFrame(X_test)
    df1['y'] = y_test
    df1.to_csv(os.path.join(data_path, "banking_preprocessed_data.csv"), index=False)

    min_values = df.min().to_numpy()
    max_values = df.max().to_numpy()
    x = np.array([min_values, max_values])
    df_m = pd.DataFrame(x, columns=df.columns)
    df_m.to_csv(label_path + "/minmax.csv", index=False)

    # zip model and save in zip_file folder
    shutil.make_archive(base_name=os.path.join(zip_path, 'model'), format="zip", root_dir=model_path)

    # zip data and save in zip_file folder
    shutil.make_archive(base_name=os.path.join(zip_path, 'data'), format="zip", root_dir=data_path)

    # zip model and save in zip_file folder
    shutil.make_archive(base_name=os.path.join(zip_path, 'minmax'), format="zip", root_dir=label_path)

    # save data to clearml server
    dataset = Dataset.create(dataset_project="xgboost_bfsi_demo", dataset_name="sample_banking_data")
    dataset.add_files(zip_path)
    dataset.finalize(auto_upload=True)

    print("done train_model dataset id : {}".format(dataset.id))
    return dataset.id


@PipelineDecorator.component()
def model_registration(dataset_id: str, url: str, org_id: str, x_api_key: str):
    from clearml import Dataset
    import os
    import requests
    import json

    dataset = Dataset.get(dataset_id=dataset_id)
    local_folder = dataset.get_local_copy()
    data_path = os.path.join(local_folder, "data.zip")
    label_path = os.path.join(local_folder, "minmax.zip")
    model_path = os.path.join(local_folder, "model.zip")

    print("model path: {}, data path : {} and label path: {}".format(model_path, data_path, label_path))

    model_registration_url = url + "/model_registration/upload"
    model_registration_payload = {
        'task_type': "TC",
        "analysis_type": "MEA"
    }
    headers = {"org-id": org_id,
               "x-api-key": x_api_key,
               "Cache-Control": 'no-cache'
               }
    new_request = requests.request(method="POST", url=model_registration_url, headers=headers,
                                   data=json.dumps(model_registration_payload))
    new_request = json.loads(new_request.text)
    print("response received is : {}".format(new_request))
    model_id = new_request['data']['model_id']
    print('model_id: ', model_id)
    data_upload_url = new_request['data']['urls']['data_upload_url']
    label_upload_url = new_request['data']['urls']['minmax_upload_url']
    model_upload_url = new_request['data']['urls']['model_upload_url']

    def upload_file(url, file_path):
        """
        url: URL to upload
        file_path: file to be uploaded
        """
        new_request = requests.request(method="PUT", url=url, data=open(file_path, 'rb'))
        status_cd = new_request.status_code
        if status_cd == 200:
            status = 'upload sucessful'
        else:
            status = 'upload failed'
        return status

    data_upload_status = upload_file(data_upload_url, data_path)
    label_upload_status = upload_file(label_upload_url, label_path)
    model_upload_status = upload_file(model_upload_url, model_path)

    print('data_upload_status: ', data_upload_status)
    print('minmax upload_status: ', label_upload_status)
    print('model_upload_status: ', model_upload_status)

    return model_id


@PipelineDecorator.component()
def model_analysis(model_id: str, payload: dict, url: str, org_id: str, x_api_key: str):
    import requests
    import json

    model_analysis_url = url + "/model_analyse/{}".format(model_id)
    headers = {"org-id": org_id,
               "x-api-key": x_api_key,
               "Cache-Control": 'no-cache'
               }
    new_request = requests.request(method="POST", url=model_analysis_url, json=payload, headers=headers)
    new_request = json.loads(new_request.text)
    for k, v in new_request.items():
        print("* {} : {}".format(k, v))

    return new_request['job_id']


@PipelineDecorator.component()
def monitor_api_progress(new_job_id: str, payload: dict, url: str, org_id: str, x_api_key: str):
    import time
    import requests
    import json
    from clearml import Task
    job_status_url = url + "/job_status_detailed?job_id=" + new_job_id

    # status dictionary
    status_dictionary = {
        'ModelExploration_Status': 'na',
        'SanityCheck_Status': 'na',
        'QueryGenerator_Status': 'na',
        'VunerabilityEngine_Status': 'na',
        'DefenseReport_Status': 'na',
    }
    headers = {"org-id": org_id,
               "x-api-key": x_api_key,
               "Cache-Control": 'no-cache'
               }
    counts = [0] * len(status_dictionary)
    failed_api_hit_count = 0
    while True:
        time.sleep(2)
        try:
            job_status_response = requests.request("GET", job_status_url, params={},
                                                   headers=headers)

            job_status_payload = json.loads(job_status_response.text)
            failing_key = 'ModelExploration_Status'
            for i, key in enumerate(status_dictionary.keys()):
                if status_dictionary[key] == 'na':
                    if job_status_payload[key] == 'inprogress' and status_dictionary[key] == 'na':
                        status_dictionary[key] = job_status_payload[key]
                        print(str(key), ":", status_dictionary[key])

                    elif job_status_payload[key] == 'completed' or job_status_payload[key] == 'passed':
                        status_dictionary[key] = job_status_payload[key]
                        counts[i] += 1
                        print(str(key), ":", status_dictionary[key])

                    if job_status_payload[key] == 'failed':
                        failing_key = key
                        status_dictionary[key] = job_status_payload[key]
                        print(str(key), ":", status_dictionary[key])

                elif job_status_payload[key] == 'completed' or job_status_payload[key] == 'passed':
                    status_dictionary[key] = job_status_payload[key]
                    if counts[i] < 1:
                        print(str(key), ":", status_dictionary[key])
                    counts[i] += 1

                else:
                    if job_status_payload[key] == 'failed':
                        failing_key = key
                        status_dictionary[key] = job_status_payload[key]
                        print(str(key), ":", status_dictionary[key])

            if job_status_payload[failing_key] == 'failed':
                break

            if status_dictionary['VunerabilityEngine_Status'] == 'passed' or status_dictionary[
                'VunerabilityEngine_Status'] == 'completed' and job_status_payload[
                'CurrentStatus'] == "Defense generation is not triggered":
                print("\n Vulnerability score {} failed to cross vulnerability threshold of {}".format(
                    job_status_payload['VulnerabiltyScore'], payload['vulnerability_threshold']))
                break
            if job_status_payload['DefenseReport_Status'] == 'completed':
                break
        except Exception as e:
            failed_api_hit_count += 1
            print("Error {}. trying {} ...".format(str(e), failed_api_hit_count))
            if failed_api_hit_count >= 3:
                break

    # push vulnerability score to dashboard
    task = Task.current_task()
    print(job_status_payload)
    task.connect(job_status_payload)
    return status_dictionary, job_status_payload


@PipelineDecorator.component()
def download_artifact(job_id: str, url: str, org_id: str, x_api_key: str, file_format: int = 0, report_type=None):
    """
    job_id: job_id  received after successful api call
    report_type: report to be downloaded
    file_format: change file_format to : 0- all report in zip
                        1- report in .txt
                        2- report in .pdf
                        3- report in .json
                        4- report in .xml
    """
    import requests
    import os
    import shutil
    from clearml import Dataset, Task
    import zipfile
    import pickle
    import imp
    import numpy as np
    import pandas as pd
    import json

    def zip_extractor(file, extract_path=None, delete_zip=False):
        """
        extract zip file to the given path

        Parameters
        ----------
        file : path of zip file
        extract_path : path to extract zip file, default considered parent directory
        delete_zip: True, delete zip file after unzipping it

        Returns
        -------
        None.
        """
        if extract_path is None:
            extract_path = os.path.dirname(file)
        print("Extracting : {}".format(file))
        zf = zipfile.ZipFile(file=file, mode='r')
        zf.extractall(extract_path)
        zf.close()
        if delete_zip:
            os.remove(file)
            print("{} removed successfully.".format(file))

    report_path = "./Artifacts/{}".format(job_id)
    vulnerability_report = os.path.join(report_path, 'vulnerability_report')
    defense_report = os.path.join(report_path, 'defense_report')
    sample_data = os.path.join(report_path, 'sample_data')
    defense_artifact = os.path.join(report_path, 'defense_artifact')

    for path in [report_path, vulnerability_report, defense_report, sample_data, defense_artifact]:
        if os.path.isdir(path):
            if os.path.isfile(path):
                os.remove(path=path)
            else:
                shutil.rmtree(path=path)

        os.makedirs(path)

    reports_type = []
    if report_type is None:
        reports_type.append("Vulnerability")
        reports_type.append("Attack_samples")
        reports_type.append("Defense")
        reports_type.append("Defense_artifact")
        file_format = 0

    else:
        reports_type.append(report_type)

    for report_type in reports_type:
        print("received report_type : {} and file format is: {}".format(report_type, file_format))
        report_url = url + "/" + "get_report?job_id=" + str(
            job_id) + "&report_type=" + report_type + "&file_format=" + str(file_format)

        headers = {"org-id": org_id,
                   "x-api-key": x_api_key,
                   "Cache-Control": 'no-cache'
                   }
        headers1 = headers
        headers1["content-type"] = "application/zip"

        response = requests.request("GET", report_url, params={}, headers=headers1)

        if file_format == 0 or file_format == "Attack_samples":
            file_path = os.path.join(report_path, report_type + ".zip")
            with open(file_path, 'wb') as f:
                f.write(response.content)

            if report_type.lower() == "vulnerability".lower():
                zip_extractor(file=file_path, extract_path=vulnerability_report)
            elif report_type.lower() == "Attack_samples".lower():
                zip_extractor(file=file_path, extract_path=sample_data)
            elif report_type.lower() == "Defense".lower():
                zip_extractor(file=file_path, extract_path=defense_report)
            elif report_type.lower() == "Defense_artifact".lower():
                zip_extractor(file=file_path, extract_path=defense_artifact)

        elif file_format == 1:
            with open(os.path.join(report_path, report_type + ".txt"), 'wb') as f:
                f.write(response.content)
        elif file_format == 2:
            with open(os.path.join(report_path, report_type + ".pdf"), 'wb') as f:
                f.write(response.content)
        elif file_format == 3:
            with open(os.path.join(report_path, report_type + ".json"), 'wb') as f:
                f.write(response.content)
        elif file_format == 4:
            with open(os.path.join(report_path, report_type + ".xml"), 'wb') as f:
                f.write(response.content)

    task = Task.current_task()
    # read vulnerability reports
    print("Reading vulnerability report: \n\n\n")
    f = open(os.path.join(vulnerability_report, "Vulnerability.txt"), "r")
    print(f.read())

    # logging report to clerml as artifact
    task.upload_artifact(name='Vulnerability Report', artifact_object=vulnerability_report)

    print("Reading defense report: \n\n\n")
    f = open(os.path.join(defense_report, "Defense.txt"), "r")
    print(f.read())

    # logging report to clerml as artifact
    task.upload_artifact(name='Defense Report', artifact_object=defense_report)

    # upload attack sample to clearml as artifact
    task.upload_artifact('attack sample', artifact_object=sample_data)

    # upload defense artifact to clearml as artifact
    task.upload_artifact('defense artifact', artifact_object=defense_artifact)

    # Description: Load defense model

    defense_model_path = os.path.join(defense_artifact, 'defense_model.pkl')
    defense_model = pickle.load(open(defense_model_path, 'rb'))

    # saving model as artifact on clearml
    # task.upload_artifact(name='defense_model', artifact_object=defense_model)
    task.update_output_model(model_name="defense_model", model_path=defense_model_path, auto_delete_file=False)

    # Description: Use defense model
    module_path = os.path.join(defense_artifact, 'predict.py')
    predict = imp.load_source("predict", module_path)
    defense = predict.AISDefenseModel(defense_model, model_framework="scikit-learn")

    # read sample data
    sample_data_path = os.path.join(sample_data, 'attack_samples.csv')
    df_sample = pd.read_csv(sample_data_path)
    X_sample = np.array(df_sample)

    attack_label, attack_prob = defense.predict(X_sample)
    print("attack_label: {}".format(attack_label))
    print("attack_prob: {}".format(attack_prob))

    # save data to clearml server
    artifact = Dataset.create(dataset_project="AIShield-pipeline-demos", dataset_name="AIShield-Artifact")
    artifact.add_files(report_path)
    artifact.finalize(auto_upload=True)
    return artifact.id


@PipelineDecorator.pipeline(
    name='xgboost_pipeline',
    project='xgboost_bfsi_demo',
    version='0.2'
)
def pipeline(data_url: str, project: str):
    from clearml import Task

    url = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # provide url here
    org_id = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  # provide org-id here
    x_api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # provide x-api-key here

    dataset_id = create_dataset(source_url=data_url, project=project, dataset_name="banking-data")

    preprocessed_dataset_id = preprocess_dataset(dataset_id=dataset_id)
    sample_data_id = train_model(dataset_id=preprocessed_dataset_id, training_args=None)
    model_id = model_registration(sample_data_id, url, org_id, x_api_key)
    payload = {
        "model_details": "NA",
        "use_model_api": "no",
        "model_id": model_id,
        'input_dimensions': "10297,18",
        "number_of_classes": 2,
        "number_of_attack_queries": 50000,
        "vulnerability_threshold": 0,
        "attack_type": 'blackbox',
        "encryption_strategy": 0,  # no encryption
        "normalize_data": "yes",
        "defense_bestonly": "yes",
        "is_category_columns": "yes",
        "categorical_columns_info": "job,marital,education,default,housing,loan,contact,poutcome",
        "model_api_details": "na",
        "model_framework": "scikit-learn"
    }

    job_id = model_analysis(model_id, payload, url, org_id, x_api_key)
    status_dictionary, job_status_payload = monitor_api_progress(job_id, payload, url, org_id, x_api_key)
    # push vulnerability score to dashboard
    task = Task.current_task()
    task.connect(job_status_payload)
    service_failed = None
    for key, value in status_dictionary.items():
        if value == 'failed':
            service_failed = key
            break
    if service_failed is None:
        artifact_id = download_artifact(job_id, url, org_id, x_api_key, file_format=0)
        print("Artifact id is {}".format(artifact_id))
    else:
        print("Model Analysis failed at {}".format(service_failed))

    print("selected model_id = {}".format(model_id))


if __name__ == "__main__":
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
    PipelineDecorator.run_locally()

    pipeline(data_url=url, project="xgboost_bfsi_demo")