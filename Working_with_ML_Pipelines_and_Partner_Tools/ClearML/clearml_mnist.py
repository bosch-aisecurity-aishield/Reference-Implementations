# import library
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
import cv2
from clearml import Task
import py_compile
import requests
import json
import time
import zipfile
from humanfriendly import format_timespan

task = Task.init(project_name='ClearML integration', task_name='AIShield clearml integration')


def make_directory(directory):
    """
    create directory

    Parameters
    ----------
    directorys : list containing the directorys path to create
    Returns
    -------
    None.

    """
    for d in directory:
        if os.path.isdir(d):
            print("directory {} already exist".format(d))
        if os.path.isdir(d) == False:
            os.mkdir(path=d)
            print("directory {} created successfully".format(d))


def delete_directory(directorys):
    """
    delete directory

    Parameters
    ----------
    directorys : list containing the directorys to deleate along with all the files

    Returns
    -------
    None.

    """
    if len(directorys) >= 1:
        for d in directorys:
            if os.path.isdir(d):
                try:
                    if os.path.isfile(d):
                        os.remove(path=d)
                    else:
                        shutil.rmtree(path=d)
                        print("Removed: {}".format(d))
                except:
                    print("Failed to removed: {}".format(d))


def make_archive(base_name, root_dir, zip_format='zip'):
    """
    created zip for given folder

    Parameters
    ----------
    base_name : name of zip file
    root_dir : directory to archive/zip
    zip_format : zip or tar
        DESCRIPTION. The default is 'zip'.

    Returns
    -------
    None.

    """
    shutil.make_archive(base_name=base_name, format=zip_format, root_dir=root_dir)


def plot(x, y=None, row: int = 2):
    """
    to visualize random sample
    """
    rows = row
    random_indices = random.sample(range(x.shape[0]), rows * rows)
    sample_images = x[random_indices, :]
    if y is not None:
        sample_labels = y[random_indices]

    fig, axs = plt.subplots(nrows=rows, ncols=rows, figsize=(12, 9), sharex=True, sharey=True)
    for i in range(rows * rows):
        subplot_row = i // rows
        subplot_col = i % rows
        axs[subplot_row, subplot_col].imshow(sample_images[i, :])
        if y is not None:
            axs[subplot_row, subplot_col].set_title("Class. %d" % sample_labels[i])
    plt.tight_layout()
    plt.show()


def get_file_path(path):
    """
    To get full file path from directory and child directory
    Args:
        path: Directory path

    Returns: list containing full path of all the file from the given directory

    """
    list_id = []
    for (root, dirs, files) in os.walk(path, topdown=True):
        if len(dirs) == 0:
            for file in files:
                list_id.append(os.path.join(root, file))
    return list_id


# set path
"""
Description : Create data, model and label folder
"""
mnist_data_path = os.path.join(os.getcwd(), "data")
mnist_model_path = os.path.join(os.getcwd(), "model")
mnist_label_path = os.path.join(os.getcwd(), "label")
zip_path = os.path.join(os.getcwd(), "zip")
pyc_model_path = os.path.join(os.getcwd(), "pyc_model")
report_path = os.path.join(os.getcwd(), "reports")
sample_data = os.path.join(report_path, "sample_data")
defense_artifact = os.path.join(report_path, "defense_artifact")
# deleting folder
delete_directory(
    directorys=[mnist_data_path, mnist_model_path, mnist_label_path, zip_path, pyc_model_path, report_path])

# creating folder
make_directory([mnist_data_path, mnist_model_path, mnist_label_path, zip_path, pyc_model_path, report_path, sample_data,
                defense_artifact])

# download mnist dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# split dataset for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1, random_state=42)

# print and check train, test and validation shape
print("shape of x_train: ", X_train.shape)
print("shape of y_train: {}".format(y_train.shape))
print(f'shape of x_test: {X_test.shape}')
print(f'shape of y_test: {y_test.shape}')
print(f'shape of x_val: {X_val.shape}')
print(f'shape of y_val: {y_val.shape}')

# visualize sample training images
plot(x=X_train, y=y_train, row=3)

# set parameter value
img_row, img_col, channel = 28, 28, 1
num_classes = 10
input_shape = (img_row, img_col, channel)
epochs = 20
batch_size = 64
configuration_dict = {
    'num_classes': num_classes,
    'input_shape': input_shape,
    'epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': 1e-3,
    'encryption_strategy': 0,
    'attack_type': 'blackbox',
    'number_of_attack_queries': 80000,
    'model_framework': 'tensorflow',
    'vulnerability_threshold': 0,
    'normalize_data': 'yes',
    'defense_bestonly': 'no',
    'use_model_api': 'no',
    'model_api_details': 'no',
    'url': "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # provide url here
    "headers": {'Cache-Control': 'no-cache',
                'Org-Id': 'xxxxxxxxxxxxxxxxxxxxxxxxx',  # provide org-id here
                'x-api-key': "xxxxxxxxxxxxxxxxxxxxxxxxx",  # provide x-api-key here
                },
    'number_of_sample_to_save': None,
    'api_call': True
}
# adding configuration to clearml
task.connect(configuration_dict)

# normalize and reshape data to the input shape
X_train = X_train.reshape(-1, *configuration_dict['input_shape']) / 255.0
X_val = X_val.reshape(-1, *configuration_dict['input_shape']) / 255.0
X_test = X_test.reshape(-1, *configuration_dict['input_shape']) / 255.0

# convert label to one hot encoder
y_train = tf.keras.utils.to_categorical(y_train, configuration_dict['num_classes'])
y_test = tf.keras.utils.to_categorical(y_test, configuration_dict['num_classes'])


def create_model(shape: tuple = input_shape, classes: int = num_classes, model_weight=None):
    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=shape),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes, activation='softmax')
    ])
    if model_weight is None:
        # Compile model
        cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',
                          metrics=['accuracy'])
    else:
        print("model weight loaded successfully")
        cnn_model.load_weights(model_weight)

    return cnn_model


# create model architecture
model = create_model(shape=input_shape, classes=num_classes)

# visualize model architecture
print("model architecture is : ")
model.summary()

"""
Description: train model
"""


def train_model(model, X_train, y_train, X_test, y_test, batch_size=16, epochs=5, filename='mnist_model'):
    '''
    Description:Training  model
    Args:
        model:model to train
        X_train: X_train for training
        X_test: for validation
        y_train: label for X_train
        y_test: label for X_test
        batch_size: batch size for training model
        epochs: number of epochs to train model
        filename : name to save extracted model
    '''
    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filename + '.h5', monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='auto')
    ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='min',
                                          restore_best_weights=True)

    # train and record time for training
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[ES, checkpoint],
                        validation_data=(X_test, y_test))  # tensorboard_callback,,validation_split=0.05
    elapsed_time = time.time() - start_time

    print("Elapsed time: {}".format(format_timespan(elapsed_time)))

    return model, history


def Plot_model_training_parameters(training_history):
    """
    Description : plot model to visualize model training

    Args:
        training_history: history from which model training parameter can be taken for plotting
    """
    loss = training_history.history['loss']
    accuracy = training_history.history['accuracy']
    val_loss = training_history.history['val_loss']
    val_accuracy = training_history.history['val_accuracy']
    epochs_range = np.arange(1, len(loss) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Training loss ")
    plt.xlabel('Epochs', fontsize=16, fontweight='bold')
    plt.ylabel("Loss", fontsize=16, fontweight='bold')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracy, label="Training Accuracy")
    plt.plot(epochs_range, val_accuracy, label="Validation Accuracy")
    plt.title("Training accuracy")
    plt.xlabel('Epochs', fontsize=16, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=16, fontweight='bold')
    plt.legend()
    plt.show()


model, history = train_model(model, X_train, y_train, X_test, y_test, batch_size=configuration_dict['batch_size'],
                             epochs=configuration_dict['epochs'], filename='mnist_model')

"""
Description: Plot model learning 
"""
Plot_model_training_parameters(history)

model.save(os.path.join(mnist_model_path, 'mnist_model.h5'))

# evaluate trained model on unseen data
loss, accuracy = model.evaluate(X_test, y_test)
print("* Loss: {} \n* Accuracy: {}".format(loss, accuracy))

if configuration_dict['api_call']:
    # Prepare artifact for api call
    def save_sample_data_label(x, y, sample_to_save: int = None):
        """
        Description: Save data and label
        """
        label = pd.DataFrame()
        img_name = []
        img_label = []
        # Create data and label directory
        n = sample_to_save if sample_to_save is not None else x.shape[0]
        print("number of sample saved is : {}".format(n))
        for i in range(n):
            cv2.imwrite(os.path.join(mnist_data_path, str(i) + ".jpg"), x[i] * 255.0)
            img_name.append(str(i) + ".jpg")
            img_label.append(y_val[i])
        label['image'] = img_name
        label["label"] = np.array(img_label)

        # write orig_label dataframe
        label.to_csv(os.path.join(mnist_label_path, "label.csv"), index=False)


    # call function
    save_sample_data_label(x=X_val, y=y_val, sample_to_save=configuration_dict['number_of_sample_to_save'])

    if configuration_dict["encryption_strategy"]:
        # save model weight to save in pyc format
        model_wight_path = os.path.join(pyc_model_path, "model_weight.h5")
        model.save_weights(model_wight_path)

        # verify that model weight is loading
        model = create_model(shape=input_shape, classes=num_classes, model_weight=model_wight_path)

        # python code
        python_code = '''# import library
import numpy as np
import tensorflow as tf


# define class
class BaseModel:
    """
    class for base model
    """
    def __init__(self, input_shape=(28, 28, 1), num_classes=10, model_weight_path="model_weight.h5"):
        """
        constructor for class

        Parameters
        ----------
        input_shape : TYPE, optional
            DESCRIPTION. The default is (28,28,1).
        num_classes : TYPE, optional
            DESCRIPTION. The default is 10.
        model_weight_path : string, optional
            DESCRIPTION. the relative path to model weight

        Returns
        -------
        None.

        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_weight_path = model_weight_path

    def cnn_model(self, input_shape, num_classes):
        """
        model architecture

        Parameters
        ----------
        input_shape : string
            DESCRIPTION.input_shape for model e.g (28,28,1)
        num_classes : string
            DESCRIPTION.number of class e.g 10

        Returns
        -------
        model : model
            DESCRIPTION.

        """
        # create sequential model

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def predict(self, x):
        """
        predict for given data

        Parameters
        ----------
        x : numpy array
            DESCRIPTION.

        Returns
        -------
        pred : numpy array
            DESCRIPTION.

        """
        model = self.cnn_model(input_shape=self.input_shape, num_classes=self.num_classes)
        model.load_weights(self.model_weight_path)

        pred = np.argmax(model.predict(x), axis=-1)
        return pred
         '''

        # Writing to file
        with open("base_model.py", "w") as file:
            # Writing data to a file
            file.writelines(python_code)

        # convert to .pyc format
        py_compile.compile(file='base_model.py', cfile=os.path.join(pyc_model_path, 'base_model.pyc'))

        # check if pyc model working
        from pyc_model.base_model import BaseModel

        base_model = BaseModel(model_weight_path=r"pyc_model/model_weight.h5")

        print("model prediction is : ", base_model.predict(X_val[:1]))
        plt.imshow(X_val[0])
        plt.show()

    # create zip_path directory to save zipped artifact
    # zip the model
    if os.path.isfile(os.path.join(zip_path, "model.zip")):
        delete_directory(directory=[os.path.join(zip_path, "model.zip")])
    if configuration_dict["encryption_strategy"]:
        make_archive(base_name=os.path.join(zip_path, "model"), root_dir=pyc_model_path, zip_format='zip')
    else:
        make_archive(base_name=os.path.join(zip_path, "model"), root_dir=mnist_model_path, zip_format='zip')
    # data and label
    make_archive(base_name=os.path.join(zip_path, "data"), root_dir=mnist_data_path, zip_format='zip')
    make_archive(base_name=os.path.join(zip_path, "label"), root_dir=mnist_label_path, zip_format='zip')

    # AIShield api call
    """
    Description: call Model registration api to get unique model it and url to upload data, model and label
    """
    model_registration_url = configuration_dict["url"] + "/AIShieldModelRegistration/v1.5"
    model_registration_payload = {
        'task_type': "IC",
        "analysis_type": "MEA"
    }
    new_request = requests.request(method="POST", url=model_registration_url, headers=configuration_dict['headers'],
                                   params=model_registration_payload)
    print("Received response is : {}".format(new_request.text))
    new_request = json.loads(new_request.text)['Data']
    model_id = new_request['ModelID']
    data_upload_url = new_request['DataUploadURL']
    label_upload_url = new_request['LabelUploadURL']
    model_upload_url = new_request['ModelUploadURL']
    print('model_id: ', model_id)

    # get data file path
    data_path = os.path.join(zip_path, 'data.zip')  # full path of data zip
    label_path = os.path.join(zip_path, 'label.zip')  # full path of label zip
    model_path = os.path.join(zip_path, 'model.zip')  # full path of model zip


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


    """
    Description: Hit AIShield File Upload API
    """
    data_upload_status = upload_file(data_upload_url, data_path)
    label_upload_status = upload_file(label_upload_url, label_path)
    model_upload_status = upload_file(model_upload_url, model_path)

    print('data_upload_status: ', data_upload_status)
    print('label_upload_status: ', label_upload_status)
    print('model_upload_status: ', model_upload_status)

    """
    Description: Payload for AIShield VulnerabilityReport api call
    """
    payload = {}
    payload['model_id'] = model_id
    payload['input_dimensions'] = str(configuration_dict["input_shape"])
    payload['number_of_classes'] = str(configuration_dict['num_classes'])
    payload['attack_type'] = "blackbox"
    payload['number_of_attack_queries'] = configuration_dict['number_of_attack_queries']
    payload['model_framework'] = 'tensorflow'
    payload['vulnerability_threshold'] = "0"
    payload['normalize_data'] = "yes"
    payload['defense_bestonly'] = "no"
    payload['encryption_strategy'] = configuration_dict["encryption_strategy"]
    payload['model_api_details'] = "no"
    payload['use_model_api'] = "no"

    """
    Description: Hit AIShield VulnerabilityReport api
    """
    model_analysis_url = configuration_dict['url'] + "/AIShieldModelAnalysis/v1.5"
    if data_upload_status == "upload sucessful" and model_upload_status == "upload sucessful" and \
            label_upload_status == "upload sucessful":
        new_request = requests.request(method="POST", url=model_analysis_url, params=payload,
                                       headers=configuration_dict['headers'])
        new_request = json.loads(new_request.text)
        for k, v in new_request.items():
            print("* {} : {}".format(k, v))

    """
    Description: Get job id from api response
    """
    job_id = new_request['job_id']
    print(f"Job id : {job_id}")


    def monitor_api_progress(new_job_id):
        job_status_url = configuration_dict['url'] + "/AIShieldModelAnalysis/" + "JobStatusDetailed?JobID=" + new_job_id

        # status dictionary
        status_dictionary = {
            'ModelExploration_Status': 'na',
            'QueryGenerator_Status': 'na',
            'VunerabilityEngine_Status': 'na',
            'DefenseReport_Status': 'na',
        }
        counts = [0] * len(status_dictionary)
        failed_api_hit_count = 0
        while True:
            time.sleep(60*5)
            try:
                job_status_response = requests.request("GET", job_status_url, params={},
                                                       headers=configuration_dict['headers'])

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
                    print("\n Vulnerability score {} failed to cross vulnerability threshoold of {}".format(
                        job_status_payload['VulnerabiltyScore'], payload['vulnerability_threshold']))
                    break
                if job_status_payload['DefenseReport_Status'] == 'completed':
                    break
            except Exception as e:
                failed_api_hit_count += 1
                print("Error {}. trying {} ...".format(str(e), failed_api_hit_count))
                if failed_api_hit_count >= 3:
                    break
        return status_dictionary


    """
    Description: Continuos monitoring of jod progress
    """
    status_dictionary = monitor_api_progress(new_job_id=job_id)


    def download_artifact(job_id, report_type='Vulnerability', file_format=0):
        """
        job_id: job_id  received after successful api call
        report_type: report to be downloaded
        file_format: change file_format to : 0- all report in zip
                            1- report in .txt
                            2- report in .pdf
                            3- report in .json
                            4- report in .xml
        """
        print("received report_type : {} and file format is: {}".format(report_type, file_format))
        report_url = configuration_dict['url'] + "/AIShieldModelAnalysis/" + "GetReport?JobID=" + str(
            job_id) + "&ReportType=" + report_type + "&FileFormat=" + str(file_format)

        headers1 = configuration_dict['headers']
        headers1["content-type"] = "application/zip"

        response = requests.request("GET", report_url, params={}, headers=headers1)

        if file_format == 0 or file_format == "Attack_samples":
            with open(os.path.join(report_path, report_type + ".zip"), 'wb') as f:
                f.write(response.content)
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


    if status_dictionary["VunerabilityEngine_Status"] == 'completed':
        download_artifact(job_id=job_id, report_type='Vulnerability', file_format=1)
        download_artifact(job_id=job_id, report_type='Attack_samples', file_format=0)

    if status_dictionary["DefenseReport_Status"] == 'completed':
        download_artifact(job_id=job_id, report_type='Defense', file_format=1)
        download_artifact(job_id=job_id, report_type='Defense_artifact', file_format=0)

    # read vulnerability reports
    print("Reading vulnerability report: \n\n\n")
    f = open(os.path.join(report_path, "Vulnerability.txt"), "r")
    print(f.read())

    # logging report to clerml as artifact
    task.upload_artifact(name='Vulnerability Report', artifact_object=os.path.join(report_path, "Vulnerability.txt"))

    print("Reading defense report: \n\n\n")
    f = open(os.path.join(report_path, "Defense.txt"), "r")
    print(f.read())

    # logging report to clerml as artifact
    task.upload_artifact(name='Defense Report', artifact_object=os.path.join(report_path, "Defense.txt"))

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


    zip_extractor(file=os.path.join(report_path, 'Attack_samples.zip'), extract_path=sample_data)
    # upload attack sample to clearml as artifact
    task.upload_artifact('attack sample', artifact_object=sample_data)

    # load attack samples
    def load_data(path):
        files_path = get_file_path(path)
        x = []
        for file in files_path:
            img = cv2.imread(file, 0)
            x.append(img)
        x = np.expand_dims(np.array(x), axis=-1)
        return x


    x_attack = load_data(path=sample_data)

    plot(x=x_attack, row=5)

    zip_extractor(file=os.path.join(report_path, 'Defense_artifact.zip'), extract_path=defense_artifact)
    # upload attack sample to clearml as artifact
    task.upload_artifact('defense artifact', artifact_object=defense_artifact)
    # load defense model
    defense_model_path = os.path.join(defense_artifact,'defense_model.h5')
    defense_model = tf.keras.models.load_model(defense_model_path)

    # load predict.py
    """
    Description: Use defense model
    """
    from reports.defense_artifact import predict

    defense = predict.AISDefenseModel(defense_model)
    """
    Description: Pass sample data to get prediction
    """
    label, prob = defense.predict(X_val[:5])
    print("original data:\nlabel: {} \nprob: {}".format(label, prob))

    label, prob = defense.predict(x_attack[:5])
    print("attack data:\nabel: {} \nprob: {}".format(label, prob))

