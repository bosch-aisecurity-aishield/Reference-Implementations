import os
import json
import random
import datetime
import numpy as np
import time
import urllib.request as urllib
import requests
import pandas as pd
from joblib import dump
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Download Iris dataset and save it as csv
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# raw_data = urllib.urlopen(url)
# try:
#     os.mkdir("dataset/")
# except Exception as e:
#     print(" 'dataset' directory already existed. Moving forward")
# # Save data as csv
# with open('dataset/Iris.csv', 'wb') as file:
#     file.write(raw_data.read())


# reading the data
data = pd.read_csv('dataset/Iris.csv', header=None)

# Separating the independent variables from dependent variables
X = data.iloc[:, 0:4].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# # Train a classifier
# print("Train started.")
# model = SVC()
# model.fit(x_train, y_train)
# print("Train finished.")
# # Save the model
# dump(model, 'model.joblib')
# print("Model saved as model.joblib")




labels = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"] 

# attack data
attack_dataset = np.random.rand(1000,4)
for i in range(x_train.shape[1]):
    tmp = x_train[:,i]
    attack_dataset[:,i] = np.random.rand(1000)*(tmp.max()-tmp.min()) + tmp.min()
        

url = "http://127.0.0.1:5000/api/v1"

healthy = requests.get(url+"/health")
if healthy.ok:
    # original data
    print("Sending 10 original vectors :")
    # r = np.random.randint(0,)
    for k in range(10):       
        # Build a payload with random values
        payload = dict(zip(labels, x_test[k]))  # labels are keys and data is values
        response = requests.post(f"{url}/predict", json=payload)
        print(response)
        if response.ok:
            print(response.json())
            time.sleep(5)

    print("Delay of 10s")
    time.sleep(10)

    # attack data
    print("Sending 10 attack vectors : ")
    for k in range(10):       
        # Build a payload with random values
        payload = dict(zip(labels, attack_dataset[k]))
        response = requests.post(f"{url}/predict", json=payload)
        print(response)
        if response.ok:
            print(response.json())
            time.sleep(5)