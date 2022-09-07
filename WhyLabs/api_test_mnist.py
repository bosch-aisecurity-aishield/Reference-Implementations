import urllib.request as urllib
import requests
import time


import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from keras.layers import *
from keras.models import *
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras import backend as K

# batch_size = 128
# num_classes = 10
# epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# # convert class vectors to binary class matrices
# y_train = to_categorical(y_train, num_classes)
# y_test = to_categorical(y_test, num_classes)



# # model
# input_shape = (28,28,1)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25, name = 'layer1'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu',name = 'layer2'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer="Adam" ,
#               metrics=['accuracy'])


# history = model.fit(X_train, y_train,
#           batch_size, epochs=3,
#           verbose=1,
#           validation_data=(X_test, y_test))


# model.save("model_mnist.h5")


# attack vectors
queries = np.random.rand(60000,28,28,1)
for i in range(queries.shape[0]):
    for _ in range(np.random.randint(0,15)):
        x1 = np.random.randint(0,20)
        y1 = np.random.randint(0,20)
        x2 = np.random.randint(x1,28)
        y2 = np.random.randint(y1,28)
        queries[i,x1:x2,y1:y2,0] = 1


url = "http://127.0.0.1:5000/api/v1"

healthy = requests.get(url+"/health")
if healthy.ok:
    # original data
    print("Sending 20 original vectors :")
    for k in range(20):
        # Build a payload with random values
        payload = {}
        payload["features"] = X_test[k].reshape(28*28).tolist()
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
        payload = {}
        payload["features"] = queries[k].reshape(28*28).tolist()
        response = requests.post(f"{url}/predict", json=payload)
        print(response)
        if response.ok:
            print(response.json())
            time.sleep(5)