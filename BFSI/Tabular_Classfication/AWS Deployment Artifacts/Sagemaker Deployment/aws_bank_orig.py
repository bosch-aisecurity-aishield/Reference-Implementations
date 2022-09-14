"""
File: aws_bank_orig
Date: 9/14/2022
Author: Amlan Jyoti
Description: Description: Basic script used to deploy a Scikit-learn model to AWS Sagemaker.
"""

import argparse
import numpy as np
import os
import pandas as pd
import joblib
import pickle
from sklearn.linear_model import LogisticRegression

# Dictionary to convert indices to labels
INDEX_TO_LABEL = {
    0: 'Iris-virginica',
    1: 'Iris-versicolor',
    2: 'Iris-setosa'
}

"""
model_fn
    model_dir: (sting) specifies location of saved model

This function is used by AWS Sagemaker to load the model for deployment.
It does this by simply loading the model that was saved at the end of the
__main__ training block above and returning it to be used by the predict_fn
function below.
"""
def model_fn(model_dir):
    model = pickle.load(open(os.path.join(model_dir, "Banking_SVM_rbf_base.pkl"),"rb"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are
only going to accept text/csv and raise an error for all other formats.
"""
def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        samples = []
        for r in request_body.split('|'):
            samples.append(list(map(float,r.split(','))))
        return np.array(samples)
    else:
        raise ValueError("Thie model only supports text/csv input")

"""
predict_fn
    input_data: (numpy array) returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above

This function is used by AWS Sagemaker to make the prediction on the data
formatted by the input_fn above using the trained model.
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned

This function reformats the predictions returned from predict_fn to the final
format that will be returned as the API call response.

Note: While we don't use content_type in this example, oftentimes you will use
that argument to handle different expected return types.
"""
# def output_fn(prediction, content_type):
#     return '|'.join([INDEX_TO_LABEL[t] for t in prediction])
