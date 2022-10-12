""" View functions for every endpoint. """
import app
import numpy as np
from api.utils import get_prediction_iris, get_prediction_mnist , prenet_iris , prenet_mnist
from flask import Blueprint, request
from flask_pydantic import validate
from schemas import FeatureVector_IRIS , FeatureVector_MNIST
from utils import object_response

import whylogs as why
import datetime
from whylogs.api.writer.whylabs import WhyLabsWriter
writer = WhyLabsWriter()

blueprint = Blueprint("api", __name__, url_prefix="/api/v1")


@blueprint.route("/health", methods=["GET"])
def health():
    return object_response({"state": "healthy"}, 200)


@blueprint.route("/predict", methods=["POST"])
@validate()
def predict(body : FeatureVector_MNIST):  #body is a dict 
    # Predict the output given the input vector
    # vector = [
    #     body.sepal_length_cm,
    #     body.sepal_width_cm,
    #     body.petal_length_cm,
    #     body.petal_width_cm,
    # ]

    vector = body.features 

    # # passing to prenet
    # def_out = prenet_iris(vector)
    def_out = prenet_mnist(vector)

    # passing to original model
    # pred = get_prediction_iris(vector)
    pred =  get_prediction_mnist(vector)

    # Log to whylabs platform
    # Log input vector as dictionary
    # app.whylabs_logger.log(request.json)

    # Defense Response Startegy : Fixed Replacement Strategy
    # if attack is detected as true then overwrite the original models output with letter A
    if def_out == 1 :
        pred = "A"

    # select timestamp
    current_date = datetime.datetime.utcnow() - datetime.timedelta(days=2)

    # log prenet output    
    profile = why.log({"attack": def_out}).profile()
    profile.set_dataset_timestamp(current_date)
    writer.write(file=profile.view())


    # Log predicted class
    profile = why.log({"class": pred}).profile()
    profile.set_dataset_timestamp(current_date)
    writer.write(file=profile.view())

    return object_response({"class": pred,"attack": def_out}, 200)
