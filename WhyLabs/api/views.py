""" View functions for every endpoint. """
import app
import numpy as np
from api.utils import get_prediction_iris, get_prediction_mnist, initialize_logger , prenet_iris , prenet_mnist
from flask import Blueprint, request
from flask_pydantic import validate
from schemas import FeatureVector_IRIS , FeatureVector_MNIST
from utils import object_response

blueprint = Blueprint("api", __name__, url_prefix="/api/v1")
initialize_logger()


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

    # log prenet output
    app.whylabs_logger.log({"attack": def_out})

    # Log predicted class
    app.whylabs_logger.log({"class": pred})

    return object_response({"class": pred,"attack": def_out}, 200)
