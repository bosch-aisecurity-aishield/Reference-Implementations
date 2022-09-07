# import atexit
import datetime
import os
from typing import List

import app
import numpy as np
import  pickle
import keras
import tensorflow
import numpy as np

# from flask.globals import current_app
from utils import MessageException


def initialize_logger():
    # Initialize session
    n_attemps = 3
    dt = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    while n_attemps > 0:
        # Initialize logger
        app.whylabs_logger = app.whylabs_session.logger(
            dataset_name=os.environ["WHYLABS_DEFAULT_DATASET_ID"],
            dataset_timestamp=dt ,
            with_rotation_time= os.environ["ROTATION_TIME"]  # WARNING:whylogs.app.logger:dataset_timestamp specified with log rotation! dataset_timestamp ignored
        )
        if app.whylabs_logger is not None:
            break
        else:
            n_attemps -= 1
    if n_attemps <= 0:
        raise MessageException("Logger could not be initialized.", 500)


def get_prediction_iris(data: List[float]) -> str:
    # Convert into nd-array
    try:
        data = np.array(data).reshape(1, -1)
        pred = app.model.predict(data)[0]
    except Exception as e:
        raise MessageException("Model could not be loaded.", 500, str(e))
    return pred



def prenet_iris(data, path = "/app/api/prenet_iris" ):
    try:
        data = np.array(data).reshape(1, -1)
        loaded_model = pickle.load(open(path, 'rb'))
        y =  loaded_model.predict(data)[0]
    except Exception as e:
        raise MessageException("Defense Model could not be loaded.", 500, str(e))
    return y




def get_prediction_mnist(data: List[float]) -> int:
    # Convert into nd-array
    try:
        data = np.array(data).reshape(1,28,28,1)
        pred = int(np.argmax(app.model.predict(data),-1)[0])
    except Exception as e:
        raise MessageException("Model could not be loaded.", 500, str(e))
    return pred




def prenet_mnist(data, path = "/app/api/prenet_mnist.h5" ):
    try:
        data = np.array(data).reshape(1,28,28,1)
        loaded_model = keras.models.load_model(path)
        y =  loaded_model.predict(data)[0]
        if y > 0.5:
            y = 1
        else:
            y = 0
    except Exception as e:
        raise MessageException("Defense Model could not be loaded.", 500, str(e))
    return y
