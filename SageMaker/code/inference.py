# inference.py
# from tensorflow import keras
import numpy as np
import requests
import base64
import json
import os
import ast

IMG_HEIGHT = 128
IMG_WIDTH = 128


def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    decoded_data = data.read().decode('utf-8')
    decoded_data = json.loads(json.loads(decoded_data))
    print('decoded_data ', type(decoded_data))
    image_array = np.array(decoded_data['instances'])
    image_array = image_array.reshape((-1, 28, 28, 1))
    print('image_array shape: ', image_array.shape)
    processed_input = json.dumps({"instances": image_array.tolist()})
    print('model call')
    # processed_input = _process_input(decoded_data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _process_output(response, context)


# def _process_input(decoded_data, context):
#     # if context.request_content_type == 'application/json':
#     # decoded_data = data.read().decode('utf-8')
#     # Converts the JSON object to an array
#     # that meets the model signature.
#     # image_array = image_to_array(decoded_data)
#     print('decoded_data: ', type(decoded_data['instances']))
#     image_array = np.array(decoded_data['instances'])
#     print('image_array type: ', type(image_array), image_array.shape)
#     image_array = image_array.reshape((-1, 28, 28, 1))
#     print('image_array shape: ', image_array.shape)
#     return json.dumps({"instances": image_array.tolist()})

#     raise ValueError('{{"error": "unsupported content type {}"}}'.format(
#         context.request_content_type or "unknown"))


def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = json.loads(data.content.decode('utf-8'))
    # segmented_image = mask_to_image(prediction["outputs"])
    predicted_class = np.argmax(prediction['predictions'], axis=1)
    print('predicted_image ', predicted_class)
    return json.dumps({"output": predicted_class.tolist()}), response_content_type