from tflite_runtime.interpreter import Interpreter
import numpy as np
import time as t
import time 
import json
from os import access, environ, path
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT

def mnist(arr):
    # Path to Model Directory, Test Data & Labels
    MODEL_DIR = path.expandvars(environ.get("SMEM_IC_MODEL_DIR"))


    # Path to Certificates required to make MQTT Connection
    ENDPOINT = "a3a9xe9433meru-ats.iot.ap-northeast-1.amazonaws.com"
    CLIENT_ID = "testDevice"
    PATH_TO_CERTIFICATE = path.expandvars(environ.get("DEFAULT_CERTIFICATE_PATH")) 
    PATH_TO_PRIVATE_KEY = path.expandvars(environ.get("DEFAULT_PRIVATE_KEY_PATH")) 
    PATH_TO_AMAZON_ROOT_CA_1 = path.expandvars(environ.get("DEFAULT_ROOT_CA_PATH")) 

    PATH_TO_CERTIFICATE = PATH_TO_CERTIFICATE + "/thingCert.crt"
    PATH_TO_PRIVATE_KEY = PATH_TO_PRIVATE_KEY + "/privKey.key"
    PATH_TO_AMAZON_ROOT_CA_1 = PATH_TO_AMAZON_ROOT_CA_1 + "/rootCA.pem"

    TOPIC = "aishield/application"

    # Connecting to the ENDPOINT
    myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(CLIENT_ID)
    myAWSIoTMQTTClient.configureEndpoint(ENDPOINT, 8883)
    myAWSIoTMQTTClient.configureCredentials(PATH_TO_AMAZON_ROOT_CA_1, PATH_TO_PRIVATE_KEY, PATH_TO_CERTIFICATE)
    print("Publishing")
    myAWSIoTMQTTClient.connect()
    print('Begin Publish')

    #Model Path
    model_path = MODEL_DIR + "/application.tflite"
    #Loading input data
    arr = arr.reshape(-1,28,28,1)
    #invoking Model
    interpreter = Interpreter(model_path)
    print("Model Loaded successfully")
    interpreter.allocate_tensors()

    output_index = interpreter.get_output_details()[0]['index']
    input_index = interpreter.get_input_details()[0]['index']
    
    z = interpreter.set_tensor(input_index, arr)
    time1 = time.time()
    interpreter.invoke()
    y = interpreter.get_tensor(output_index)
    y = np.argmax(y, axis=1)
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    message = {"Model:": "Application Model",
        "Inference Time (sec): ": classification_time,
        "Predicted Label: ": str(y)}
    
    myAWSIoTMQTTClient.publish(TOPIC, json.dumps(message), 1)
    print("Published: '" + json.dumps(message) + "'to the topic: " + "'aishield/application'")
    t.sleep(0.1)
    print("Inference Time = ", classification_time, "seconds")
    

    myAWSIoTMQTTClient.disconnect()
    
if __name__ == '__main__':
    mnist()
