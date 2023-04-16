import time
import datetime
import json
import numpy as np
from os import path, environ
import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import (
    PublishToTopicRequest,
    PublishMessage,
    BinaryMessage
)

print("Step 1")
TIMEOUT = 10
publish_rate = 1.0
IMAGE_DIR = path.expandvars(environ.get("DEFAULT_SMEM_IC_IMAGE_DIR"))
ipc_client = awsiot.greengrasscoreipc.connect()
print("Step 1")

data = np.load(IMAGE_DIR + "/datafeeder.npy")
data = data.reshape(-1,1,28,28,1)

for i in range(data.shape[0]):
    data1 = data[i]
    topic = "aishield"
    print("Starting to Publish")
    message = bytes(data1)
    request = PublishToTopicRequest()
    request.topic = topic
    publish_message = PublishMessage()
    publish_message.binary_message = BinaryMessage()
    publish_message.binary_message.message = message
    request.publish_message = publish_message
    operation = ipc_client.new_publish_to_topic()
    operation.activate(request)
    future = operation.get_response()
    future.result(TIMEOUT)
        
    print("publish, Data = {}".format(i))
    time.sleep(1/publish_rate)