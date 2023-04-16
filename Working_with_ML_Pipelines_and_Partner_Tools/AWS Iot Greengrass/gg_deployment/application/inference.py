import sys
import traceback
import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
import numpy as np
import os
from application import mnist
from awsiot.greengrasscoreipc.model import (
    SubscribeToTopicRequest,
    SubscriptionResponseMessage
)


ipc_client = awsiot.greengrasscoreipc.connect()

class StreamHandler(client.SubscribeToTopicStreamHandler):
    def __init__(self):
        super().__init__()

    def on_stream_event(self, event: SubscriptionResponseMessage) -> None:
        message_string = event.binary_message.message
        arr = np.frombuffer(message_string, dtype=np.float32).reshape(-1,28,28,1)
        y = mnist(arr)
        return arr
    
    def on_stream_error(self, error: Exception) -> bool:
        print("Received a stream error.", file=sys.stderr)
        traceback.print_exc()
        return True

    def on_stream_closed(self) -> None:
        print('Subscribe to topic stream closed.')
        pass
    

topic = "aishield"

request = SubscribeToTopicRequest()
request.topic = topic
handler = StreamHandler()
operation = ipc_client.new_subscribe_to_topic(handler)
future = operation.activate(request)

while True:
    pass