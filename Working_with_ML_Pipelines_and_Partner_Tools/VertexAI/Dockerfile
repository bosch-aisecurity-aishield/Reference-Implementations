
FROM tensorflow/tensorflow:2.9.1-gpu
WORKDIR /root

RUN apt-get install python3
RUN apt-get install python3-pip
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev


RUN apt-get update &&     apt-get install -y curl gnupg &&     echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - &&     apt-get update -y &&     apt-get install google-cloud-sdk -y
    
RUN pip3 install opencv-python numpy==1.22 google-cloud-storage==1.35.0 scikit-learn==1.0.2 pandas==1.1.5 protobuf==3.20.*

COPY train.py ./train.py

ENTRYPOINT ["python3", "train.py"]