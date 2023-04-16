# AIShield Integration with AWS IoT GreenGrass

<p align="center"> <img src="https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/images/AIShield_logo.png" alt="AIShield Logo"> </p>

This repository contains a reference implementation of a MNIST model for image classification, focused on handwritten digit data. It integrates the AIShield package for model vulnerability analysis and defense generation, alongwith flow to deploy it to the Edge Device with the help of AWS IoT GreenGrass.

## Architectural Overview
<p align="center"> <img src="" alt="Architectural Overview"> </p>

## Dependencies
Before running the Jupyter notebook, make sure you have the following dependencies installed:

You can install these dependencies by running the following pip commands:
```
!pip install numpy==1.22
!pip install matplotlib==3.3.4
!pip install tensorflow==2.9.1
!pip install scikit-learn==1.0.2
!pip install humanfriendly==9.2
!pip install tqdm==4.61.1
!pip install requests==2.28.0
!pip install opencv-python
!pip install aishield
!pip install boto3
```

## Folder Structure
### 1. vulnerability_assessment: It contains a `ipynb` file in which an AI model will be trained on MNIST Handwritten Dataset. Following the training process, the file provides instructions for performing Vulnerability Analysis and generating a Defense Model using the AIShield API.

 -  Outcome: - A PDF report containing the results of the vulnerability assessment will be created, along with a folder that includes two tflite model files. These     models will be subsequently deployed on an Edge Device.

### 2. gg_deployment: This folder includes a `ipynb` file that outlines steps for creating an S3 bucket, zipping the application model, datafeeder and defense model artifacts, uploading the artifacts to a folder within the bucket, creating GreenGrass Components on AWS IoT, and deploying the components to an Edge device.
- Sub Folder:
    * application : The files contained in this folder aid in performing inference on the Edge Device and sending telemetry to the Cloud.
        -   `application.tflite, application.py, inference.py, privkey.key, rootCA.pem, thingsCert.crt` 
    * datafeeder : The files contained in this folder enable the Edge Device to perform inference on both the application model and defense model concurrently, allowing for the detection of any possible attacks.
        -   `datafeeder.npy, datafeeder.py`
    * defense: The folder contains the defence model and script to detect potential attacks on the edge device and send the telemetry to the Cloud.
        -   `defense.py, defense.tflite, defense_inference.py, privkey.key, rootCA.pem, thingsCert.crt`

## Obtaining AIShield Credentials

In order to use AIShield, you will need to obtain API keys, org ID, and other necessary details. Here are the steps to obtain them:

 -  If you are working in AWS cloud, you can subscribe to the product by clicking on this [link](https://aws.amazon.com/marketplace/pp/prodview-ppbwtiryaohti).
    
 -  If you are not working in AWS cloud , you can subscribe to the product by clicking on this [link](https://boschaishield.com/trial-request) 
    
 -  Once you have access to AIShield, you can obtain your API keys and org ID by following the instructions provided in your welcome email. To obtain the API key and org ID, you can follow these steps:
	 - Log in to the [AIShield Developer
   Portal](https://portal.aws.boschaishield.com/) using your credential
	 - Under the **API** tab, copy the URL under **Endpoint URL**
	 - Under the **My Dashboard** tab, copy the **Subscription Key**
	 - Copy the **Org_Id** from the welcome email you would have received after signing up

## Contact

If you have any questions or issues with the implementation or AIShield package, please contact AIShield at [AIShield.Contact@bosch.com](mailto:AIShield.Contact@bosch.com).

## License

This implementation is licensed under the [MIT License](https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/LICENSE).
