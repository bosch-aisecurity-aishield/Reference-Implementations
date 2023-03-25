
# MNIST Image Classification Model with AIShield Integration

<p align="center"> <img src="https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/images/AIShield_logo.png" alt="AIShield Logo"> </p>

This repository contains a reference implementation of an MNIST model for image classification, focused on handwritten digit data. It also integrates the AIShield package for model vulnerability analysis and defense generation.

*Note* : You can click the "Open in Colab" button below to open the notebook directly in Google Colab. 
You will need a active subscription to obtain credentials (api keys,org_id). In case you don't have a active subscription, Click [AWS](https://aws.amazon.com/marketplace/pp/prodview-ppbwtiryaohti) OR [non-AWS](https://boschaishield.com/trial-request) , to get access.

Reference Implementation using AIShield PyPi package:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/Image_Classification/Extraction/PyPi_Extraction_Reference_Implementation_MNIST.ipynb)

Reference Implementation using AIShield APIs:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/Image_Classification/Extraction/Extraction_Reference_Implementation_MNIST.ipynb)

## Dataset

The dataset used in this implementation is part of TensorFlow Datasets

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
! pip install aishield
```

Alternatively, you can create a virtual environment and install the dependencies using the `requirements.txt` file included in the repository by running:
```
#Create a virtual environment
python -m venv env

#Activate the virtual environment
source env/bin/activate

#Install dependencies
pip install -r requirements.txt
```
Note: If you choose to use the pip install commands, you should uncomment the relevant code in the notebook before running it.


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

## Running the notebook
This repository contains two reference implementations for the MNIST model with AIShield integration. The first implementation uses the AIShield Python package to perform vulnerability analysis and defense generation. The second implementation calls the underlying AIShield APIs directly to perform these tasks.

To run the Jupyter notebook, first make sure you have installed all the required dependencies and obtained your AIShield credentials. Then, open the notebook in Jupyter and follow the instructions provided in the notebook to train and evaluate the model.

## Contact

If you have any questions or issues with the implementation or AIShield package, please contact AIShield at [AIShield.Contact@bosch.com](mailto:AIShield.Contact@bosch.com).

## License

This implementation is licensed under the [MIT License](https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/LICENSE).
