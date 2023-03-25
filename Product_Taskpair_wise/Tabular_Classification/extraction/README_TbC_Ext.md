# XGBoost Tabular Classification Model with AIShield Integration

<p align="center"> <img src="https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/images/AIShield_logo.png" alt="AIShield Logo"> </p>

This repository contains a reference implementation of an XGBoost model for tabular classification, focused on banking marketing campaign data. It also integrates the AIShield package for model vulnerability analysis and defense generation.

## Dataset

The dataset used in this implementation can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

## Dependencies

Before running the Jupyter notebook, make sure you have the following dependencies installed:

You can install these dependencies by running the following pip commands:
```
! pip install xgboost == 1.6.2
! pip install pandas == 1.5.1
! pip install scikit-learn == 1.1.3
! pip install numpy == 1.23.4
! pip install pyminizip
! pip install requests == 2.28.0
! pip install humanfriendly == 9.2
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

Before running the notebook, you will need to obtain your Org ID and API Key from the AIShield Developer Portal and welcome email, respectively.

1.  Log in to the [AIShield Developer Portal](https://portal.aws.boschaishield.com/) using your credentials.
2.  Under the **API** tab, copy the URL under **Endpoint URL**.
3.  Under the **My Dashboard** tab, copy the **Subscription Key**.
4.  In the welcome email you received after signing up for AIShield, copy the **Org_Id**.

## Running the notebook
This repository contains two reference implementations for the XGBoost model with AIShield integration. The first implementation uses the AIShield Python package to perform vulnerability analysis and defense generation. The second implementation calls the underlying AIShield APIs directly to perform these tasks.

To run the Jupyter notebook, first make sure you have installed all the required dependencies and obtained your AIShield credentials. Then, open the notebook in Jupyter and follow the instructions provided in the notebook to train and evaluate the model.

You can also click the "Open in Colab" button below to open the notebook directly in Google Colab:

Reference Implementation using AIShield PyPi package:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/Tabular_Classification/extraction/PyPi_Extraction_Reference_Implementation_BFSI.ipynb)

Reference Implementation using AIShield APIs:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/Tabular_Classification/extraction/PyPi_Extraction_Reference_Implementation_BFSI.ipynb)

## Contact

If you have any questions or issues with the implementation or AIShield package, please contact AIShield at [AIShield.Contact@bosch.com](mailto:AIShield.Contact@bosch.com).

## License

This implementation is licensed under the [MIT License](https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/LICENSE).
