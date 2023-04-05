# VertexAI Integration with AIShield for Model Analysis

This repository contains sample code and instructions for integrating Google Cloud's VertexAI with AIShield platform (Powered by Bosch ) for model analysis. Specifically, the code provided here shows how to deploy a machine learning model on Vertex AI and analyze its performance using AI Shield's model analysis tools (API or SDK).

## Requirements

To run the code in this repository, you will need the following:

-   A Google Cloud Platform account with billing enabled.
-   Google Cloud SDK CLI (https://cloud.google.com/sdk/docs/install)
-   Access to VertexAI and AIShield.
-   A trained machine learning model in a format supported by Vertex AI (such as TensorFlow SavedModel)
-   A dataset to evaluate the performance of the model.
-  Unzip Training_Dataset.zip file.

## Usage

Before running the code, you will need to set up authentication for Google Cloud and AIShield. Follow the instructions provided by Google Cloud and  AIShield to authenticate your credentials.

Once you have set up authentication, you can run the code :
`your-gcp-project-id` with your Google Cloud project ID, `us-central1` with your desired region for Vertex AI deployment, `https://your-aishield-url` with the URL for your AIShield platform, and `your-aishield-apikey` with your AIShield api key and  `your-aishield-org_id` with your AIShield OrgId .

The  script  `VertexAI_AIShield_Integration.ipynb` will deploy your machine learning model on Vertex AI and run inference on the test dataset. It will then use AIShield's model analysis API tools to evaluate the performance of the model and generate a report.

The  script  `Pypi_VertexAI_AIShield_Integration.ipynb` will deploy your machine learning model on Vertex AI and run inference on the test dataset. It will then use AIShield's model analysis SDK tools to evaluate the performance of the model and generate a report.

##### The above two Reference implementations of AIShield with vertexAI is done for Covid dataset(chest x-ray dataset).