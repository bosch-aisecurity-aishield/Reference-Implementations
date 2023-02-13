The Product_Taskpair_wise: Tutorials for vulnerability analysis & defense generation process based on tasks like image classification, tabular classification, etc. The features are available in the released product version & tutorials can be run off-the-shelf.

Further within every task-pairs, tutorials with AIShield is leveraged by 1. direct API call, and 2. using AIShield PyPI package.



* Working_with_ML_Pipelines_and_Partner_Tools: Reference implementations for working with/integration with different partners like(SageMaker, whylabs, etc.) and ML pipelines(MLFlow)

* SageMaker- Tutorial for integrating AWS SageMaker & AIShield. Here model is trained on  mnist data & the call AIShield for model vulnerability analysis & defense generation. Further this also contains instructions for deploying the defense model along with original model & demo showing inference  
Known Issue: This reference implemntation is tested to work with Python 3.8.x notebook kernel. In other versions, library compatibility issues might be observed

* MLFlow- Tutorial for integrating MLFlow & AIShield

* AzureML- Tutorial for integrating AzureML & AIShield. Here model is trained on  mnist data & the call AIShield for model vulnerability analysis & defense generation. Further this also contains instructions for deploying the defense model along with original model & demo showing inference
Known Issue: This tutorial is tested with the older versions of AIShield API. New changes are not currently maintained