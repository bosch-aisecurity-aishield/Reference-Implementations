Requirements :

1. AWS Subscription - for using SageMaker and its associated services
2. AIShield subscription details - URL and subscription key for for making call to image classification service for vulnerability analysis
3. Training script and its requirements file (for required library installation) is already present inside code folder. The code folder can be copied the current working directory and its path need to be specified accordingly in the reference notebook.

Steps :
1. Create a SageMaker notebook with configuration as - conda_tensorflow_p38. 'CPU optimized' configuration should solve the purpose if its not compute heavy. 
2. Import and open the provided notebook and code folder to the environment.
3. Follow the steps as mentioned in the notebook.
4. Variable/parameter values need to be changed as per run configuration.
5. Note: There are 2 requirements file. One at the notebook level is for installing requiremnts required for the notebook ready; and the requirements file present inside code folder is responsible for the deployment related requirements.