import os
import time
import tensorflow as tf
import aishield as ais
from shutil import make_archive

#field converter
def str_to_tuple(s):
    return tuple(map(int, s.split(',')))
def str_to_bool(s):
    return s.lower() == "true"
 
# Load your trained model
model = tf.keras.models.load_model('/path/to/your/trained/model')

# Save the model to disk as a zip file
model_path = "/path/to/model/folder"
os.makedirs(model_path, exist_ok=True)
model.save(os.path.join(model_path, "mnist_cnn"))
make_archive(base_name=os.path.join(model_path, "mnist_cnn"), root_dir=model_path, zip_format="zip")

# Configuration for AIShield API
baseurl = os.environ["AISHIELD_API_URL"]
api_key = os.environ["AISHIELD_API_KEY"]
org_id = os.environ["AISHIELD_ORG_ID"]
url = baseurl + "/api/ais/v1.5"

#AISHield API Parameters
task_type_val = os.environ["TASK_TYPE"]
analysis_type_val = os.environ["ANALYSIS_TYPE"]
input_shape_val = os.environ["INPUT_SHAPE"]
num_classes_val= int(os.environ["NUM_CLASSES"])
attack_type_val = os.environ["ATTACK_TYPE"]
attack_queries_val = int(os.environ["ATTACK_QUERIES"])
encryption_strategy_val = int(os.environ["ENCRYPTION_STRATEGY"])
defense_generate_val = os.environ["DEFENSE_GENERATE"]


# Initialize the AIShield API
client = ais.AIShieldApi(api_url=url, api_key=api_key, org_id=org_id)

# Define the task and analysis type
task_type = ais.get_type("task", task_type_val)
analysis_type = ais.get_type("analysis", analysis_type_val)

# Register model and upload the input artifacts
status, job_details = client.register_model(task_type=task_type, analysis_type=analysis_type)
model_id = job_details.model_id

data_path = "/path/to/data.zip"
label_path = "/path/to/label.zip"
model_path = os.path.join(model_path, "mnist_cnn.zip")

upload_status = client.upload_input_artifacts(
    job_details=job_details,
    data_path=data_path,
    label_path=label_path,
    model_path=model_path,
)
print('Upload status: {}'.format(', '.join(upload_status)))

# Vulnerability analysis configuration
input_shape = str_to_tuple(input_shape_val)
num_classes = numclasses_val

vuln_config = ais.VulnConfig(task_type=task_type,
                             analysis_type=analysis_type,
                             defense_generate=str_to_bool(defense_generate_val))

vuln_config.input_dimensions = input_shape
vuln_config.number_of_classes = num_classes
vuln_config.attack_type = attack_type_val
vuln_config.attack_queries = ttack_queries_val
vuln_config.encryption_strategy = encryption_strategy_val

# Run vulnerability analysis
job_status, job_details = client.vuln_analysis(model_id, vuln_config)
job_id = job_details.job_id

# Monitor progress for the given Job ID
print("Job URL: {}".format(client.get_job_url(job_id)))

# Periodically check the job status
job_status = client.job_status(job_id)
while job_status.state != "success":
    print("Job status: {}".format(job_status.state))
    time.sleep(10)
    job_status = client.get_job_status(job_id)

# Download and save reports and artifacts
