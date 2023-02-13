Requirments : 
1. Conda or a python environment  
2. Docker : Install from https://docs.docker.com/desktop/windows/install/

Steps : (https://docs.whylabs.ai/docs)
0. Took the Iris code from github 
1. Signup to the the whylabs website , note the API key and edit the .env file with respective API key and org ID .API key is valid for one year 
2. Create conda environment using "conda env create -f environment.yml" . This will create env and install all the packaghes inside environemt.yml file . But I created manually
 using "conda create -n whylabs python = 3.7.13 numpy == 1.21.5" and installed the necessaruy packages based on requiement . 
3.(Go out of VPN)Build the Dockerfile without vpn and proxy off using "docker build -f Dockerfile --build-arg PYTHON_VERSION=3.7 -t whylabs-flask ." This will create a image/container with name whylabs-flask . Every time u need to do build .
 which had all the packges present in requirement.txt . 
4. Run the dcker using "docker run --rm -p 5000:5000 whylabs-flask"
5. We can see the whylogs once we hit the api endpoint . It can be done through swagger Ui or python post request
6. We can see the logs in "https://hub.whylabsapp.com/models" . 
7. First create model ID is whylabs website and then enter that ID in .env file
8. Made changes to views.py , added the product defense prenet over there and also added the logging changes to whylabs(about MNIST)



Baseline for MNIST : 
0 to 9 : original model output 
if pred is non int :
    generate alert in whylabs 


