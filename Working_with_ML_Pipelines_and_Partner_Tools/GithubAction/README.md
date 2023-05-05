<a name="getting-started"></a>

<div align="left">
    <img src="images/AIShield_logo.png"
         alt="Image of AIShield logo"/>
</div>

-----------------------    Github Action - AIShield MLOPs Workflow  -------------------------

About GitHub Actions: allow you to automate your workflows, making it faster to build, test, and deploy your code.

    When an event occurs (such as a push or a pull request) it automatically triggers the workflow, a workflow consisting of one or more jobs will be triggered.
    Jobs are independent of each job is a set of steps that runs inside its own virtual machine runner or inside a container.
    Steps are dependent on each other and are executed in order.
 
 How can we intergrate AIShield: In order to integrate AIShield into your MLOps pipeline using GitHub Actions, follow these steps.
1.  First, ensure that your project repository is set up on GitHub, with your machine learning model code, test dataset, and requirements.txt file containing all   necessary packages.
2.	Create a “.github” folder in the root directory of your repository (if it doesn't already exist), and inside it, create another folder called “workflows”.
3.	Inside the workflows folder, create a new YAML file named action.yaml(file name can be anything). This file will define the GitHub Actions workflow for your project.
4.	If you want to keep rest of the files in the same root directory, please commented out 17th line in yaml file. if not no changes are required, Please follow same folder structure for the below files.
   I.	requirement.txt
   II.	extraction_reference_implementation_mnist.py
5.	Next you must update the file: “extraction_reference_implementation_mnist.py” with api-key, org-id and baseurl details below
      baseurl = "xxxxxxxx" 
     'x-api-key': "xxxxxxxx"
     'Org-Id' : "xxxxxx" 
6.	Commit and push your changes to the branch. The GitHub Actions workflow will be triggered automatically. You can view the progress and results of the workflow by going to the "Actions" tab in your GitHub repository.


Github Action Workflow execution steps: When action triggers the workflow, you will see the below flow. 

Step-1:  It perform below actions.
    Setup OS (in our case, ubuntu)
    Get the github read access for contents, metadata, and package.
    Prepare workflow directory.
    Prepared all required actions.

Step-2:  Checkout code
    run actions/checkout:
    Initialize the repository authentication.
    Checkout our repository so that the workflow can access files from our repository.

Step-3:  Setup the environment
    run actions/setup-python:
    setup a python environment for our workflow

Setup-4: Setup cml
    run iterative/setupcml
    Set up the Continuous Machine Learning (CML) with the latest version for workflow.

Step-5: ML workflow goes here.
    requirement.txt file gets executed, and it make sure that all the dependencies are installed for workflow.
    Build & train the module.
    monitor_link get generated (enter this link in the browser and see the output.)
    Finally, you can see the results.

With these steps, you have successfully integrated AIShield into an MLOps CI/CD pipeline using GitHub Actions. This will help you evaluate and maintain the robustness of your machine learning model against adversarial attacks throughout the development process.


   
<a name="license"></a>
# License

```
See LICENSE File for details. 
```

<a name="i-want-to-know-more"></a>
# I want to know more!

Please reach us at aishield.contact@bosch.com


<a name="want-to-contribute"></a>
# Want to Contribute?

This is an open source project, and we'd love to see your contributions!
Please git clone this project and send us a pull request. Thanks.




   
   

