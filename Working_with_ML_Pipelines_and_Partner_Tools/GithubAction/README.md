<a name="getting-started"></a>

<div align="left">
    <img src="images/AIShield_logo.png"
         alt="Image of AIShield logo"/>
</div>

-----------------------    Github Action - AIShield MLOPs Workflow  -------------------------

About GitHub Actions: allow you to automate your workflows, making it faster to build, test, and deploy your code.

    When an event occurs (such as a push or a pull request) it automatically triggers the workflow, a workflow consisting of one or more jobs will be triggered.
    Jobs are independent of each Each job is a set of steps that runs inside its own virtual machine runner or inside a container.
    Steps are dependent on each other and are executed in order.
 

Let's jump into workflow execution steps:

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




   
   

