
# Automatic Speech Recognition Assessment with AIShield Integration

![AIShield Logo](https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/images/AIShield_logo.png)

This repository contains reference implementations of an Whisper model for Automatic speech recognition, specifically focusing on audio signals. The implementations cover various analysis types, including evasion. The repository also integrates the AIShield package for model vulnerability analysis and defense generation. This reference implementation focusing on uploading whisper model in onnx format  then how we can do the vulnerability analysis.


**Note:** To directly open the notebook in Google Colab, click the "Open in Colab" button provided below. Please ensure that you have an active subscription to obtain the necessary credentials (API keys, org ID).

## Reference Implementations

- **Evasion:** This implementation demonstrates the evasion assessment on Whisper ASR model
  - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/Audio_Speech_Recognition/audio_speech_recognition_evasion.ipynb)
 
- **Poisoning:** This implementation demonstrates the poisoning assessment on ASR Dataset
  - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/Audio_Speech_Recognition/audio_speech_recognition_poisoning.ipynb)

## Dataset

The implementation utilizes the Audio dataset.

## Obtaining AIShield Credentials

To utilize AIShield's capabilities, you need to obtain AIShield URL, org ID, and API Key. Follow these steps to obtain the credentials:

1. If you are using AWS cloud, subscribe to AIShield by clicking [here](https://aws.amazon.com/marketplace/pp/prodview-ppbwtiryaohti).
2. If you are not using AWS cloud, subscribe to AIShield by clicking [here](https://boschaishield.com/trial-request).
3. Once you have access to AIShield, retrieve your API keys and org ID by following the instructions provided in your welcome email:
   - After Subscription, you will receive a **welcome email with your Organization ID**, which is required to set up AIShield for vulnerability analysis and reports.
   - Copy the **Org_ID** from the welcome email you received upon signing up and Generate API Key.
   - For **generating an API key**, refer to [https://docs.boschaishield.com/api-docs].

## Running the Notebook

This repository provides reference implementations of the whisper model with AIShield integration, along with HTML versions of the notebooks for quick access and overview.

To run the Jupyter notebook, using the obtained AIShield credentials, open the notebook in Jupyter and follow the instructions provided to train, evaluate, and perform vulnerability analysis on your model.

## Contact

If you have any questions or issues regarding the implementation or AIShield package, please contact AIShield at [AIShield.Contact@bosch.com](mailto:AIShield.Contact@bosch.com). For more information, check out [https://docs.boschaishield.com/].

## License

This implementation is licensed under the [MIT License](https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/LICENSE).
