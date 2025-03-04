# TimeSeries Forecasting Model with AIShield Integration

<p align="center">
  <img src="https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/images/AIShield_logo.png" alt="AIShield Logo">
</p>

This repository contains a reference implementation of an XGBoost Regressor / LSTM model for time series forecasting. The implementation focuses on the House Energy Consumption dataset (Sci-kit learn framework) and the Pump Sensor dataset (TensorFlow model). It also integrates the AIShield package for model vulnerability analysis and defense generation.

**Note:** To directly open the notebook in Google Colab, click the "Open in Colab" button below. Please ensure that you have an active subscription to obtain the necessary credentials (API keys, org ID).

## Reference Implementations

- **Machine Learning (ML) Model Extraction:** This implementation demonstrates extraction attacks on House Energy Consumption dataset(XGBoost Regressor).
  - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/TimeSeries_Forecasting/Timeseries_Forecasting_Extraction_ML.ipynb)

- **Deep Learning (DL) Model Extraction:** This implementation demonstrates extraction attacks on Pump Sensor dataset(LSTM).
  - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bosch-aisecurity-aishield/Reference-Implementations/blob/main/Product_Taskpair_wise/TimeSeries_Forecasting/Timeseries_Forecasting_Extraction_DL.ipynb)

## Dataset

The implementation utilizes the following datasets:

- DL Framework:
	- [Pump Sensor](https://www.kaggle.com/code/winternguyen/water-pump-maintenance-shutdown-prediction/input?select=sensor.csv)
- ML Framework:
	- [House Energy Consumption](https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost/input?select=PJME_hourly.csv)

## Obtaining AIShield Credentials

To utilize AIShield's capabilities, you need to obtain AIShield URL, org ID, and API Key. Follow these steps to obtain the credentials:

1. If you are using AWS cloud, subscribe to AIShield by clicking [here](https://aws.amazon.com/marketplace/pp/prodview-ppbwtiryaohti).
2. If you are not using AWS cloud, subscribe to AIShield by clicking [here](https://boschaishield.com/trial-request).
3. Once you have access to AIShield, retrieve your API keys and org ID by following the instructions provided in your welcome email:
   - After Subscription, you will receive a **welcome email with your Organization ID**, which is required to set up AIShield for vulnerability analysis and reports.
   - Copy the **Org_ID** from the welcome email you received upon signing up and Generate API Key.
   - For **generating an API key**, refer to [https://docs.boschaishield.com/api-docs].

## Running the Notebook

This repository provides reference implementations of the XGBoost Regressor / LSTM model with AIShield integration, along with HTML versions of the notebooks for quick access and overview.

To run the Jupyter notebook, using the obtained AIShield credentials, open the notebook in Jupyter and follow the instructions provided to train, evaluate, and perform vulnerability analysis on your model.

## Contact

If you have any questions or issues regarding the implementation or AIShield package, please contact AIShield at [AIShield.Contact@bosch.com](mailto:AIShield.Contact@bosch.com). For more information, check out [https://docs.boschaishield.com/].

## License

This implementation is licensed under the [MIT License](https://github.com/bosch-aisecurity-aishield/Reference-Implementations/blob/main/LICENSE).