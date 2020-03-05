# Source
# https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e

import os

from utils.data import *

if __name__ == '__main__':
    pv_dataset_path = os.path.join("datasets", 'ninja_pv_47.2846_-1.5167_corrected.csv')
    wind_dataset_path = os.path.join("datasets", 'ninja_wind_47.2846_-1.5167_corrected.csv')

    dataset = init_dataset(pv_dataset_path=pv_dataset_path, wind_dataset_path=wind_dataset_path)

    # model_wind_LR, model_pv_LR = fit_linear_regression(dataset=dataset)
    # model_wind_GB, model_pv_GB = fit_gradient_boosting_regression(dataset=dataset)
    fit_linear_regression(dataset=dataset)
    fit_gradient_boosting_regression(dataset=dataset)

    predict_model(os.path.join('models', 'model_wind_lr.joblib'), os.path.join('models', 'model_pv_lr.joblib'))

    predict_model(os.path.join('models', 'model_wind_gb.joblib'), os.path.join('models', 'model_pv_gb.joblib'))
