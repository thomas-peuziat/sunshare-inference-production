# Source
# https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e

import os

from utils.data import *

if __name__ == '__main__':
    pv_dataset_path = os.path.join("datasets", 'ninja_pv_47.2846_-1.5167_corrected.csv')
    wind_dataset_path = os.path.join("datasets", 'ninja_wind_47.2846_-1.5167_corrected.csv')

    dataset = init_dataset(pv_dataset_path=pv_dataset_path, wind_dataset_path=wind_dataset_path)

    model_wind, model_pv = fit_linear_regression(dataset=dataset)

    print("-------------")

    # wind_speed : m/s
    # electricity : kW; max : 1kW
    wind_data = [[8]]
    wind_prediction = model_wind.predict(wind_data)

    print("Wind data :", wind_data, "; Prediction :", wind_prediction)

    # irradiance_direct : kW/m²
    # irradiance_diffuse : kW/m²
    # electricity : kW; max : 1kW
    pv_data = [[0.4, 0.3]]
    pv_prediction = model_pv.predict(pv_data)

    print("PV data :", pv_data, "; Prediction :", pv_prediction)
