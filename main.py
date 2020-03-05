# Source
# https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e

import os

from utils import data, parser

if __name__ == '__main__':
    pv_dataset_path = os.path.join("datasets", 'ninja_pv_47.2846_-1.5167_corrected.csv')
    wind_dataset_path = os.path.join("datasets", 'ninja_wind_47.2846_-1.5167_corrected.csv')
    MAX_POWER = 2

    dataset = data.init_dataset(pv_dataset_path=pv_dataset_path, wind_dataset_path=wind_dataset_path)

    model_wind, model_pv = data.fit_linear_regression(dataset=dataset)

    print("-------------")

    daily_data = parser.parse_input(json_path=os.path.join('input', 'input_example.json'))
    predictions = data.daily_predict(model_wind=model_wind, model_solar=model_pv, daily_data=daily_data, max_power=MAX_POWER)
    parser.write_output(predictions=predictions)
