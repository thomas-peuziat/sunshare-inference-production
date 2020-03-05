# Source
# https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e

import os
import time
from pprint import pprint

from utils import data, parser

if __name__ == '__main__':
    start = time.time()
    pv_dataset_path = os.path.join("datasets", 'ninja_pv_47.2846_-1.5167_corrected.csv')
    wind_dataset_path = os.path.join("datasets", 'ninja_wind_47.2846_-1.5167_corrected.csv')
    MAX_POWER_WIND = 2
    MAX_POWER_SOLAR = 10

    dataset = data.init_dataset(pv_dataset_path=pv_dataset_path, wind_dataset_path=wind_dataset_path)

    # model_wind_lr, model_pv_lr = data.fit_linear_regression(dataset=dataset)
    # model_wind_gb, model_pv_gb = data.fit_gradient_boosting_regression(dataset=dataset)

    # model_wind_lr, model_pv_lr = data.load_model(os.path.join('models', 'model_wind_lr.joblib'),
    #                                              os.path.join('models', 'model_pv_lr.joblib'))
    model_wind_gb, model_pv_gb = data.load_model(os.path.join('models', 'model_wind_gb.joblib'),
                                                 os.path.join('models', 'model_pv_gb.joblib'))

    daily_data = parser.parse_input(json_path=os.path.join('input', 'input_example.json'))
    predictions = data.daily_predict(model_wind=model_wind_gb, model_solar=model_pv_gb, daily_data=daily_data,
                                     max_power_wind=MAX_POWER_WIND, max_power_solar=MAX_POWER_SOLAR)
    parser.write_output(predictions=predictions)
    
    end = time.time()
    
    print("Prévisions météo :")
    pprint(daily_data)
    print("\nPrédictions production :")
    pprint(predictions)
    print("\nTemps de réponse :", end-start, "s")
