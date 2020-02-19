# Source
# https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e


# import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# %matplotlib inline

# Renewable energy production data
# read the csv file containing the renewable energy production data relative to Germany.
production = pd.read_csv("data/time_series_60min_singleindex.csv",
                         usecols=(lambda s: s.startswith('utc') | s.startswith('DE')),
                         parse_dates=[0], index_col=0)
# Get the data relative to 2016
production = production.loc[production.index.year == 2016, :]
# print(production.info())

production_wind_solar = production[['DE_wind_generation_actual', 'DE_solar_generation_actual']]

# Weather data
weather = pd.read_csv("data/weather_data_GER_2016.csv",
                      parse_dates=[0], index_col=0)
weather_bis = weather[
    ['DE_windspeed_10m', 'DE_radiation_direct_horizontal', 'DE_radiation_diffuse_horizontal', 'DE_temperature']]
# print(weather_bis.loc[weather.index == '2016-01-01 00:00:00', :])
# print(weather_bis.info())

# merge production_wind_solar and weather_by_day DataFrames
combined = pd.merge(production_wind_solar, weather_bis, how='right', left_index=True, right_index=True)

# print(production_wind_solar.info())
# print(weather_bis.info())
print(combined.info())

# instantiate LinearRegression
lr = LinearRegression()

# wind generation
X_wind = combined[['DE_windspeed_10m']]
y_wind = combined['DE_wind_generation_actual']
scores_wind = cross_val_score(lr, X_wind, y_wind, cv=5)
print(scores_wind, "\naverage =", np.mean(scores_wind))

# solar generation
X_solar = combined[['DE_radiation_direct_horizontal', 'DE_radiation_diffuse_horizontal', 'DE_temperature']]
y_solar = combined['DE_solar_generation_actual']
scores_solar = cross_val_score(lr, X_solar, y_solar, cv=5)
print(scores_solar, "\naverage =", np.mean(scores_solar))
