# Source
# https://medium.com/hugo-ferreiras-blog/predicting-wind-and-solar-generation-from-weather-data-using-machine-learning-998d7db8415e


import os

# import necessary modules
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score

# %matplotlib inline

# Renewable energy production data
# read the csv file containing the renewable energy production data relative to Germany.
production = pd.read_csv(os.path.join("datasets", "time_series_60min_singleindex.csv"),
                         usecols=(lambda s: s.startswith('utc') | s.startswith('DE')),
                         parse_dates=[0], index_col=0)


# Get the data relative to 2016
production = production.loc[production.index.year == 2016, :]
print(production)

production_wind_solar = production[['DE_wind_generation_actual', 'DE_solar_generation_actual']]
print(production_wind_solar)

# Weather data
weather = pd.read_csv(os.path.join("datasets", "weather_data_GER_2016.csv"),
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


# -------------

pv_dataset_path = os.path.join("datasets", 'ninja_pv_47.2846_-1.5167_corrected.csv')
df_pv = pd.read_csv(pv_dataset_path, skiprows=3, index_col=0, parse_dates=True, usecols=["local_time", "electricity", "irradiance_direct", "irradiance_diffuse", "temperature"])
df_pv = df_pv.rename(columns={"electricity": "generation_pv"})

wind_dataset_path = os.path.join("datasets", 'ninja_wind_47.2846_-1.5167_corrected.csv')
df_wind = pd.read_csv(wind_dataset_path, skiprows=3, index_col=0, parse_dates=True, usecols=["local_time", "electricity", "wind_speed"])
df_wind = df_wind.rename(columns={"electricity": "generation_wind"})

weather_dataset_path = os.path.join("datasets", "ninja_weather_47.2846_-1.5167_uncorrected.csv")
df_weather = pd.read_csv(weather_dataset_path, skiprows=3, index_col=0, parse_dates=True, usecols=["local_time", "precipitation", "snowfall", "snow_mass", "air_density", "radiation_surface", "radiation_toa", "cloud_cover"])

# merge production_wind_solar and weather_by_day DataFrames
combined = pd.merge(df_pv, df_wind, how='right', left_index=True, right_index=True)
print(combined.info())

combined = pd.merge(combined, df_weather, how='right', left_index=True, right_index=True)
print(combined.info())

# instantiate LinearRegression
lr_ninja = LinearRegression()

# wind generation
X_wind = combined[['wind_speed']]
y_wind = combined['generation_wind']
scores_wind = cross_val_score(lr_ninja, X_wind, y_wind, cv=5, scoring=make_scorer(r2_score))
print(scores_wind, "\n Wind average =", np.mean(scores_wind))

# solar generation
X_solar = combined[['irradiance_direct', 'irradiance_diffuse']]
y_solar = combined['generation_pv']
scores_solar = cross_val_score(lr_ninja, X_solar, y_solar, cv=5, scoring=make_scorer(r2_score))
print(scores_solar, "\n Solar average =", np.mean(scores_solar))
