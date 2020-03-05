import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import cross_val_score


def init_dataset(pv_dataset_path: str, wind_dataset_path: str):
    # pv_dataset_path = os.path.join("datasets", 'ninja_pv_47.2846_-1.5167_corrected.csv')
    df_pv = pd.read_csv(pv_dataset_path, skiprows=3, index_col=0, parse_dates=True,
                        usecols=["local_time", "electricity", "irradiance_direct", "irradiance_diffuse"])
    df_pv = df_pv.rename(columns={"electricity": "generation_pv"})
    # df_pv.plot.scatter(y='generation_pv',
    #                    x='irradiance_direct',
    #                    c='irradiance_diffuse',
    #                    colormap='viridis')

    # wind_dataset_path = os.path.join("datasets", 'ninja_wind_47.2846_-1.5167_corrected.csv')
    df_wind = pd.read_csv(wind_dataset_path, skiprows=3, index_col=0, parse_dates=True,
                          usecols=["local_time", "electricity", "wind_speed"])
    df_wind = df_wind.rename(columns={"electricity": "generation_wind"})
    # df_wind.plot.scatter(y='generation_wind',
    #                      x='wind_speed')

    dataset = pd.merge(df_pv, df_wind, how='right', left_index=True, right_index=True)

    return dataset


def fit_linear_regression(dataset):
    # wind generation
    model_wind = LinearRegression()
    x_wind = dataset[['wind_speed']]
    y_wind = dataset['generation_wind']
    scores_wind = cross_val_score(model_wind, x_wind, y_wind, cv=5, scoring=make_scorer(r2_score))
    print("Score Wind average =", np.mean(scores_wind))

    model_wind.fit(x_wind, y_wind)

    # plt.scatter(X_wind, Y_wind, color='g')
    # plt.plot(X_wind, lr_wind.predict(X_wind), color='k')
    # plt.show()

    # solar generation
    model_pv = LinearRegression()
    x_solar = dataset[['irradiance_direct', 'irradiance_diffuse']]
    y_solar = dataset['generation_pv']
    scores_solar = cross_val_score(model_pv, x_solar, y_solar, cv=5, scoring=make_scorer(r2_score))
    print("Score Solar average =", np.mean(scores_solar))

    model_pv.fit(x_solar, y_solar)

    # plt.scatter(X_solar["irradiance_direct"], Y_solar, color='g')
    # plt.scatter(X_solar["irradiance_diffuse"], Y_solar, color='r')
    # # plt.plot(X_solar, lr_solar.predict(X_solar), color='b')
    # plt.show()

    return model_wind, model_pv

