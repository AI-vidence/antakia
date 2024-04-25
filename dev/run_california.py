import os

from tests.test_antakia import run_antakia
from antakia.config import AppConfig

os.environ['SEND_ANONYMOUS_LOGS'] = '0'
AppConfig.ATK_SEND_LOG = False
import pandas as pd
from antakia.utils.examples import fetch_dataset

df = fetch_dataset('california_housing')
df = df.sample(len(df))
limit = int(20000 / 0.8)
df = df.iloc[:limit]
split_row = int(len(df) * 0.8)
df_train = df[:split_row]
df_test = df[split_row:]
X_train = df_train.iloc[:, :8]  # the dataset
y_train = df_train.iloc[:, 9]  # the target variable
shap_values_train = df_train.iloc[:,
                                  [10, 11, 12, 13, 14, 15, 16, 17
                                   ]]  # the SHAP values from a previous model

X_test = df_test.iloc[:, :8]  # the dataset
y_test = df_test.iloc[:, 9]  # the target variable
from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor(random_state=9)
regressor.fit(X_train, y_train)

variables_df = pd.DataFrame(
    {
        'col_index': [0, 1, 2, 3, 4, 5, 6, 7],
        'descr': [
            'Median income', 'House age', 'Average nb rooms',
            'Average nb bedrooms', 'Population', 'Average occupancy',
            'Latitude', 'Longitude'
        ],
        # 'type': ['float64', 'int', 'float64', 'float64', 'int', 'float64', 'float64', 'float64'],
        'unit': [
            'k$', 'years', 'rooms', 'rooms', 'people', 'ratio', 'degrees',
            'degrees'
        ],
        'critical': [True, False, False, False, False, False, False, False],
        'lat': [False, False, False, False, False, False, True, False],
        'lon': [False, False, False, False, False, False, False, True]
    },
    index=[
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
        'AveOccup', 'Latitude', 'Longitude'
    ])
from antakia.antakia import AntakIA

for _ in range(100):
    atk = AntakIA(X_train,
                  y_train,
                  regressor,
                  variables=variables_df,
                  X_test=X_test,
                  y_test=y_test,
                  verbose=0
                  # X_exp=shap_values_train
                  )
# atk.start_gui()
#
# from tests.interactions import *
#
# gui = atk.gui
# select_dim(gui,3)
# select_dim(gui,2)
# select_dim(gui,3)
#
# set_color(gui, 0)
# set_color(gui, 1)
# set_color(gui, 2)
#
# select_points(gui, 1)
# unselect(gui)

# run_antakia(atk, True)
