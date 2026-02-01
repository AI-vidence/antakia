import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from antakia.antakia import AntakIA
from antakia.config import AppConfig
from antakia.utils.dummy_datasets import load_dataset
from tests.test_antakia import run_antakia
from tests.utils_fct import DummyModel

AppConfig.ATK_MIN_POINTS_NUMBER = 10
AppConfig.ATK_MAX_DOTS = 100

num_samples = 2000
num_features = 200

X, y = load_dataset("Corner", num_samples, random_seed=42, num_cols=num_features)

X_test, y_test = load_dataset("Corner", 100, random_seed=56, num_cols=num_features)

regression_DT = DecisionTreeRegressor().fit(X, y)
regression_DT_np = DecisionTreeRegressor().fit(X.values, y.values)
regression_any = DummyModel()
classifier_DT = DecisionTreeClassifier().fit(X, y)
x_exp = pd.DataFrame(np.zeros((num_samples, num_features)), columns=X.columns, index=X.index)
x_exp.iloc[:, 0] = (X.iloc[:, 0] > 0.5) * 0.5
x_exp.iloc[:, 1] = (X.iloc[:, 1] > 0.5) * 0.5

atk = AntakIA(X, y, regression_DT)
run_antakia(atk, True)
