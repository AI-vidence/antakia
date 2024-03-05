import numpy as np
import pandas as pd

from antakia.antakia import AntakIA
from antakia.utils.dummy_datasets import load_dataset
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from antakia import config
from tests.test_antakia import run_antakia
from tests.utils_fct import DummyModel

config.ATK_MIN_POINTS_NUMBER = 10
config.ATK_MAX_DOTS = 100

X, y = load_dataset('Corner', 1000, random_seed=42)
X = pd.DataFrame(X, columns=['X1', 'X2'])
X['X3'] = np.random.random(len(X))
y = pd.Series(y)

X_test, y_test = load_dataset('Corner', 100, random_seed=56)
X_test = pd.DataFrame(X_test, columns=['X1', 'X2'])
X_test['X3'] = np.random.random(len(X_test))
y_test = pd.Series(y_test)

regression_DT = DecisionTreeRegressor().fit(X, y)
regression_DT_np = DecisionTreeRegressor().fit(X.values, y.values)
regression_any = DummyModel()
classifier_DT = DecisionTreeClassifier().fit(X, y)
x_exp = pd.concat(
    [(X.iloc[:, 0] > 0.5) * 0.5, (X.iloc[:, 1] > 0.5) * 0.5, (X.iloc[:, 2] > 2) * 1], axis=1)

atk = AntakIA(X, y, regression_DT)
run_antakia(atk, True)