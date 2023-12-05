from typing import List

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split

from antakia.compute.model_subtitution.model_class import MLModel

from joblib import Parallel, delayed

from antakia.compute.model_subtitution.regression_models import LinearRegression, LassoRegression, RidgeRegression, GaM, \
    EBM, DecisionTreeRegressor


class InterpretableModels:
    def __init__(self, custom_score):
        self.custom_score = custom_score.upper()
        self.models = {}
        self.scores = {}
        self.perfs = pd.DataFrame()

    def _get_available_models(self, task_type) -> List[type[MLModel]]:
        if task_type == 'regression':
            return [LinearRegression, LassoRegression, RidgeRegression, GaM,
                    EBM, DecisionTreeRegressor]
        return []

    def _init_models(self, task_type):
        for model_class in self._get_available_models(task_type):
            model = model_class()
            if model.name not in self.models:
                self.models[model.name] = model

    def _init_scores(self, task_type):
        if task_type == 'regression':
            self.scores = {
                'MSE': mean_squared_error,
                'MAE': mean_absolute_error,
                'R2': r2_score
            }
        else:
            self.scores = {
                'ACC': accuracy_score,
                'F1': f1_score,
                'precision'.upper(): precision_score,
                'recall'.upper(): recall_score,
                'R2': r2_score
            }
        if callable(self.custom_score):
            self.scores[self.custom_score.__name__.upper()] = self.custom_score

    def _train_models(self, X, y):
        Parallel(n_jobs=1)(delayed(model.fit)(X, y) for model_name, model in self.models.items() if not model.fitted)

    def get_models_performance(self, customer_model, X: pd.DataFrame, y: pd.Series, task_type='regression'):
        if len(X) <= 50 or len(X.T) >= len(X):
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self._init_scores(task_type)
        self._init_models(task_type)
        if customer_model is not None:
            self.models['customer_model'] = MLModel(customer_model, 'customer_model', True)
        self._train_models(X_train, y_train)
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            for score_name, score in self.scores.items():
                self.perfs.loc[model_name, score_name] = score(y_test, y_pred)

        if isinstance(self.custom_score, str):
            return self.perfs.sort_values(self.custom_score, ascending=False)
        else:
            return self.perfs.sort_values(self.custom_score.__name__.upper(), ascending=False)


if __name__ == '__main__':
    X = pd.read_csv('../../../examples/X.csv').set_index('Unnamed: 0')
    y = pd.read_csv('../../../examples/y.csv').set_index('Unnamed: 0')
    InterpretableModels().get_models_performance(None, X, y.iloc[:, 0])
