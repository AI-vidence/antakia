from typing import List

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split

from antakia.compute.model_subtitution.model_class import MLModel

from joblib import Parallel, delayed

from antakia.compute.model_subtitution.regression_models import LinearRegression, LassoRegression, RidgeRegression, GaM, \
    EBM, DecisionTreeRegressor, AvgBaselineModel
import re


def pretty_model_name(model_name):
    return model_name.replace('_', ' ').title()


def reduce_name(model_name):
    parts = re.split('\W+', model_name)
    name = ''
    for part in parts:
        name += part[0].upper()
        for char in part[1:]:
            if char.isupper():
                name += char
    if len(name) == 1:
        return model_name[:2].capitalize()
    return name


class InterpretableModels:
    available_scores = {
        'MSE': mean_squared_error,
        'MAE': mean_absolute_error,
        'R2': r2_score,
        'ACC': accuracy_score,
        'F1': f1_score,
        'precision'.upper(): precision_score,
        'recall'.upper(): recall_score,
    }
    customer_model_name = pretty_model_name('original_model')

    def __init__(self, custom_score):
        if callable(custom_score):
            self.custom_score_str = custom_score.__name__.upper()
            self.custom_score = custom_score
        else:
            self.custom_score_str = custom_score.upper()
            self.custom_score = self.available_scores[custom_score.upper()]

        self.models = {}
        self.scores = {}
        self.perfs = pd.DataFrame()
        self.selected_model = None

    def _get_available_models(self, task_type) -> List[type[MLModel]]:
        if task_type == 'regression':
            return [LinearRegression, LassoRegression, RidgeRegression, GaM,
                    EBM, DecisionTreeRegressor, AvgBaselineModel]
        return [AvgBaselineModel]

    def _init_models(self, task_type):
        for model_class in self._get_available_models(task_type):
            model = model_class()
            if model.name not in self.models:
                self.models[model.name] = model

    def _init_scores(self, task_type):
        if task_type == 'regression':
            scores_list = ['MSE', 'MAE', 'R2']
        else:
            scores_list = ['ACC', 'F1', 'precision'.upper(), 'recall'.upper(), 'R2']
        self.scores = {
            score: self.available_scores[score] for score in scores_list
        }
        self.scores[self.custom_score_str] = self.custom_score

    def _train_models(self, X, y):
        Parallel(n_jobs=1)(delayed(model.fit)(X, y) for model_name, model in self.models.items() if not model.fitted)

    def get_models_performance(
            self,
            customer_model,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame | None,
            y_test: pd.Series | None,
            task_type='regression') -> pd.DataFrame:
        if len(X_train) <= 50 or len(X_train.T) >= len(X_train):
            return pd.DataFrame()
        if X_test is None or len(X_test) == 0:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
        self._init_scores(task_type)
        self._init_models(task_type)
        if customer_model is not None:
            self.models[self.customer_model_name] = MLModel(customer_model, self.customer_model_name, True)
        self._train_models(X_train, y_train)
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            for score_name, score in self.scores.items():
                self.perfs.loc[pretty_model_name(model_name), score_name] = score(y_test, y_pred)

        self.perfs['delta'] = self.perfs[self.custom_score_str] - self.perfs.loc[
            self.customer_model_name, self.custom_score_str]
        get_delta_color = lambda x: 'red' if x > 0.01 else 'green' if x < -0.01 else 'orange'
        self.perfs['delta_color'] = self.perfs['delta'].apply(get_delta_color)
        return self.perfs.sort_values(self.custom_score_str, ascending=True)

    def select_model(self, model_name):
        self.selected_model = model_name

    def selected_model_str(self) -> str:
        perf = self.perfs.loc[self.selected_model]
        reduced_name = reduce_name(self.selected_model)
        display_str = f'{reduced_name} - {self.custom_score_str}:{perf[self.custom_score_str]:.2f} ({perf["delta"]:.2f})'
        return display_str


if __name__ == '__main__':
    X = pd.read_csv('../../../examples/X.csv').set_index('Unnamed: 0')
    y = pd.read_csv('../../../examples/y.csv').set_index('Unnamed: 0')
    InterpretableModels(mean_squared_error).get_models_performance(None, X, y.iloc[:, 0], None, None)
