import pandas as pd

from antakia.compute.model_subtitution.model_class import MLModel, LinearMLModel, AvgBaseline
from sklearn import linear_model
from sklearn import tree
import pygam
from interpret.glassbox import ExplainableBoostingRegressor


####################################
# Baseline Models                  #
####################################

class AvgBaselineModel(MLModel):
    def __init__(self):
        super().__init__(AvgBaseline(), 'average baseline')


####################################
# Linear Models                    #
####################################
class LinearRegression(LinearMLModel):
    def __init__(self):
        super().__init__(linear_model.LinearRegression(), 'linear regression')


class LassoRegression(LinearMLModel):
    def __init__(self):
        super().__init__(linear_model.LassoCV(), 'Lasso regression')


class RidgeRegression(LinearMLModel):
    def __init__(self):
        super().__init__(linear_model.RidgeCV(), 'Ridge regression')


####################################
# DecisionTree Models              #
####################################

class DecisionTreeRegressor(MLModel):
    def __init__(self):
        super().__init__(tree.DecisionTreeRegressor(), 'Decision Tree')


####################################
# GAM Models                       #
####################################

class GaM(MLModel):
    def __init__(self):
        super().__init__(pygam.pygam.GAM(distribution='normal', link='identity'), 'linear GAM')


####################################
# GAM Models                       #
####################################
class EBM(MLModel):
    def __init__(self):
        super().__init__(ExplainableBoostingRegressor(), 'Explainable Boosting Tree')
