from antakia.compute.model_subtitution.model_class import MLModel
from sklearn import linear_model
from sklearn import tree
import pygam
from interpret.glassbox import ExplainableBoostingRegressor

####################################
# Linear Models                    #
####################################
class LinearRegression(MLModel):
    def __init__(self):
        super().__init__(linear_model.LinearRegression(), 'linear regression')


class LassoRegression(MLModel):
    def __init__(self):
        super().__init__(linear_model.LassoCV(), 'Lasso regression')


class RidgeRegression(MLModel):
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
        super().__init__(ExplainableBoostingRegressor(),'Explainable Boosting Tree')
