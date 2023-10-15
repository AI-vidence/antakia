from antakia.data import ExplanationMethod, DimReducMethod
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import ensemble

DEFAULT_EXPLANATION_METHOD = ExplanationMethod.SHAP
DEFAULT_VS_DIMENSION = DEFAULT_ES_DIMENSION = 2
DEFAULT_VS_PROJECTION = DEFAULT_ES_PROJECTION = DimReducMethod.PaCMAP


# Convention :
VS = 0
ES = 1

def get_default_submodels():
    """
    Returns a list of surrogates models
    """
    return [linear_model.LinearRegression(), RandomForestRegressor(random_state=9), ensemble.GradientBoostingRegressor(random_state=9)]

