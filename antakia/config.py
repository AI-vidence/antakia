from antakia.data import ExplanationMethod, DimReducMethod
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
# from sklearn import ensemble

DEFAULT_EXPLANATION_METHOD = ExplanationMethod.SHAP
DEFAULT_VS_DIMENSION = DEFAULT_ES_DIMENSION = 2
DEFAULT_VS_PROJECTION = DEFAULT_ES_PROJECTION = DimReducMethod.PaCMAP

INIT_FIG_WIDTH = 1800

# Rule format
USE_INTERVALS_FOR_RULES = True

SHOW_LOG_MODULE_WIDGET = True
