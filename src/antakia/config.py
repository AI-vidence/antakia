import os

DEFAULT_EXPLANATION_METHOD = int(os.environ.get('DEFAULT_EXPLANATION_METHOD', 1))
DEFAULT_DIMENSION = int(os.environ.get('DEFAULT_VS_DIMENSION', 2))
DEFAULT_PROJECTION = 'PaCMAP'

INIT_FIG_WIDTH = int(os.environ.get('INIT_FIG_WIDTH', 1800))
MAX_DOTS = int(os.environ.get('MAX_DOTS', 5000))

# Rule format
USE_INTERVALS_FOR_RULES = os.environ.get('USE_INTERVALS_FOR_RULES', 'True') == 'True'
MAX_RULES_DESCR_LENGTH = int(os.environ.get('MAX_RULES_DESCR_LENGTH', 200))

SHOW_LOG_MODULE_WIDGET = os.environ.get('SHOW_LOG_MODULE_WIDGET', 'False') == 'True'

#Auto cluster
MIN_POINTS_NUMBER = 100
