import os

ATK_DEFAULT_EXPLANATION_METHOD = int(os.environ.get('DEFAULT_EXPLANATION_METHOD', 1))
ATK_DEFAULT_DIMENSION = int(os.environ.get('DEFAULT_VS_DIMENSION', 2))
ATK_DEFAULT_PROJECTION = 'PaCMAP'

ATK_INIT_FIG_WIDTH = int(os.environ.get('INIT_FIG_WIDTH', 1800))
ATK_MAX_DOTS = int(os.environ.get('MAX_DOTS', 5000))

# Rule format
ATK_USE_INTERVALS_FOR_RULES = os.environ.get('USE_INTERVALS_FOR_RULES', 'True') == 'True'
ATK_MAX_RULES_DESCR_LENGTH = int(os.environ.get('MAX_RULES_DESCR_LENGTH', 200))

ATK_SHOW_LOG_MODULE_WIDGET = os.environ.get('SHOW_LOG_MODULE_WIDGET', 'False') == 'True'

# Auto cluster
ATK_MIN_POINTS_NUMBER = 100

ATK_SEND_LOG =  not (os.environ.get('ATK_SEND_LOG', 'True') == '0')
