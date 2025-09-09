"""
configuration variables
"""
import os

ATK_DEFAULT_EXPLANATION_METHOD = int(
    os.environ.get('DEFAULT_EXPLANATION_METHOD', 2))  # 2 = LIME (plus rapide que SHAP)
ATK_DEFAULT_DIMENSION = int(os.environ.get('DEFAULT_VS_DIMENSION', 2))
ATK_DEFAULT_PROJECTION = os.environ.get('DEFAULT_PROJECTION', 'PCA')  # PCA est beaucoup plus rapide que PaCMAP

ATK_INIT_FIG_WIDTH = int(os.environ.get('INIT_FIG_WIDTH', 1800))
ATK_MAX_DOTS = int(os.environ.get('MAX_DOTS', 5000))

# Optimisations M2 Max
ATK_MEMORY_LIMIT = int(os.environ.get('ATK_MEMORY_LIMIT', 24))  # GB pour M2 Max 32GB
ATK_CHUNK_SIZE = int(os.environ.get('ATK_CHUNK_SIZE', 10000))  # Taille des chunks
ATK_PARALLEL_PROCESSING = os.environ.get('ATK_PARALLEL_PROCESSING', 'False').lower() == 'true'
ATK_NUM_WORKERS = int(os.environ.get('ATK_NUM_WORKERS', 10))  # 10 workers sur 12 cœurs

# Rule format
ATK_USE_INTERVALS_FOR_RULES = os.environ.get('USE_INTERVALS_FOR_RULES',
                                             'True') == 'True'
ATK_MAX_RULES_DESCR_LENGTH = int(os.environ.get('MAX_RULES_DESCR_LENGTH', 200))

ATK_SHOW_LOG_MODULE_WIDGET = os.environ.get('SHOW_LOG_MODULE_WIDGET',
                                            'False') == 'True'

# Auto cluster
ATK_MIN_POINTS_NUMBER = 100

ATK_SEND_LOG = os.environ.get('SEND_ANONYMOUS_LOGS', 'True') != '0'
