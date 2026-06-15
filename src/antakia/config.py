"""
configuration variables
"""
import os


class AppConfig:
    ATK_DEFAULT_EXPLANATION_METHOD = int(os.environ.get("DEFAULT_EXPLANATION_METHOD", 1))
    ATK_DEFAULT_DIMENSION = int(os.environ.get("DEFAULT_VS_DIMENSION", 2))
    ATK_DEFAULT_PROJECTION = "UMAP"  # PaCMAP a un bug avec certains datasets

    ATK_INIT_FIG_WIDTH = int(os.environ.get("INIT_FIG_WIDTH", 1800))
    ATK_MAX_DOTS = int(os.environ.get("MAX_DOTS", 2000))

    # Rule format
    ATK_USE_INTERVALS_FOR_RULES = os.environ.get("USE_INTERVALS_FOR_RULES", "True") == "True"
    ATK_MAX_RULES_DESCR_LENGTH = int(os.environ.get("MAX_RULES_DESCR_LENGTH", 200))

    ATK_SHOW_LOG_MODULE_WIDGET = os.environ.get("SHOW_LOG_MODULE_WIDGET", "False") == "True"

    # Auto cluster
    ATK_MIN_POINTS_NUMBER = 100

    # RC7 — règles descriptives multi-view (VS + ES conjoint)
    ATK_MULTIVIEW_RULES_DEFAULT = os.environ.get("ATK_MULTIVIEW_RULES", "True") == "True"

    # Substitution tab — Feature Importance (nombreuses variables)
    ATK_FI_TOP_N = int(os.environ.get("ATK_FI_TOP_N", 25))
    ATK_FI_BAR_HEIGHT = int(os.environ.get("ATK_FI_BAR_HEIGHT", 20))
    ATK_FI_MAX_HEIGHT = int(os.environ.get("ATK_FI_MAX_HEIGHT", 900))
    ATK_FI_LABEL_WIDTH = int(os.environ.get("ATK_FI_LABEL_WIDTH", 220))

    ATK_SEND_LOG = os.environ.get("SEND_ANONYMOUS_LOGS", "True") != "0"
    verbose = 3
    log_with_time = True
    dev = False
