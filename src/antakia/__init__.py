"""
AntakIA - Explainability through Tessellation.

Ce package fournit des outils pour :
- Tessellation : découpage de l'espace en régions homogènes
- Explanation : explications LLM des modèles ML
- Interactions : calcul des interactions de Shapley
- Parcellation : méthodes de clustering dyadique
- Monitoring : détection de dérive
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("antakia")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

__author__ = "AI-vidence"


# Lazy imports pour éviter les dépendances circulaires
def __getattr__(name):
    """Lazy import des sous-modules."""
    if name == "AntakIA":
        from antakia.antakia import AntakIA
        return AntakIA
    elif name == "AntakIAAPI":
        from antakia.api import AntakIAAPI
        return AntakIAAPI
    elif name == "quick_tessellate":
        from antakia.api import quick_tessellate
        return quick_tessellate
    elif name == "tessellation":
        from antakia import tessellation

        return tessellation
    elif name == "explanation":
        from antakia import explanation

        return explanation
    elif name == "interactions":
        from antakia import interactions

        return interactions
    elif name == "parcellation":
        from antakia import parcellation

        return parcellation
    elif name == "monitoring":
        from antakia import monitoring

        return monitoring
    elif name == "observability":
        from antakia import observability

        return observability
    elif name == "preprocessing":
        from antakia import preprocessing

        return preprocessing
    raise AttributeError(f"module 'antakia' has no attribute '{name}'")
