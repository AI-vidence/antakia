"""
Patches et correctifs pour antakia_core.

Ce module contient des corrections et adaptations pour les problèmes
identifiés dans antakia_core sans modifier le package directement.
"""

from antakia.patches.dim_reduction import get_safe_n_neighbors, safe_compute_projection
from antakia.patches.explanations import compute_explanations_with_logging

__all__ = [
    "safe_compute_projection",
    "get_safe_n_neighbors",
    "compute_explanations_with_logging",
]
