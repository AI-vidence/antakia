"""
Patches pour la réduction dimensionnelle.

Corrige le bug PaCMAP qui crash avec des petits datasets
quand n_neighbors > n_samples - 1.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_safe_n_neighbors(n_samples: int, requested_n_neighbors: int = 10) -> int:
    """
    Calcule un n_neighbors sûr en fonction de la taille du dataset.

    PaCMAP crash si n_neighbors + 50 >= n_samples (bug dans generate_pair).
    Cette fonction retourne une valeur safe.

    Parameters
    ----------
    n_samples : int
        Nombre de points dans le dataset
    requested_n_neighbors : int
        Nombre de voisins demandé (défaut: 10)

    Returns
    -------
    int
        Nombre de voisins sûr (garanti de ne pas crasher)

    Examples
    --------
    >>> get_safe_n_neighbors(100, 10)
    10
    >>> get_safe_n_neighbors(50, 10)  # Réduit automatiquement
    5
    >>> get_safe_n_neighbors(10, 10)  # Très petit dataset
    2
    """
    # PaCMAP utilise n_neighbors_extra = min(n_neighbors + 50, n - 1)
    # Mais il faut aussi que n_neighbors_extra >= 1
    # Et que le dataset ait assez de points pour les paires MN et FP

    # Règle de sécurité : n_neighbors <= (n_samples - 1) / 6
    # Car PaCMAP a besoin de : n_neighbors, n_MN (0.5x), n_FP (2x)
    max_safe = max(2, (n_samples - 1) // 6)

    safe_n_neighbors = min(requested_n_neighbors, max_safe)

    if safe_n_neighbors < requested_n_neighbors:
        logger.warning(
            f"n_neighbors réduit de {requested_n_neighbors} à {safe_n_neighbors} "
            f"pour un dataset de {n_samples} points (évite crash PaCMAP)"
        )

    return safe_n_neighbors


def safe_compute_projection(
    X: pd.DataFrame,
    method: str,
    n_components: int = 2,
    n_neighbors: int = 10,
    progress_callback: Callable | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Calcule une projection de manière sûre.

    Adapte automatiquement les paramètres pour éviter les crashes
    avec des petits datasets.

    Parameters
    ----------
    X : pd.DataFrame
        Données à projeter
    method : str
        Méthode de projection ('PCA', 'UMAP', 'PaCMAP')
    n_components : int
        Dimensions de sortie (2 ou 3)
    n_neighbors : int
        Nombre de voisins (pour UMAP/PaCMAP)
    progress_callback : Callable, optional
        Callback de progression
    **kwargs
        Paramètres additionnels pour la méthode

    Returns
    -------
    pd.DataFrame
        Données projetées

    Raises
    ------
    ValueError
        Si le dataset est trop petit pour la projection
    """
    n_samples = len(X)

    # Validation minimale
    if n_samples < 3:
        raise ValueError(
            f"Dataset trop petit pour la projection: {n_samples} points " f"(minimum: 3)"
        )

    # Adaptation des paramètres selon la méthode
    if method.upper() == "PACMAP":
        # PaCMAP est le plus sensible aux petits datasets
        safe_neighbors = get_safe_n_neighbors(n_samples, n_neighbors)

        if n_samples < 50:
            logger.warning(
                f"Dataset très petit ({n_samples} points). "
                f"Utilisation de PCA au lieu de PaCMAP pour éviter les crashes."
            )
            method = "PCA"
        else:
            kwargs["n_neighbors"] = safe_neighbors

    elif method.upper() == "UMAP":
        # UMAP est plus robuste mais peut aussi avoir des problèmes
        safe_neighbors = min(n_neighbors, max(2, n_samples - 1))
        if safe_neighbors < n_neighbors:
            logger.warning(f"n_neighbors UMAP réduit de {n_neighbors} à {safe_neighbors}")
        kwargs["n_neighbors"] = safe_neighbors

    # Import de la fonction de projection
    from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
    from antakia_core.compute.dim_reduction.dim_reduction import compute_projection

    method_int = DimReducMethod.dimreduc_method_as_int(method)

    try:
        return compute_projection(X, method_int, n_components, progress_callback, **kwargs)
    except ValueError as e:
        if "broadcast" in str(e) or "shape" in str(e):
            # Bug PaCMAP connu, fallback sur PCA
            logger.error(f"Erreur de projection {method}: {e}. " f"Fallback sur PCA.")
            pca_method = DimReducMethod.dimreduc_method_as_int("PCA")
            return compute_projection(X, pca_method, n_components, progress_callback)
        raise


def patch_pacmap_for_small_datasets():
    """
    Applique un monkey-patch sur PaCMAP pour supporter les petits datasets.

    Note: Cette fonction modifie le comportement global de PaCMAP.
    À utiliser avec précaution.
    """
    try:
        from antakia_core.compute.dim_reduction.pacmap_progress import pacmap_progress

        original_decide_num_pairs = pacmap_progress.PaCMAP.decide_num_pairs

        def patched_decide_num_pairs(self, n):
            """Version patchée qui valide n_neighbors."""
            if self.n_neighbors is None:
                if n <= 10000:
                    self.n_neighbors = 10
                else:
                    self.n_neighbors = int(round(10 + 15 * (np.log10(n) - 4)))

            # PATCH: Limiter n_neighbors pour les petits datasets
            max_safe = max(2, (n - 1) // 6)
            if self.n_neighbors > max_safe:
                logger.warning(
                    f"PaCMAP: n_neighbors réduit de {self.n_neighbors} à {max_safe} "
                    f"(dataset de {n} points)"
                )
                self.n_neighbors = max_safe

            self.n_MN = int(round(self.n_neighbors * self.MN_ratio))
            self.n_FP = int(round(self.n_neighbors * self.FP_ratio))

            if self.n_neighbors < 1:
                raise ValueError("n_neighbors doit être >= 1")
            if self.n_FP < 1:
                raise ValueError("n_FP doit être >= 1")

        pacmap_progress.PaCMAP.decide_num_pairs = patched_decide_num_pairs
        logger.info("Patch PaCMAP appliqué avec succès")

    except Exception as e:
        logger.warning(f"Impossible d'appliquer le patch PaCMAP: {e}")
