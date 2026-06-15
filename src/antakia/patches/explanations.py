"""
Patches pour le calcul des explications SHAP/LIME.

Ajoute du logging pour les fallbacks et améliore la robustesse.

Compatible SHAP 0.43+ et SHAP 0.50+ (nouvelle API Explanation).
"""

import logging
import time

import pandas as pd
import shap
from antakia_core.utils.splittable_callback import ProgressCallback
from antakia_core.utils.utils import ProblemCategory

logger = logging.getLogger(__name__)

# Détecter la version de SHAP
SHAP_VERSION = tuple(map(int, shap.__version__.split(".")[:2]))
SHAP_050_PLUS = SHAP_VERSION >= (0, 50)


def compute_explanations_with_logging(
    X: pd.DataFrame,
    model,
    explanation_method: int,
    task_type: ProblemCategory,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """
    Calcule les explications avec logging détaillé.

    Amélioration de antakia_core.explanation.compute_explanations:
    - Logging du type d'explainer utilisé
    - Logging des fallbacks
    - Timing détaillé

    Parameters
    ----------
    X : pd.DataFrame
        Données d'entrée
    model : object
        Modèle à expliquer
    explanation_method : int
        0 = Importé, 1 = SHAP, 2 = LIME
    task_type : ProblemCategory
        Type de tâche (régression, classification)
    progress_callback : ProgressCallback, optional
        Callback de progression

    Returns
    -------
    pd.DataFrame
        Valeurs d'explication
    """
    from antakia_core.explanation import ExplanationMethod

    if explanation_method == ExplanationMethod.SHAP:
        return _compute_shap_with_logging(X, model, task_type, progress_callback)
    elif explanation_method == ExplanationMethod.LIME:
        # LIME n'a pas de fallback, utiliser l'implémentation standard
        from antakia_core.explanation.explanations import LIMExplanation

        return LIMExplanation(X, model, task_type, progress_callback).compute()
    else:
        raise ValueError(f"Méthode d'explication invalide: {explanation_method}")


def _compute_shap_with_logging(
    X: pd.DataFrame,
    model,
    task_type: ProblemCategory,
    progress_callback: ProgressCallback | None = None,
    use_new_api: bool | None = None,
) -> pd.DataFrame:
    """
    Calcule les SHAP values avec logging détaillé.

    Priorité d'explainer:
    1. TreeExplainer (rapide, pour modèles d'arbres)
    2. LinearExplainer (pour modèles linéaires)
    3. KernelExplainer (universel mais lent)

    Parameters
    ----------
    X : pd.DataFrame
        Données d'entrée
    model : object
        Modèle à expliquer
    task_type : ProblemCategory
        Type de tâche
    progress_callback : ProgressCallback, optional
        Callback de progression
    use_new_api : bool, optional
        Utiliser la nouvelle API SHAP 0.50+ (explainer(X)).
        Si None, utilise la nouvelle API si SHAP >= 0.50.
    """
    start_time = time.time()
    n_samples = len(X)
    explainer = None
    explainer_type = None

    # Détecter quelle API utiliser
    if use_new_api is None:
        use_new_api = SHAP_050_PLUS

    logger.info(
        f"SHAP: version={shap.__version__}, "
        f"API={'nouvelle (explainer())' if use_new_api else 'ancienne (shap_values())'}"
    )

    # 1. Essayer TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        explainer_type = "TreeExplainer"
        logger.info(f"SHAP: Utilisation de TreeExplainer " f"(modèle: {type(model).__name__})")
    except Exception as e:
        logger.warning(f"SHAP: TreeExplainer non disponible pour {type(model).__name__}: {e}")

    # 2. Essayer LinearExplainer
    if explainer is None:
        try:
            explainer = shap.LinearExplainer(model, X.sample(min(200, n_samples)))
            explainer_type = "LinearExplainer"
            logger.info(
                f"SHAP: Utilisation de LinearExplainer " f"(modèle: {type(model).__name__})"
            )
        except Exception as e:
            logger.debug(f"SHAP: LinearExplainer non disponible: {e}")

    # 3. Fallback sur KernelExplainer
    if explainer is None:
        background_size = min(100, n_samples)
        background = shap.sample(X, background_size)

        # Déterminer la fonction de prédiction
        if task_type == ProblemCategory.regression:
            predict_fn = model.predict
            link = "identity"
        else:
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            link = "logit"

        explainer = shap.KernelExplainer(predict_fn, background, link=link)
        explainer_type = "KernelExplainer"

        logger.warning(
            f"SHAP: Fallback sur KernelExplainer "
            f"(modèle: {type(model).__name__}). "
            f"Le calcul sera LENT pour {n_samples} points."
        )

    # Calcul par chunks avec progression
    chunk_size = max(200, n_samples // 100)
    shap_values_list = []

    if progress_callback:
        progress_callback(0)

    for i in range(0, n_samples, chunk_size):
        chunk = X.iloc[i : i + chunk_size]

        # Calcul SHAP pour ce chunk
        if use_new_api and explainer_type != "KernelExplainer":
            # Nouvelle API SHAP 0.50+ : explainer(X) retourne un Explanation
            explanation = explainer(chunk)
            chunk_shap = explanation.values
        else:
            # Ancienne API : explainer.shap_values(X)
            chunk_shap = explainer.shap_values(chunk)

        # Gérer le cas de la classification (liste de arrays)
        if isinstance(chunk_shap, list):
            # Utiliser la classe positive (dernière)
            chunk_shap = chunk_shap[-1]

        shap_values_list.append(pd.DataFrame(chunk_shap, columns=X.columns, index=chunk.index))

        if progress_callback:
            progress = min(100, int(100 * (i + chunk_size) / n_samples))
            progress_callback(progress)

    # Concaténation
    shap_values = pd.concat(shap_values_list)

    elapsed = time.time() - start_time
    logger.info(
        f"SHAP calculées: {explainer_type}, "
        f"{n_samples} points en {elapsed:.1f}s "
        f"({n_samples/elapsed:.0f} pts/s)"
    )

    if progress_callback:
        progress_callback(100)

    return shap_values


def check_model_shap_compatibility(model) -> dict:
    """
    Vérifie la compatibilité d'un modèle avec les différents explainers SHAP.

    Parameters
    ----------
    model : object
        Modèle à vérifier

    Returns
    -------
    dict
        Dictionnaire avec les explainers compatibles et leur priorité

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> model = GradientBoostingRegressor()
    >>> check_model_shap_compatibility(model)
    {'TreeExplainer': True, 'LinearExplainer': False, 'KernelExplainer': True}
    """
    result = {
        "TreeExplainer": False,
        "LinearExplainer": False,
        "KernelExplainer": True,  # Toujours compatible (universel)
        "recommended": None,
    }

    # Vérifier TreeExplainer
    tree_models = (
        "GradientBoosting",
        "RandomForest",
        "ExtraTrees",
        "XGB",
        "LGBM",
        "CatBoost",
        "DecisionTree",
    )
    model_name = type(model).__name__
    if any(name in model_name for name in tree_models):
        result["TreeExplainer"] = True
        result["recommended"] = "TreeExplainer"

    # Vérifier LinearExplainer
    linear_models = ("LinearRegression", "LogisticRegression", "Ridge", "Lasso")
    if any(name in model_name for name in linear_models):
        result["LinearExplainer"] = True
        if result["recommended"] is None:
            result["recommended"] = "LinearExplainer"

    # Fallback
    if result["recommended"] is None:
        result["recommended"] = "KernelExplainer"

    return result
