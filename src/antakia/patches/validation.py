"""
Validation des inputs pour AntakIA.

Vérifie la cohérence des données d'entrée et fournit des messages d'erreur clairs.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Erreur de validation des inputs."""

    pass


def validate_X(X: Any, name: str = "X") -> pd.DataFrame:
    """
    Valide et convertit X en DataFrame.

    Parameters
    ----------
    X : array-like
        Données d'entrée
    name : str
        Nom pour les messages d'erreur

    Returns
    -------
    pd.DataFrame
        X validé et converti

    Raises
    ------
    ValidationError
        Si X n'est pas valide
    """
    # Conversion si nécessaire
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
        logger.debug(f"{name}: converti de ndarray en DataFrame")

    if not isinstance(X, pd.DataFrame):
        raise ValidationError(
            f"{name} doit être un DataFrame ou ndarray, " f"reçu: {type(X).__name__}"
        )

    # Vérifications
    if X.empty:
        raise ValidationError(f"{name} est vide (0 lignes)")

    if X.shape[1] == 0:
        raise ValidationError(f"{name} n'a pas de colonnes")

    # Vérifier les NaN
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        nan_pct = 100 * nan_count / X.size
        logger.warning(f"{name} contient {nan_count} valeurs NaN ({nan_pct:.1f}%)")

    # Vérifier les colonnes dupliquées
    if X.columns.duplicated().any():
        dups = X.columns[X.columns.duplicated()].tolist()
        raise ValidationError(f"{name} a des colonnes dupliquées: {dups}")

    return X


def validate_y(y: Any, X: pd.DataFrame, name: str = "y") -> pd.Series:
    """
    Valide et convertit y en Series.

    Parameters
    ----------
    y : array-like
        Target
    X : pd.DataFrame
        X associé (pour vérifier l'alignement)
    name : str
        Nom pour les messages d'erreur

    Returns
    -------
    pd.Series
        y validé et converti
    """
    # Conversion
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
        logger.debug(f"{name}: converti de ndarray en Series")

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValidationError(
                f"{name} est un DataFrame avec {y.shape[1]} colonnes, "
                f"attendu 1 colonne ou une Series"
            )
        y = y.iloc[:, 0]

    if not isinstance(y, pd.Series):
        raise ValidationError(
            f"{name} doit être une Series ou ndarray, " f"reçu: {type(y).__name__}"
        )

    # Vérifier l'alignement avec X
    if len(y) != len(X):
        raise ValidationError(f"{name} a {len(y)} éléments, X a {len(X)} lignes")

    return y


def validate_X_exp(X_exp: Any | None, X: pd.DataFrame, name: str = "X_exp") -> pd.DataFrame | None:
    """
    Valide les valeurs d'explication (SHAP, LIME).

    Parameters
    ----------
    X_exp : array-like or None
        Valeurs d'explication
    X : pd.DataFrame
        X associé
    name : str
        Nom pour les messages d'erreur

    Returns
    -------
    pd.DataFrame or None
        X_exp validé, ou None si non fourni

    Raises
    ------
    ValidationError
        Si X_exp n'est pas cohérent avec X
    """
    if X_exp is None:
        logger.info(f"{name} non fourni, sera calculé depuis le GUI")
        return None

    # Conversion
    X_exp = validate_X(X_exp, name)

    # Vérifier la forme
    if X_exp.shape != X.shape:
        raise ValidationError(
            f"{name} a la forme {X_exp.shape}, "
            f"X a la forme {X.shape}. "
            f"Les deux doivent avoir la même forme."
        )

    # Vérifier les colonnes (ordre et noms)
    if not X_exp.columns.equals(X.columns):
        # Essayer de réordonner
        missing = set(X.columns) - set(X_exp.columns)
        extra = set(X_exp.columns) - set(X.columns)

        if missing or extra:
            raise ValidationError(
                f"{name} a des colonnes différentes de X. " f"Manquantes: {missing}, Extra: {extra}"
            )

        # Réordonner
        logger.warning(f"{name}: colonnes réordonnées pour correspondre à X")
        X_exp = X_exp[X.columns]

    # Vérifier l'index
    if not X_exp.index.equals(X.index):
        logger.warning(f"{name}: index différent de X, réindexation effectuée")
        # Vérifier que les index sont compatibles
        if set(X_exp.index) != set(X.index):
            raise ValidationError(
                f"{name} a des index différents de X. "
                f"Utilisez le même index ou laissez X_exp=None."
            )
        X_exp = X_exp.reindex(X.index)

    # Vérifier les valeurs (SHAP doivent sommer à y_pred - E[y_pred])
    # On fait juste un check que les valeurs ne sont pas aberrantes
    max_abs = X_exp.abs().max().max()
    if max_abs > 1e6:
        logger.warning(
            f"{name} contient des valeurs très grandes (max={max_abs:.2e}). "
            f"Vérifiez que ce sont bien des valeurs SHAP."
        )

    return X_exp


def validate_model(model: Any) -> None:
    """
    Valide qu'un modèle est utilisable avec AntakIA.

    Parameters
    ----------
    model : object
        Modèle à valider

    Raises
    ------
    ValidationError
        Si le modèle n'est pas valide
    """
    if model is None:
        raise ValidationError("Le modèle ne peut pas être None")

    # Vérifier predict
    if not hasattr(model, "predict"):
        raise ValidationError(f"Le modèle {type(model).__name__} n'a pas de méthode 'predict'")

    # Vérifier score (optionnel mais recommandé)
    if not hasattr(model, "score"):
        logger.warning(
            f"Le modèle {type(model).__name__} n'a pas de méthode 'score'. "
            f"Les métriques de substitution ne seront pas calculées."
        )


def validate_inputs(
    X: Any,
    y: Any,
    model: Any,
    X_exp: Any | None = None,
    X_test: Any | None = None,
    y_test: Any | None = None,
) -> dict:
    """
    Valide tous les inputs de AntakIA.

    Parameters
    ----------
    X, y, model, X_exp, X_test, y_test
        Inputs à valider

    Returns
    -------
    dict
        Inputs validés et convertis

    Raises
    ------
    ValidationError
        Si un input n'est pas valide
    """
    result = {}

    # Validation principale
    result["X"] = validate_X(X, "X")
    result["y"] = validate_y(y, result["X"], "y")
    validate_model(model)
    result["model"] = model

    # Validation optionnelle
    result["X_exp"] = validate_X_exp(X_exp, result["X"], "X_exp")

    if X_test is not None:
        result["X_test"] = validate_X(X_test, "X_test")
        if y_test is None:
            raise ValidationError("y_test requis si X_test fourni")
        result["y_test"] = validate_y(y_test, result["X_test"], "y_test")

        # Vérifier que X_test a les mêmes colonnes que X
        if not result["X_test"].columns.equals(result["X"].columns):
            raise ValidationError("X_test a des colonnes différentes de X")
    else:
        result["X_test"] = None
        result["y_test"] = None

    logger.info(
        f"Inputs validés: X={result['X'].shape}, "
        f"X_exp={'fourni' if result['X_exp'] is not None else 'à calculer'}, "
        f"X_test={'fourni' if result['X_test'] is not None else 'non fourni'}"
    )

    return result
