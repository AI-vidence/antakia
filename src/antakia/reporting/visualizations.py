"""
Visualisations pour les rapports de tessellation.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_shap_summary(
    shap_values: pd.DataFrame,
    max_features: int = 10,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Crée un summary plot SHAP (beeswarm simplifié).

    Parameters
    ----------
    shap_values : pd.DataFrame
        Valeurs SHAP (samples x features)
    max_features : int
        Nombre max de features à afficher
    figsize : tuple
        Taille de la figure

    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    # Importance moyenne
    importance = shap_values.abs().mean().sort_values(ascending=True)
    top_features = importance.tail(max_features)

    fig, ax = plt.subplots(figsize=figsize)

    # Barplot horizontal
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features.values, color=colors)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")

    # Ajouter les valeurs sur les barres
    for i, (bar, val) in enumerate(zip(bars, top_features.values)):
        ax.text(val + 0.01 * top_features.max(), i, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_pdp_comparison(
    feature: str,
    grid: np.ndarray,
    pdp_initial_all: np.ndarray,
    pdp_initial_filtered: np.ndarray,
    pdp_surrogate: Optional[np.ndarray] = None,
    region_num: int = 0,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Crée un plot PDP comparatif.

    Montre 3 courbes :
    - Modèle initial (tout le dataset)
    - Modèle initial (filtré sur la tesselle)
    - Modèle surrogate (si disponible)

    Parameters
    ----------
    feature : str
        Nom de la feature
    grid : np.ndarray
        Grille de valeurs
    pdp_initial_all : np.ndarray
        PDP du modèle initial sur tout le dataset
    pdp_initial_filtered : np.ndarray
        PDP du modèle initial filtré sur la tesselle
    pdp_surrogate : np.ndarray, optional
        PDP du modèle surrogate
    region_num : int
        Numéro de la région (pour le titre)
    figsize : tuple
        Taille de la figure

    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Courbe 1: Modèle initial (tout)
    ax.plot(
        grid,
        pdp_initial_all,
        color="blue",
        linewidth=2,
        linestyle="-",
        label="Modèle initial (global)",
    )

    # Courbe 2: Modèle initial (filtré)
    ax.plot(
        grid,
        pdp_initial_filtered,
        color="orange",
        linewidth=2,
        linestyle="--",
        label="Modèle initial (tesselle)",
    )

    # Courbe 3: Modèle surrogate
    if pdp_surrogate is not None:
        ax.plot(
            grid, pdp_surrogate, color="green", linewidth=2, linestyle=":", label="Modèle surrogate"
        )

    ax.set_xlabel(feature)
    ax.set_ylabel("Prédiction partielle")
    ax.set_title(f"PDP Comparatif: {feature} (Tesselle {region_num})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance_comparison(
    global_importance: pd.Series,
    local_importance: pd.Series,
    region_num: int = 0,
    max_features: int = 10,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Compare l'importance des features globale vs locale.

    Parameters
    ----------
    global_importance : pd.Series
        Importance globale (tout le dataset)
    local_importance : pd.Series
        Importance locale (tesselle)
    region_num : int
        Numéro de la région
    max_features : int
        Nombre max de features
    figsize : tuple
        Taille de la figure

    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    # Union des top features
    top_global = set(global_importance.nlargest(max_features).index)
    top_local = set(local_importance.nlargest(max_features).index)
    features = list(top_global | top_local)

    # Limiter
    features = features[:max_features]

    # Données pour le plot
    x = np.arange(len(features))
    width = 0.35

    global_vals = [global_importance.get(f, 0) for f in features]
    local_vals = [local_importance.get(f, 0) for f in features]

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width / 2, global_vals, width, label="Global", color="steelblue")
    bars2 = ax.bar(x + width / 2, local_vals, width, label=f"Tesselle {region_num}", color="coral")

    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"Comparaison Importance Features: Global vs Tesselle {region_num}")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_tessellation_overview(
    region_stats: List[dict],
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Vue d'ensemble des tesselles (camembert + barplot).

    Parameters
    ----------
    region_stats : list of dict
        Liste avec {'num': int, 'color': str, 'coverage': float, 'score': float}
    figsize : tuple
        Taille de la figure

    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Camembert des couvertures
    labels = [f"T{r['num']}" for r in region_stats]
    sizes = [r["coverage"] for r in region_stats]
    colors = [r["color"] for r in region_stats]

    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Couverture par tesselle")

    # Barplot des scores
    scores = [r.get("score", 0) for r in region_stats]
    x = range(len(labels))

    bars = ax2.bar(x, scores, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Score (R² ou accuracy)")
    ax2.set_title("Performance des surrogates")
    ax2.axhline(
        y=np.mean(scores), color="red", linestyle="--", label=f"Moyenne: {np.mean(scores):.3f}"
    )
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_rules_venn(
    region1_mask: pd.Series,
    region2_mask: pd.Series,
    labels: Tuple[str, str] = ("Tesselle 1", "Tesselle 2"),
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Diagramme de Venn pour deux tesselles.

    Parameters
    ----------
    region1_mask : pd.Series
        Masque booléen de la région 1
    region2_mask : pd.Series
        Masque booléen de la région 2
    labels : tuple
        Labels des deux régions
    figsize : tuple
        Taille de la figure

    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    try:
        from matplotlib_venn import venn2
    except ImportError:
        # Fallback si matplotlib_venn n'est pas installé
        fig, ax = plt.subplots(figsize=figsize)
        only_1 = (region1_mask & ~region2_mask).sum()
        only_2 = (~region1_mask & region2_mask).sum()
        both = (region1_mask & region2_mask).sum()

        ax.text(0.3, 0.5, f"{labels[0]}\n{only_1} points", ha="center", fontsize=12)
        ax.text(0.7, 0.5, f"{labels[1]}\n{only_2} points", ha="center", fontsize=12)
        ax.text(0.5, 0.3, f"Intersection\n{both} points", ha="center", fontsize=10)
        ax.set_title("Chevauchement des tesselles")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    only_1 = (region1_mask & ~region2_mask).sum()
    only_2 = (~region1_mask & region2_mask).sum()
    both = (region1_mask & region2_mask).sum()

    venn2(subsets=(only_1, only_2, both), set_labels=labels, ax=ax)
    ax.set_title("Chevauchement des tesselles")

    return fig
