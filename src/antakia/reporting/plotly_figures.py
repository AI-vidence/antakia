"""
Figures Plotly pour les rapports de tessellation.

Réutilise la même logique visuelle que l'onglet Tesselles (Tab3 / ModelExplorer)
pour un rendu professionnel et cohérent. Export en PNG via kaleido pour HTML.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Taille par défaut pour export PNG
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 450


def _plotly_fig_to_base64(fig: "go.Figure", width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> str:
    """
    Convertit une figure Plotly en base64 PNG.

    Utilise kaleido si disponible, sinon échoue (fallback matplotlib géré par l'appelant).
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly requis")
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        if not img_bytes or len(img_bytes) < 100:
            raise ValueError("Export Plotly a produit une image vide. Vérifiez kaleido.")
        return base64.b64encode(img_bytes).decode("utf-8")
    except ValueError as e:
        if "kaleido" in str(e).lower() or "orca" in str(e).lower():
            raise ImportError(
                "kaleido requis pour l'export PNG. Installez avec: pip install kaleido"
            ) from e
        raise


def create_shap_summary_figure(
    shap_values: pd.DataFrame,
    max_features: int = 10,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> "go.Figure":
    """
    Crée une figure Plotly pour l'importance SHAP (barres horizontales).

    Aligné visuellement avec l'onglet Tesselles (Feature Importance).
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly requis")
    importance = shap_values.abs().mean().sort_values(ascending=True)
    top_features = importance.tail(max_features)
    if len(top_features) == 0:
        raise ValueError("Aucune feature SHAP à afficher (shap_values vide)")
    vmin, vmax = top_features.min(), top_features.max()
    if vmin == vmax:
        vmin, vmax = 0, max(vmax, 1e-6)
    fig = go.Figure(
        go.Bar(
            x=top_features.values,
            y=top_features.index,
            orientation="h",
            marker=dict(
                color=top_features.values,
                colorscale="RdYlBu_r",
                cmin=vmin,
                cmax=vmax,
            ),
            text=[f"{v:.3f}" for v in top_features.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Feature Importance (SHAP)",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        height=height,
        width=width,
        margin=dict(l=120, r=80, t=50, b=50),
        showlegend=False,
    )
    return fig


def create_pdp_comparison_figure(
    feature: str,
    grid: np.ndarray,
    pdp_initial_all: np.ndarray,
    pdp_initial_filtered: np.ndarray,
    pdp_surrogate: Optional[np.ndarray] = None,
    region_num: int = 0,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> "go.Figure":
    """
    Crée une figure Plotly pour le PDP comparatif.

    Même sémantique que ModelExplorer : initial (global), initial (tesselle), surrogate.
    Couleurs alignées : bleu (initial global), orange (tesselle), vert (surrogate).
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly requis")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=pdp_initial_all,
            mode="lines",
            name="Modèle initial (global)",
            line=dict(color="blue", width=2.5, dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=pdp_initial_filtered,
            mode="lines",
            name="Modèle initial (tesselle)",
            line=dict(color="orange", width=2.5, dash="dash"),
        )
    )
    if pdp_surrogate is not None:
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=pdp_surrogate,
                mode="lines",
                name="Modèle surrogate",
                line=dict(color="green", width=2.5, dash="dot"),
            )
        )
    fig.update_layout(
        title=f"PDP Comparatif: {feature} (Tesselle {region_num})",
        xaxis_title=feature,
        yaxis_title="Prédiction partielle",
        height=height,
        width=width,
        margin=dict(l=60, r=40, t=50, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )
    return fig


def create_tessellation_overview_figure(
    region_stats: List[Dict[str, Any]],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> "go.Figure":
    """
    Vue d'ensemble des tesselles : camembert (couverture) + barplot (scores).

    Aligné avec le style de l'onglet Tesselles.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly requis")
    labels = [f"T{r['num']}" for r in region_stats]
    sizes = [r["coverage"] for r in region_stats]
    colors = [r["color"] for r in region_stats]
    scores = [r.get("score", 0) for r in region_stats]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Couverture par tesselle", "Performance des surrogates"),
    )
    fig.add_trace(
        go.Pie(labels=labels, values=sizes, marker=dict(colors=colors), textinfo="label+percent"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=scores, marker_color=colors, text=[f"{s:.3f}" for s in scores]),
        row=1,
        col=2,
    )
    if scores:
        mean_score = np.mean(scores)
        fig.add_hline(y=mean_score, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Score (R² ou accuracy)", row=1, col=2)
    return fig


def fig_to_base64_plotly_or_matplotlib(fig: Any) -> str:
    """
    Convertit une figure en base64. Accepte Plotly (go.Figure) ou matplotlib (plt.Figure).

    Priorité Plotly ; fallback matplotlib si la figure est matplotlib.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly requis pour les figures du rapport")
    # Détection : Plotly a .to_image, matplotlib a .savefig
    if hasattr(fig, "to_image"):
        try:
            return _plotly_fig_to_base64(fig)
        except ImportError as e:
            logger.warning(
                f"Export Plotly échoué ({e}). Utilisez 'pip install kaleido' pour un rendu optimal."
            )
            raise
    # Fallback matplotlib (legacy)
    import io

    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64
