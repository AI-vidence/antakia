"""
Vue duale par feature : ES (beeswarm) | VS (histogram) côte à côte.

Une ligne par feature : à gauche les contributions SHAP, à droite la distribution
des valeurs. Inspiré des prototypes EPITA Flow/Beeswarm.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from antakia.gui.components.beeswarm_plot import compute_beeswarm_jitter


def create_feature_dual_figure(
    X: pd.DataFrame,
    X_exp: pd.DataFrame,
    selection_mask: Optional[pd.Series] = None,
    max_features: int = 8,
    max_points: int = 400,
    height_per_feature: int = 100,
) -> go.Figure:
    """
    Create dual view: ES (beeswarm SHAP) | VS (histogram) per feature.

    Parameters
    ----------
    X : pd.DataFrame
        Original feature values (VS)
    X_exp : pd.DataFrame
        SHAP values (ES)
    selection_mask : pd.Series, optional
        Boolean mask for selected points
    max_features : int
        Max features to display
    max_points : int
        Max points per feature (subsample)
    height_per_feature : int
        Pixel height per row

    Returns
    -------
    go.Figure
    """
    common_cols = [c for c in X_exp.columns if c in X.columns]
    if not common_cols:
        common_cols = X_exp.columns[:max_features].tolist()
    features = common_cols[:max_features]
    n_features = len(features)
    if n_features == 0:
        return go.Figure()

    # 2 columns: ES (beeswarm) | VS (histogram)
    fig = make_subplots(
        rows=n_features,
        cols=2,
        column_widths=[0.55, 0.45],
        shared_xaxes=False,
        vertical_spacing=0.03,
        row_titles=features,
        horizontal_spacing=0.08,
    )

    for i, feat in enumerate(features):
        shap_vals = X_exp[feat].values
        feat_vals = X[feat].values
        n = len(shap_vals)

        if n > max_points:
            idx = np.random.RandomState(42).choice(n, max_points, replace=False)
            shap_vals = shap_vals[idx]
            feat_vals = feat_vals[idx]
            sel = (
                selection_mask.values[idx]
                if selection_mask is not None
                else np.ones(max_points, dtype=bool)
            )
        else:
            sel = (
                selection_mask.values
                if selection_mask is not None
                else np.ones(n, dtype=bool)
            )

        # Left: beeswarm SHAP
        y_jitter = compute_beeswarm_jitter(shap_vals)
        f_min, f_max = feat_vals.min(), feat_vals.max()
        if f_max > f_min:
            color_norm = (feat_vals - f_min) / (f_max - f_min)
        else:
            color_norm = np.ones_like(feat_vals) * 0.5

        for is_sel, name in [(False, "other"), (True, "selected")]:
            mask = sel if is_sel else ~sel
            if not mask.any():
                continue
            marker_c = (
                dict(
                    color=color_norm[mask],
                    colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
                    showscale=(i == 0),
                    colorbar=dict(title="VS value", len=0.3) if i == 0 else None,
                )
                if is_sel
                else dict(color="#cccccc")
            )
            fig.add_trace(
                go.Scattergl(
                    x=shap_vals[mask],
                    y=y_jitter[mask],
                    mode="markers",
                    marker=dict(size=5, line=dict(width=0), **marker_c),
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )

        # Right: histogram VS (all + selected overlay)
        fig.add_trace(
            go.Histogram(
                x=feat_vals,
                nbinsx=min(30, max(10, n // 20)),
                marker_color="rgba(150,150,150,0.5)",
                name="all",
                showlegend=False,
            ),
            row=i + 1,
            col=2,
        )
        if sel.any():
            fig.add_trace(
                go.Histogram(
                    x=feat_vals[sel],
                    nbinsx=min(30, max(10, int(sel.sum()) // 10)),
                    marker_color="rgba(230,126,34,0.7)",
                    name="selected",
                    showlegend=False,
                ),
                row=i + 1,
                col=2,
            )

    fig.update_layout(
        height=n_features * height_per_feature,
        margin=dict(l=60, r=40, t=20, b=40),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="SHAP", col=1)
    fig.update_xaxes(title_text="Value", col=2)
    return fig


class FeatureDualView:
    """Widget wrapper for the feature dual view (ES | VS per feature)."""

    def __init__(self, data_store, height_per_feature: int = 100):
        self.data_store = data_store
        self.height_per_feature = height_per_feature
        self._figure = None
        self._widget = None

    def build_widget(self):
        """Build the Plotly widget. Returns None if data not ready."""
        self._widget = self._create_figure_widget()
        return self._widget

    def _create_figure_widget(self):
        try:
            fig = self._compute_figure()
            if fig is None:
                return None
            w = go.FigureWidget(fig)
            w.layout.height = fig.layout.height
            return w
        except Exception:
            return None

    def _compute_figure(self) -> Optional[go.Figure]:
        X = self.data_store.X
        X_exp = self.data_store.X_exp
        if X_exp is None or X is None:
            return None
        selection_mask = None
        if hasattr(self.data_store, "selection_mask") and self.data_store.selection_mask is not None:
            selection_mask = self.data_store.selection_mask
        return create_feature_dual_figure(
            X=X,
            X_exp=X_exp,
            selection_mask=selection_mask,
            height_per_feature=self.height_per_feature,
        )

    def refresh(self):
        """Refresh the plot with current data."""
        if self._widget is None:
            return
        fig = self._compute_figure()
        if fig is None:
            return
        if hasattr(self._widget, "data"):
            self._widget.data = fig.data
            self._widget.layout = fig.layout
