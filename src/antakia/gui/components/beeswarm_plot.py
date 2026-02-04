"""
Beeswarm SHAP plot component for AntakIA.

Displays SHAP values as a beeswarm plot, showing the impact of each feature
on the model predictions.
"""

from typing import Optional

import ipyvuetify as v
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from antakia.gui.helpers.data import DataStore


def compute_beeswarm_jitter(values: np.ndarray, n_bins: int = 50, width: float = 0.4) -> np.ndarray:
    """
    Compute y-axis jitter for beeswarm plot.

    Distributes points vertically to avoid overlap while preserving
    the distribution shape.

    Parameters
    ----------
    values : np.ndarray
        Values to compute jitter for (usually SHAP values)
    n_bins : int
        Number of bins for density estimation
    width : float
        Maximum jitter width

    Returns
    -------
    np.ndarray
        Y-axis jitter values
    """
    n = len(values)
    if n == 0:
        return np.array([])

    x_min, x_max = np.min(values), np.max(values)
    if x_min == x_max:
        return np.zeros(n)

    # Bin the values
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_indices = np.digitize(values, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute jitter based on bin density
    y_jitter = np.zeros(n)
    for b in range(n_bins):
        mask = bin_indices == b
        count = np.sum(mask)
        if count > 0:
            positions = np.linspace(-width / 2, width / 2, count)
            y_jitter[mask] = positions

    return y_jitter


def create_beeswarm_shap_plot(
    X: pd.DataFrame,
    X_exp: pd.DataFrame,
    selection_mask: Optional[np.ndarray] = None,
    max_features: int = 10,
    max_points: int = 500,
    height_per_feature: int = 70,
) -> go.Figure:
    """
    Create a beeswarm plot for SHAP values.

    Parameters
    ----------
    X : pd.DataFrame
        Feature values
    X_exp : pd.DataFrame
        SHAP values (same columns as X)
    selection_mask : np.ndarray, optional
        Boolean mask for selected points
    max_features : int
        Maximum number of features to display
    max_points : int
        Maximum points per feature
    height_per_feature : int
        Plot height per feature in pixels

    Returns
    -------
    go.Figure
        Plotly figure with beeswarm plot
    """
    # Get common columns
    common_cols = [c for c in X.columns if c in X_exp.columns]

    # Select top features by mean absolute SHAP
    features = sorted(common_cols, key=lambda f: np.abs(X_exp[f]).mean(), reverse=True)[
        :max_features
    ]

    n_features = len(features)
    fig = go.Figure()

    for i, feat in enumerate(features):
        shap_vals = X_exp[feat].values
        feat_vals = X[feat].values
        n = len(shap_vals)

        # Subsample if too many points
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
        else:
            idx = np.arange(n)

        shap_vals = shap_vals[idx]
        feat_vals = feat_vals[idx]
        sel = selection_mask[idx] if selection_mask is not None else np.ones(len(idx), dtype=bool)

        # Compute y jitter
        y_jitter = compute_beeswarm_jitter(shap_vals)

        # Normalize feature values for color
        f_min, f_max = np.min(feat_vals), np.max(feat_vals)
        if f_max > f_min:
            color_norm = (feat_vals - f_min) / (f_max - f_min)
        else:
            color_norm = np.zeros_like(feat_vals)

        # Plot selected and non-selected points
        for is_sel, name in [(False, "other"), (True, "selected")]:
            mask = sel == is_sel
            if not np.any(mask):
                continue

            marker_c = color_norm[mask]

            fig.add_trace(
                go.Scatter(
                    x=shap_vals[mask],
                    y=i + y_jitter[mask],
                    mode="markers",
                    marker=dict(
                        size=5 if is_sel else 4,
                        color=marker_c,
                        colorscale="RdBu_r",
                        opacity=0.8 if is_sel else 0.3,
                        line=dict(width=1, color="black") if is_sel else dict(width=0),
                    ),
                    name=f"{feat} ({name})",
                    showlegend=False,
                    hovertemplate=f"{feat}<br>SHAP: %{{x:.3f}}<br>Value: %{{marker.color:.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        height=height_per_feature * n_features,
        yaxis=dict(
            tickvals=list(range(n_features)),
            ticktext=features,
            title="Feature",
        ),
        xaxis=dict(title="SHAP value"),
        showlegend=False,
        margin=dict(l=120, r=20, t=30, b=40),
    )

    return fig


class BeeswarmPlot:
    """
    Interactive beeswarm SHAP plot widget for AntakIA.

    Displays SHAP values as a beeswarm plot and updates
    when selection changes.
    """

    def __init__(self, data_store: DataStore, height_per_feature: int = 70):
        """
        Initialize the beeswarm plot.

        Parameters
        ----------
        data_store : DataStore
            Data store containing X and SHAP values
        height_per_feature : int
            Height per feature in pixels
        """
        self.data_store = data_store
        self.height_per_feature = height_per_feature
        self._widget = None
        self._figure_widget = None

    def build_widget(self) -> Optional[v.Container]:
        """
        Build the widget.

        Returns None if SHAP values are not available.
        """
        if self.data_store.X_exp is None:
            return None

        self._figure_widget = self._create_figure_widget()
        self._widget = v.Container(
            fluid=True, children=[self._figure_widget] if self._figure_widget else []
        )
        return self._widget

    def _create_figure_widget(self):
        """Create the Plotly FigureWidget."""
        fig = self._compute_figure()
        if fig is None:
            return None

        try:
            import plotly.graph_objects as go

            w = go.FigureWidget(fig)
            return w
        except Exception:
            return None

    def _compute_figure(self) -> Optional[go.Figure]:
        """Compute the beeswarm figure."""
        X = self.data_store.X
        X_exp = self.data_store.X_exp

        if X_exp is None or len(X_exp) == 0:
            return None

        selection_mask = (
            self.data_store.selection_mask if hasattr(self.data_store, "selection_mask") else None
        )

        return create_beeswarm_shap_plot(
            X,
            X_exp,
            selection_mask=selection_mask,
            height_per_feature=self.height_per_feature,
        )

    def refresh(self):
        """Refresh the plot with current data."""
        if self._figure_widget is None:
            return

        fig = self._compute_figure()
        if fig is not None:
            with self._figure_widget.batch_update():
                self._figure_widget.data = []
                for trace in fig.data:
                    self._figure_widget.add_trace(trace)
                self._figure_widget.layout = fig.layout
