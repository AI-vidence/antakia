import ipyvuetify as v
import numpy as np
import pandas as pd
from antakia_core.compute.model_subtitution.model_class import MLModel
from antakia_core.data_handler import ModelRegion
from plotly.graph_objects import Bar, FigureWidget, Scatter

from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors


class ModelExplorer:
    """
    Displays model explanations including feature importances and PDP comparisons.

    The PDP tab shows a comparison between:
    - Original model (on region data)
    - Substitution model (on region data)
    """

    def __init__(self, X: pd.DataFrame, original_model=None):
        self._build_widget()
        self.model: MLModel | None = None  # Substitution model
        self.original_model = original_model  # Original model for comparison
        self.region: ModelRegion | None = None
        self.X = X

    def _build_widget(self):
        self.feature_importance_tab = v.TabItem(  # Tab 1) feature importances # 43
            class_="mt-2", children=[]
        )
        self.pdp_feature_select = v.Select()
        self.pdp_figure = v.Container()
        self.widget = v.Tabs(
            v_model=0,  # default active tab
            children=[
                v.Tab(children=["Feature Importance"]),
                v.Tab(children=["Partial Dependency"]),
            ]
            + [
                self.feature_importance_tab,
                v.TabItem(  # Tab 2) Partial dependence
                    children=[v.Col(children=[self.pdp_feature_select, self.pdp_figure])]
                ),  # End of v.TabItem #2
            ],
        )
        self.pdp_feature_select.on_event("change", self.display_pdp)

    def update_selected_model(self, model: MLModel, region: ModelRegion):
        self.model = model
        self.region = region
        self.update_feature_importances()
        self.update_pdp_tab()

    def update_feature_importances(self):
        if self.model is not None:
            feature_importances = self.model.feature_importances_.sort_values(ascending=True)
            fig = Bar(x=feature_importances, y=feature_importances.index, orientation="h")
            self.figure_fi = FigureWidget(data=[fig])
            self.figure_fi.update_layout(
                autosize=True,
                margin={"t": 0, "b": 0, "l": 0, "r": 0},
            )
            self.figure_fi._config = self.figure_fi._config | {"displaylogo": False}

            self.feature_importance_tab.children = [self.figure_fi]
        else:
            self.feature_importance_tab.children = []

    def update_pdp_tab(self):
        if self.pdp_feature_select.v_model not in self.X.columns and self.model is not None:
            features = list(self.model.feature_importances_.sort_values(ascending=False).index)
            self.pdp_feature_select.items = features
            self.pdp_feature_select.v_model = features[0]
        self.display_pdp()

    def _compute_pdp_values(self, model, X_region, feature, grid_points=50, compute_ci=True):
        """
        Compute PDP values with optional confidence intervals.

        Parameters
        ----------
        model : fitted model
        X_region : DataFrame with region data
        feature : feature name
        grid_points : number of grid points
        compute_ci : if True, compute confidence intervals (std-based)

        Returns
        -------
        grid : array of feature values
        pdp_mean : array of mean predictions
        pdp_lower : array of lower bound (mean - 2*std)
        pdp_upper : array of upper bound (mean + 2*std)
        """
        feature_values = X_region[feature].values
        grid = np.linspace(feature_values.min(), feature_values.max(), grid_points)

        pdp_mean = []
        pdp_lower = []
        pdp_upper = []

        for val in grid:
            X_temp = X_region.copy()
            X_temp[feature] = val
            preds = model.predict(X_temp)

            mean_pred = np.mean(preds)
            pdp_mean.append(mean_pred)

            if compute_ci:
                std_pred = np.std(preds)
                pdp_lower.append(mean_pred - 2 * std_pred)
                pdp_upper.append(mean_pred + 2 * std_pred)

        return (
            grid,
            np.array(pdp_mean),
            np.array(pdp_lower) if compute_ci else None,
            np.array(pdp_upper) if compute_ci else None,
        )

    @log_errors
    def display_pdp(self, *args):
        with Log("display pdp comparison with CI and density", 2):
            if self.model is not None and self.region is not None:
                selected_feature = self.pdp_feature_select.v_model
                X_region = self.X[self.region.mask].copy()

                if X_region[selected_feature].nunique() > 1:
                    from plotly.subplots import make_subplots

                    # Create figure with secondary y-axis for histogram
                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        row_heights=[0.75, 0.25],
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                    )

                    feature_values = X_region[selected_feature].values

                    # ===== TOP PLOT: PDP with confidence intervals =====

                    # 1. PDP for substitution model with confidence interval
                    try:
                        grid_sub, pdp_sub, lower_sub, upper_sub = self._compute_pdp_values(
                            self.model, X_region, selected_feature, compute_ci=True
                        )

                        # Confidence interval fill (substitution)
                        fig.add_trace(
                            Scatter(
                                x=np.concatenate([grid_sub, grid_sub[::-1]]),
                                y=np.concatenate([upper_sub, lower_sub[::-1]]),
                                fill="toself",
                                fillcolor="rgba(0, 200, 0, 0.15)",
                                line=dict(color="rgba(255,255,255,0)"),
                                hoverinfo="skip",
                                showlegend=False,
                                name="CI Substitution",
                            ),
                            row=1,
                            col=1,
                        )

                        # Mean line (substitution)
                        fig.add_trace(
                            Scatter(
                                x=grid_sub,
                                y=pdp_sub,
                                mode="lines",
                                name="Substitution Model",
                                line=dict(color="green", width=2),
                            ),
                            row=1,
                            col=1,
                        )
                    except Exception as e:
                        Log(f"Error computing substitution PDP: {e}", 1)

                    # 2. PDP for original model with confidence interval
                    if self.original_model is not None:
                        try:
                            grid_orig, pdp_orig, lower_orig, upper_orig = self._compute_pdp_values(
                                self.original_model, X_region, selected_feature, compute_ci=True
                            )

                            # Confidence interval fill (original)
                            fig.add_trace(
                                Scatter(
                                    x=np.concatenate([grid_orig, grid_orig[::-1]]),
                                    y=np.concatenate([upper_orig, lower_orig[::-1]]),
                                    fill="toself",
                                    fillcolor="rgba(0, 100, 255, 0.15)",
                                    line=dict(color="rgba(255,255,255,0)"),
                                    hoverinfo="skip",
                                    showlegend=False,
                                    name="CI Original",
                                ),
                                row=1,
                                col=1,
                            )

                            # Mean line (original)
                            fig.add_trace(
                                Scatter(
                                    x=grid_orig,
                                    y=pdp_orig,
                                    mode="lines",
                                    name="Original Model",
                                    line=dict(color="blue", width=2, dash="dash"),
                                ),
                                row=1,
                                col=1,
                            )
                        except Exception as e:
                            Log(f"Error computing original model PDP: {e}", 1)

                    # ===== BOTTOM PLOT: Data density histogram =====
                    from plotly.graph_objects import Histogram

                    fig.add_trace(
                        Histogram(
                            x=feature_values,
                            nbinsx=30,
                            marker=dict(
                                color="rgba(100, 100, 100, 0.5)",
                                line=dict(color="rgba(50, 50, 50, 0.8)", width=0.5),
                            ),
                            name=f"Data density (n={len(feature_values)})",
                            showlegend=True,
                            hovertemplate="%{x:.2f}: %{y} points<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )

                    # Layout
                    fig.update_layout(
                        title=f"PDP: {selected_feature} (±2σ) with data density",
                        autosize=True,
                        width=None,
                        height=400,
                        margin={"t": 40, "b": 40, "l": 50, "r": 10},
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor="rgba(255,255,255,0.8)",
                        ),
                        bargap=0.05,
                    )

                    # Axis labels
                    fig.update_yaxes(title_text="Prediction", row=1, col=1)
                    fig.update_yaxes(title_text="Count", row=2, col=1)
                    fig.update_xaxes(title_text=selected_feature, row=2, col=1)

                    self.figure_pdp = FigureWidget(fig)
                    self.figure_pdp._config = self.figure_pdp._config | {"displaylogo": False}
                    self.pdp_figure.children = [self.figure_pdp]
                else:
                    self.pdp_figure.children = ["Only one feature value, no PDP to display"]
            else:
                self.pdp_figure.children = []

    def reset(self):
        self.model = None
        self.region = None
        self.update_feature_importances()
        self.update_pdp_tab()
