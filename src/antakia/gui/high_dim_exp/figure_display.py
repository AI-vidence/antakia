from __future__ import annotations

import logging as logging
from functools import partial
from typing import Callable

import antakia_core.utils as utils
import ipyvuetify as v
import numpy as np
import pandas as pd
from antakia_core.data_handler import Region, RegionSet
from antakia_core.utils import timeit
from plotly.express.colors import sample_colorscale
from plotly.graph_objects import FigureWidget, Scatter3d, Scattergl
from sklearn.neighbors import NearestNeighbors

from antakia.config import AppConfig
from antakia.gui.helpers.data import DataStore
from antakia.utils.colors import colors
from antakia.utils.logging_utils import Log, conf_logger
from antakia.utils.other_utils import NotInitialized
from antakia.utils.stats import log_errors, stats_logger

logger = logging.getLogger(__name__)
conf_logger(logger)


class FigureDisplay:
    """
    A FigureDisplay objet manages all operation on a scatter plot
    This class is only responsible for displaying the provided data

    It can display in 3 or 2 dimensions.

    Attributes :

    """

    # Trace indexes : 0 for values, 1 for rules, 2 for regions, 3 for region, 4 for archetype
    NUM_TRACES = 5
    VALUES_TRACE = 0
    RULES_TRACE = 1
    REGIONSET_TRACE = 2
    REGION_TRACE = 3
    ARCHETYPE_TRACE = 4

    @staticmethod
    def trace_name(trace_id: int) -> str:
        """
        get trace name from id
        Parameters
        ----------
        trace_id : int

        Returns
        -------

        """
        if trace_id == FigureDisplay.VALUES_TRACE:
            return "values trace"
        elif trace_id == FigureDisplay.RULES_TRACE:
            return "rules trace"
        elif trace_id == FigureDisplay.REGIONSET_TRACE:
            return "regionset trace"
        elif trace_id == FigureDisplay.REGION_TRACE:
            return "region trace"
        else:
            return "unknown trace"

    def __init__(self, data_store: DataStore, selection_changed: Callable, space: str):
        """

        Parameters
        ----------
        data_store: data to display, should be 2 or 3D
        selection_changed : callable called when a selection changed
        """
        # current active trace
        self.active_trace = 0
        # mask of value to display to limit points on graph
        self._display_mask: pd.Series | None = None
        # callback to notify gui that the selection has changed
        self.selection_changed = partial(selection_changed, self)
        self.data_store = data_store
        self.figure_data: pd.DataFrame | None = None

        self.space = space

        # Now we can init figure
        self.widget = v.Container()
        self.widget.class_ = "flex-fill"

        # is graph selectable
        self._selection_mode = "lasso"
        # is this selection first since last deselection ?
        self.first_selection = True

        # traces to show
        self._visible = [True, False, False, False, True]
        # trace_colors (archetype trace has no color series)
        self._colors: list[pd.Series | None] = [None, None, None, None, None]

        # figures
        self.figure_2D = self.figure_3D = None
        # is the class fully initialized
        self.initialized = False

    @property
    def figure(self):
        if self.dim == 2:
            return self.figure_2D
        else:
            return self.figure_3D

    @figure.setter
    def figure(self, value):
        if self.dim == 2:
            self.figure_2D = value
        else:
            self.figure_3D = value

    @property
    def dim(self):
        if self.figure_data is None:
            return AppConfig.ATK_DEFAULT_DIMENSION
        return self.figure_data.shape[1]

    def initialize(self, figure_data: pd.DataFrame | None):
        """
        inital computation called at startup, after init to compute required values
        Parameters
        ----------
        X : data to display, if NOne use the one provided during init

        Returns
        -------

        """
        if figure_data is not None:
            self.update_X(figure_data)
            # compute X if needed
            self._get_figure_data(masked=True)
        self.initialized = True

    # ---- display Methods ------

    def disable_selection(self, is_disabled: bool):
        """
        enable/disable selection on graph
        Parameters
        ----------
        is_disabled

        Returns
        -------

        """

        self._selection_mode = False if is_disabled else "lasso"  # type: ignore
        if self.dim == 2 and self.figure is not None:
            self.figure.update_layout(dragmode=self._selection_mode)

    @timeit
    def _show_trace(self, trace_id: int):
        """
        show/hide trace
        Parameters
        ----------
        trace_id : trace to change

        Returns
        -------

        """
        archetype_idx = self.data_store.get_archetype_idx()
        show_archetype = trace_id in (self.VALUES_TRACE, self.RULES_TRACE) and archetype_idx is not None
        for i in range(len(self._visible)):
            if i == self.ARCHETYPE_TRACE:
                self._visible[i] = show_archetype
                self.figure.data[i].visible = show_archetype
            else:
                self._visible[i] = trace_id == i
                self.figure.data[i].visible = trace_id == i

    @timeit
    def display_rules(self):
        """
        display a rule vs a selection
        Parameters
        ----------
        selection_mask: boolean series of selected points
        rules_mask: boolean series of rule validating points

        Returns
        -------

        """
        self._colors[self.RULES_TRACE] = self.data_store.rule_selection_color
        self._refresh_color(self.RULES_TRACE)

    @timeit
    def display_regionset(self, region_set: RegionSet):
        """
        display a region set, each region in its color
        Parameters
        ----------
        region_set

        Returns
        -------

        """
        self._colors[self.REGIONSET_TRACE] = region_set.get_color_serie()
        self._refresh_color(self.REGIONSET_TRACE)

    @timeit
    def display_region(self, region: Region):
        """
        display a single region
        Parameters
        ----------
        region

        Returns
        -------

        """
        self._colors[self.REGION_TRACE] = region.get_color_serie()
        self._refresh_color(self.REGION_TRACE)

    @timeit
    def display_region_value(self, region: Region, y: pd.Series):
        """
        display a single region, with target colors
        Parameters
        ----------
        region

        Returns
        -------

        """
        if self.figure_data is None:
            return
        if y.min() == y.max():
            y[:] = 0.5
        else:
            y = (y + max(-y.min(), y.max())) / (2 * max(-y.min(), y.max()))
        color_serie = pd.Series(index=self.figure_data.index)
        color_serie[~region.mask] = colors["gray"]

        # cmap = ['blue', 'green', 'red1']
        # cmap = [colors[c] for c in cmap]
        cmap = "Portland"

        color_serie[region.mask] = sample_colorscale(cmap, y[region.mask], low=0, high=1)
        self._colors[self.REGION_TRACE] = color_serie
        self._refresh_color(self.REGION_TRACE)

    @timeit
    def set_color(self, color: pd.Series, trace_id: int):
        """
        set the provided color as the scatter point color on the provided trace id
        do not alter show/hide trace
        Parameters
        ----------
        color
        trace_id

        Returns
        -------

        """
        self._colors[trace_id] = color
        self._refresh_color(trace_id)

    def _align_masked_colors(self, colors: pd.Series) -> pd.Series:
        """
        Return colors for displayed points, aligning display_mask with colors.index.
        Handles index mismatch after outlier removal (display_mask from data_store.X,
        colors from data_store.y - indices must match).
        """
        mask = self.display_mask
        if not mask.index.equals(colors.index):
            mask = mask.reindex(colors.index, fill_value=False).astype(bool)
        return colors[mask]

    MAX_HOVER_FEATURES = 5

    def _get_importance_ordered_columns(self, ds) -> list:
        """Retourne les colonnes X ordonnées par importance (SHAP ou feature_importances_)."""
        if ds.X_exp is not None and len(ds.X_exp.columns) > 0:
            return (
                ds.X_exp.abs()
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
        if hasattr(ds.model, "feature_importances_") and ds.model.feature_importances_ is not None:
            imp = ds.model.feature_importances_
            names = getattr(ds.model, "feature_names_in_", None) or list(ds.X.columns)
            if len(imp) == len(names):
                order = np.argsort(imp)[::-1]
                return [names[i] for i in order if names[i] in ds.X.columns]
        return list(ds.X.columns)

    def _build_hover_data(self) -> tuple[np.ndarray, str]:
        """
        Build customdata and hovertemplate for hover tooltip.
        ES: SHAP values of main variables, or fallback to VS values (y + X par importance).
        VS: y + main variables from X.
        """
        ds = self.data_store
        mask = self.display_mask
        if not mask.index.equals(ds.y.index):
            mask = mask.reindex(ds.y.index, fill_value=False).astype(bool)
        n_pts = mask.sum()

        # Base: target y
        try:
            y_masked = ds.y.loc[mask].values.astype(float)
        except Exception as e:
            logger.warning(f"Hover: y alignment failed: {e}")
            y_masked = np.full(n_pts, np.nan)

        customdata = y_masked.reshape(-1, 1)
        labels = ["y"]

        try:
            if self.space == "ES" and ds.X_exp is not None and len(ds.X_exp.columns) > 0:
                # ES: priorité 1 = valeurs SHAP des principales variables
                importance_order = self._get_importance_ordered_columns(ds)
                main_vars = [
                    v.column_name
                    for v in ds.variables.variables.values()
                    if getattr(v, "main_feature", False) and v.column_name in ds.X_exp.columns
                ]
                feat_order = main_vars if main_vars else importance_order
                feat_cols = [c for c in feat_order if c in ds.X_exp.columns][: self.MAX_HOVER_FEATURES]

                if feat_cols:
                    shap_vals = ds.X_exp.loc[mask, feat_cols].values.astype(float)
                    customdata = np.column_stack([y_masked, shap_vals])
                    labels = ["y"] + [f"SHAP {c}" for c in feat_cols]
                else:
                    # Fallback ES: valeurs VS (y + X par importance SHAP globale)
                    vs_cols = [c for c in importance_order if c in ds.X.columns][: self.MAX_HOVER_FEATURES]
                    if vs_cols:
                        x_vals = ds.X.loc[mask, vs_cols].values.astype(float)
                        customdata = np.column_stack([y_masked, x_vals])
                        labels = ["y"] + list(vs_cols)
            else:
                # VS ou ES sans X_exp: y + X par importance (feature_importances_ ou ordre)
                importance_order = self._get_importance_ordered_columns(ds)
                main_vars = [
                    v.column_name
                    for v in ds.variables.variables.values()
                    if getattr(v, "main_feature", False)
                ]
                feat_order = main_vars if main_vars else importance_order
                feat_cols = [c for c in feat_order if c in ds.X.columns][: self.MAX_HOVER_FEATURES]
                if feat_cols:
                    x_vals = ds.X.loc[mask, feat_cols].values.astype(float)
                    customdata = np.column_stack([y_masked, x_vals])
                    labels = ["y"] + list(feat_cols)
        except Exception as e:
            logger.warning(f"Hover data build failed ({self.space}): {e}")

        # Build hovertemplate
        parts = [f"{labels[0]}: %{{customdata[0]:.3f}}"]
        for i in range(1, customdata.shape[1]):
            parts.append(f"<br>{labels[i]}: %{{customdata[{i}]:.3f}}")
        hovertemplate = "".join(parts) + "<extra></extra>"
        return customdata, hovertemplate

    @timeit
    def _refresh_data(self):
        """
        refresh all traces data, create figure if absent
        Returns
        -------

        """
        if self.figure is None:
            self.create_figure()
        projection = self._get_figure_data(masked=True)

        customdata, hovertemplate = self._build_hover_data()
        with self.figure.batch_update():
            for trace_id in range(self.NUM_TRACES - 1):  # Exclude archetype trace
                self.figure.data[trace_id].x = projection[0]
                self.figure.data[trace_id].y = projection[1]
                self.figure.data[trace_id].customdata = customdata
                self.figure.data[trace_id].hovertemplate = hovertemplate
                if self.dim == 3:
                    self.figure.data[trace_id].z = projection[2]
            trace_id = self.active_trace
            colors = self._colors[trace_id]
            if colors is None:
                colors = self.data_store.y
            masked_colors = self._align_masked_colors(colors)
            self.figure.data[trace_id].marker.color = masked_colors
            # Update archetype trace
            self._refresh_archetype_trace()

    @timeit
    def _refresh_color(self, trace_id: int):
        """
        refresh the provided trace id
        do not alter show/hide trace

        Parameters
        ----------
        trace_id

        Returns
        -------

        """
        if self.figure is None:
            raise ValueError()
        else:
            if trace_id == self.active_trace:
                if len(self.figure.data[trace_id].x) == 0:
                    return self._refresh_data()
                else:
                    colors = self._colors[trace_id]
                    if colors is None:
                        colors = self.data_store.y
                    masked_colors = self._align_masked_colors(colors)
                    with self.figure.batch_update():
                        self.figure.data[trace_id].marker.color = masked_colors

    def _refresh_archetype_trace(self):
        """Update the archetype (typical point) trace with a distinct marker."""
        if self.figure is None or self.figure_data is None:
            return
        archetype_idx = self.data_store.get_archetype_idx()
        tid = self.ARCHETYPE_TRACE
        if archetype_idx is None or archetype_idx not in self.figure_data.index:
            with self.figure.batch_update():
                self.figure.data[tid].x = []
                self.figure.data[tid].y = []
                if self.dim == 3:
                    self.figure.data[tid].z = []
            return
        row = self.figure_data.loc[archetype_idx]
        with self.figure.batch_update():
            self.figure.data[tid].x = [row.iloc[0]]
            self.figure.data[tid].y = [row.iloc[1]]
            if self.dim == 3:
                self.figure.data[tid].z = [row.iloc[2]]

    @timeit
    def update_X(self, X: pd.DataFrame | None):
        """
        changes the underlying data - update the data used in display and dimension is necessary
        Parameters
        ----------
        X : data to display

        Returns
        -------

        """
        self.figure_data = X
        self._prepare_mask_extrapolation()
        self._refresh_data()
        active_trace = self._visible.index(True)
        with self.figure.batch_update():
            self._refresh_color(active_trace)
        self.widget.children = [self.figure]

    @timeit
    def _prepare_mask_extrapolation(self):
        X_train = self._get_figure_data(masked=True)
        nn = NearestNeighbors(n_neighbors=3).fit(X_train)

        X_predict = self._get_figure_data(masked=False)
        dist, neighbors = nn.kneighbors(X_predict, return_distance=True)

        neighbors_weight = 1 / (dist + 0.0001)
        neighbors_weight = (neighbors_weight.T / neighbors_weight.sum(axis=1)).T
        self._neighbors_data = neighbors_weight, neighbors

    @timeit
    def selection_to_mask(self, row_numbers: list[int]):
        """

        extrapolate selected row_numbers to full dataframe and return the selection mask on all the dataframe

        Parameters
        ----------
        row_numbers

        Returns
        -------

        """
        if self.figure_data is None:
            raise NotInitialized()
        selection = utils.rows_to_mask(self.figure_data[self.display_mask], row_numbers)
        if not selection.any() or selection.all():
            return utils.boolean_mask(self._get_figure_data(masked=False), selection.iloc[0])
        if self.display_mask.all():
            return selection

        neighbors_weight, neighbors = self._neighbors_data
        neighbors_label = np.zeros(neighbors.shape)
        for k in range(neighbors.shape[1]):
            neighbors_label[:, k] = selection.iloc[neighbors[:, k]]
        majority_label = (neighbors_label * neighbors_weight).sum(axis=1).round()

        guessed_selection = pd.Series(majority_label, index=self.figure_data.index)
        return guessed_selection.astype(bool)

    @log_errors
    @timeit
    def _selection_event(self, trace_id, trace, points, *args):
        """
        callback triggered by selection on graph
        selects points and update display on both hde (calls selection changed)
        deselects if no points selected

        update selection taking intersection
        Parameters
        ----------
        trace
        points
        args

        Returns
        -------

        """
        if trace_id == self.active_trace:
            # Previously: create_figure() + _refresh_data() were used as workaround for
            # selection not displaying. With batch_update in display_selection(), we rely
            # on selection_changed -> display_selection() to update both VS and ES figures.
            # Avoiding figure recreation prevents flicker and ensures the other space
            # receives the same selection_mask and can display it consistently.
            self.first_selection |= self.data_store.empty_selection
            stats_logger.log(
                "hde_selection",
                {
                    "first_selection": str(self.first_selection),
                    "space": str(self.space),
                    "points": self.data_store.selection_mask.mean(),
                },
            )
            extrapolated_selection = self.selection_to_mask(points.point_inds)
            self.data_store.selection_mask &= extrapolated_selection
            if not self.data_store.empty_selection:
                self.selection_changed("selection_event")
            else:
                self._deselection_event(trace_id)

    @log_errors
    @timeit
    def _deselection_event(self, trace_id, *args):
        """
        clear selection -- called by deselection on graph
        synchronize hdes

        Parameters
        ----------
        args
        rebuild

        Returns
        -------

        """
        if trace_id == self.active_trace:
            stats_logger.log(
                "hde_deselection",
                {"first_selection": str(self.first_selection), "space": str(self.space)},
            )
            # We tell the GUI
            self.first_selection = False
            self.data_store.selection_mask = utils.boolean_mask(self.figure_data, True)
            self.selection_changed("selection_event")

    @timeit
    def display_selection(self):
        """
        Display selection on figure (VS or ES).
        Uses batch_update so Plotly properly applies selectedpoints and replicates
        the selection visually. Essential for dyadic exploration: selection in one
        space must be visible in the other.
        """
        if self.figure is None:
            return
        with Log("display_selection " + self.space, level=3):
            rows = utils.mask_to_rows(self.data_store.selection_mask[self.display_mask])
            with self.figure.batch_update():
                self.figure.data[self.active_trace].selectedpoints = rows

    @property
    @timeit
    def display_mask(self) -> pd.Series:
        """
        mask should be applied on each display (x,y,z,color, selection)
        """
        return self.data_store.display_mask
        if self.figure_data is None:
            raise NotInitialized()
        if self._display_mask is None:
            limit = AppConfig.ATK_MAX_DOTS
            if len(self.figure_data) > limit:
                self._display_mask = pd.Series(
                    [False] * len(self.figure_data), index=self.figure_data.index
                )
                indices = np.random.choice(self.figure_data.index, size=limit, replace=False)
                self._display_mask.loc[indices] = True
            else:
                self._display_mask = pd.Series(
                    [True] * len(self.figure_data), index=self.figure_data.index
                )
        return self._display_mask

    @timeit
    def create_figure(self):
        """
        Builds the FigureWidget for the given dimension with no data
        """
        hde_marker = {"color": self.data_store.y, "colorscale": "Viridis"}
        if self.dim == 3:
            hde_marker["size"] = 2

        fig_args = {
            "x": [],
            "y": [],
            "mode": "markers",
            "marker": hde_marker,
            "customdata": [],
            "hovertemplate": "%{customdata:.3f}",
        }
        if self.dim == 3:
            fig_args["z"] = []
            fig_builder = Scatter3d
        else:
            fig_builder = Scattergl

        self.figure = FigureWidget(data=[fig_builder(**fig_args)])  # Trace 0 for dots
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 1 for rules
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 2 for region set
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 3 for region
        # Trace 4: archetype (typical point) - distinct marker
        arch_args = {
            "x": [],
            "y": [],
            "mode": "markers",
            "marker": {"symbol": "star", "size": 14, "color": "orange", "line": {"width": 2, "color": "darkorange"}},
            "customdata": [],
            "hovertemplate": "Typical point<extra></extra>",
        }
        if self.dim == 3:
            arch_args["z"] = []
            self.figure.add_trace(Scatter3d(**arch_args))
        else:
            self.figure.add_trace(Scattergl(**arch_args))

        self.figure.update_layout(dragmode=self._selection_mode)
        self.figure.update_traces(
            selected={"marker": {"opacity": 1.0}},
            unselected={"marker": {"opacity": 0.1}},
            selector={"type": "scatter"},
        )
        self.figure.update_layout(
            autosize=True,
            margin={"t": 0, "b": 0, "l": 0, "r": 0},
        )
        self.figure._config = self.figure._config | {"displaylogo": False}
        self.figure._config = self.figure._config | {"displayModeBar": True}
        # We don't want the name of the trace to appear :
        for trace_id in range(len(self.figure.data)):
            self.figure.data[trace_id].showlegend = False

        if self.dim == 2:
            # selection only on trace 0
            self.figure.data[0].on_selection(partial(self._selection_event, 0))
            self.figure.data[0].on_deselect(partial(self._deselection_event, 0))
            self.figure.data[1].on_selection(partial(self._selection_event, 1))
            self.figure.data[1].on_deselect(partial(self._deselection_event, 1))
        self.widget.children = [self.figure]

    @timeit
    def rebuild(self):
        with Log("rebuild " + self.space, level=3):
            self.create_figure()
            self._refresh_data()
            self._refresh_color(self.active_trace)
            self._show_trace(self.active_trace)

    def _get_figure_data(self, masked: bool) -> pd.DataFrame:
        """

        return current projection value
        its computes it if necessary - progress is published in the callback

        Parameters
        ----------
        masked

        Returns
        -------

        """
        if self.figure_data is None:
            raise NotInitialized()
        if masked:
            return self.figure_data.loc[self.display_mask]
        return self.figure_data

    def set_tab(self, tab):
        """
        show/hide trace depending on tab
        Parameters
        ----------
        tab

        Returns
        -------

        """
        self.disable_selection(tab >= 1)
        self._show_trace(tab)
        self.active_trace = tab
        self._refresh_color(tab)
