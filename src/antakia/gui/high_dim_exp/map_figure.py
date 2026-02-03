"""
Geographic map figure display for AntakIA.

Uses standard Scattergl with lat/lon as x/y coordinates.
Simple but reliable approach that works with FigureWidget.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable

import ipyvuetify as v
import pandas as pd
from plotly.graph_objects import FigureWidget, Scattergl

from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import conf_logger

logger = logging.getLogger(__name__)
conf_logger(logger)


class MapFigureDisplay:
    """
    A MapFigureDisplay shows points using lat/lon as coordinates.

    Uses Scattergl (same as FigureDisplay) but with geographic coords.
    X-axis = Longitude, Y-axis = Latitude.
    """

    NUM_TRACES = 4
    VALUES_TRACE = 0
    RULES_TRACE = 1
    REGIONSET_TRACE = 2
    REGION_TRACE = 3

    def __init__(self, data_store: DataStore, selection_changed: Callable, space: str):
        self.data_store = data_store
        self.selection_changed = partial(selection_changed, self)
        self.space = space

        self.figure_data: pd.DataFrame | None = None
        self.figure: FigureWidget | None = None

        self.widget = v.Container()
        self.widget.class_ = "flex-fill"

        self._selection_mode = "lasso"
        self.first_selection = True
        self._visible = [True, False, False, False]
        self._colors: list[pd.Series | None] = [None, None, None, None]
        self.active_trace = 0
        self.initialized = False

    @property
    def dim(self):
        return 2  # Maps are always 2D

    def initialize(self, figure_data: pd.DataFrame | None):
        """Initialize with lat/lon data."""
        if figure_data is not None:
            self.figure_data = figure_data
            self.create_figure()
            self._refresh_data()
        self.initialized = True

    def create_figure(self):
        """Create a Scattergl figure with geographic layout."""
        hde_marker = {
            "color": list(self.data_store.y),
            "colorscale": "Viridis",
            "size": 4,
            "opacity": 0.7,
        }

        fig_args = {
            "x": [],
            "y": [],
            "mode": "markers",
            "marker": hde_marker,
            "customdata": [],
            "hovertemplate": "lon: %{x:.3f}<br>lat: %{y:.3f}<br>y: %{customdata:.3f}<extra></extra>",
        }

        # Create figure with 4 traces using Scattergl
        self.figure = FigureWidget(data=[Scattergl(**fig_args)])
        self.figure.add_trace(Scattergl(**fig_args))  # Rules
        self.figure.add_trace(Scattergl(**fig_args))  # Region set
        self.figure.add_trace(Scattergl(**fig_args))  # Region

        # Configure layout for geographic view
        self.figure.update_layout(
            xaxis=dict(
                title="Longitude",
                scaleanchor="y",  # Keep aspect ratio
                scaleratio=1,
            ),
            yaxis=dict(
                title="Latitude",
            ),
            margin={"t": 10, "b": 40, "l": 50, "r": 10},
            autosize=True,
            dragmode="lasso",
        )

        # Hide legend for all traces
        for trace_id in range(len(self.figure.data)):
            self.figure.data[trace_id].showlegend = False

        # Selection events
        self.figure.data[0].on_selection(partial(self._selection_event, 0))
        self.figure.data[0].on_deselect(partial(self._deselection_event, 0))
        self.figure.data[1].on_selection(partial(self._selection_event, 1))
        self.figure.data[1].on_deselect(partial(self._deselection_event, 1))

        self.widget.children = [self.figure]

    def _refresh_data(self):
        """Refresh data on all traces."""
        if self.figure is None or self.figure_data is None:
            return

        # Column 0 = lon (x), Column 1 = lat (y)
        x = self.figure_data.iloc[:, 0]  # Longitude
        y = self.figure_data.iloc[:, 1]  # Latitude
        customdata = self.data_store.y

        with self.figure.batch_update():
            for trace_id in range(self.NUM_TRACES):
                self.figure.data[trace_id].x = x
                self.figure.data[trace_id].y = y
                self.figure.data[trace_id].customdata = customdata

    def _refresh_color(self, trace_id: int):
        """Refresh colors for a trace."""
        if self.figure is None:
            return

        colors_data = self._colors[trace_id]
        if colors_data is None:
            colors_data = self.data_store.y

        with self.figure.batch_update():
            self.figure.data[trace_id].marker.color = list(colors_data)

    def _show_trace(self, trace_id: int):
        """Show only the specified trace."""
        for i in range(len(self._visible)):
            self._visible[i] = trace_id == i
            if self.figure:
                self.figure.data[i].visible = trace_id == i

    def update_X(self, X: pd.DataFrame | None):
        """Update the underlying data."""
        self.figure_data = X
        self._refresh_data()
        self._refresh_color(self.active_trace)
        if self.figure:
            self.widget.children = [self.figure]

    def display_rules(self):
        """Display rules colors."""
        self._colors[self.RULES_TRACE] = self.data_store.rule_selection_color
        self._refresh_color(self.RULES_TRACE)

    def display_regionset(self, region_set):
        """Display region set colors."""
        self._colors[self.REGIONSET_TRACE] = region_set.get_color_serie()
        self._refresh_color(self.REGIONSET_TRACE)

    def display_region(self, region):
        """Display single region colors."""
        self._colors[self.REGION_TRACE] = region.get_color_serie()
        self._refresh_color(self.REGION_TRACE)

    def display_region_value(self, region, y):
        """Display region with target colors."""
        from plotly.express.colors import sample_colorscale

        from antakia.utils.colors import colors as color_dict

        if self.figure_data is None:
            return
        if y.min() == y.max():
            y = y.copy()
            y[:] = 0.5
        else:
            y = (y + max(-y.min(), y.max())) / (2 * max(-y.min(), y.max()))

        color_serie = pd.Series(index=self.figure_data.index)
        color_serie[~region.mask] = color_dict["gray"]
        color_serie[region.mask] = sample_colorscale("Portland", y[region.mask], low=0, high=1)

        self._colors[self.REGION_TRACE] = color_serie
        self._refresh_color(self.REGION_TRACE)

    def set_color(self, color: pd.Series, trace_id: int):
        """Set color for a trace."""
        self._colors[trace_id] = color
        self._refresh_color(trace_id)

    def set_tab(self, tab):
        """Show trace for tab."""
        self._show_trace(tab)
        self.active_trace = tab
        self._refresh_color(tab)

    def disable_selection(self, is_disabled: bool):
        """Enable/disable selection."""
        self._selection_mode = False if is_disabled else "lasso"

    def rebuild(self):
        """Rebuild the figure."""
        self.create_figure()
        self._refresh_data()
        self._refresh_color(self.active_trace)
        self._show_trace(self.active_trace)

    def _selection_event(self, trace_id, trace, points, *args):
        """Handle selection event."""
        if self.figure_data is None:
            return

        if points.point_inds:
            selection_mask = pd.Series(False, index=self.figure_data.index)
            selected_indices = self.figure_data.index[points.point_inds]
            selection_mask.loc[selected_indices] = True

            if self.first_selection:
                self.data_store.selection_mask = selection_mask
            else:
                self.data_store.selection_mask = self.data_store.selection_mask | selection_mask

            self.first_selection = False
            self.selection_changed("selection_event")

    def _deselection_event(self, trace_id, trace, points, *args):
        """Handle deselection event."""
        if self.figure_data is None:
            return

        self.first_selection = True
        from antakia_core.utils import boolean_mask

        self.data_store.selection_mask = boolean_mask(self.figure_data, True)
        self.selection_changed("deselection_event")

    def display_selection(self):
        """Display selection on figure."""
        if self.figure is None or self.figure_data is None:
            return

        selection_mask = self.data_store.selection_mask
        selected_indices = selection_mask[selection_mask].index.tolist()
        point_inds = [
            self.figure_data.index.get_loc(idx)
            for idx in selected_indices
            if idx in self.figure_data.index
        ]

        with self.figure.batch_update():
            for trace_id in range(self.NUM_TRACES):
                self.figure.data[trace_id].selectedpoints = point_inds

    @property
    def display_mask(self):
        """Return display mask (all True for maps)."""
        if self.figure_data is None:
            return pd.Series()
        return pd.Series(True, index=self.figure_data.index)

    def selection_to_mask(self, row_numbers: list[int]) -> pd.Series:
        """Convert row numbers to boolean mask."""
        if self.figure_data is None:
            return pd.Series()

        from antakia_core.utils import boolean_mask, rows_to_mask

        selection = rows_to_mask(self.figure_data, row_numbers)
        if not selection.any() or selection.all():
            return boolean_mask(self.figure_data, selection.iloc[0])
        return selection
