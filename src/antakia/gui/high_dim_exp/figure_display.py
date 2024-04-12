from __future__ import annotations

from functools import partial
from typing import Callable

import pandas as pd
import numpy as np
from antakia_core.utils import timeit, boolean_mask
from plotly.graph_objects import FigureWidget, Scattergl, Scatter3d
from plotly.express.colors import sample_colorscale
import ipyvuetify as v
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from antakia_core.data_handler import Region, RegionSet

import antakia_core.utils as utils
from antakia.config import AppConfig

import logging as logging

from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import conf_logger, Log
from antakia.utils.other_utils import NotInitialized
from antakia.utils.stats import log_errors, stats_logger
from antakia.utils.colors import colors

logger = logging.getLogger(__name__)
conf_logger(logger)


class FigureDisplay:
    """
    A FigureDisplay objet manages all operation on a scatter plot 
    This class is only responsible for displaying the provided data

    It can display in 3 or 2 dimensions.

    Attributes :

    """

    # Trace indexes : 0 for values, 1 for rules, 2 for regions
    NUM_TRACES = 4
    VALUES_TRACE = 0
    RULES_TRACE = 1
    REGIONSET_TRACE = 2
    REGION_TRACE = 3

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
            return 'values trace'
        elif trace_id == FigureDisplay.RULES_TRACE:
            return 'rules trace'
        elif trace_id == FigureDisplay.REGIONSET_TRACE:
            return 'regionset trace'
        elif trace_id == FigureDisplay.REGION_TRACE:
            return 'region trace'
        else:
            return "unknown trace"

    def __init__(self, data_store: DataStore, selection_changed: Callable,
                 space: str):
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
        self.figure_data = None

        self.space = space

        # Now we can init figure
        self.widget = v.Container()
        self.widget.class_ = "flex-fill"

        # is graph selectable
        self._selection_mode = 'lasso'
        # is this selection first since last deselection ?
        self.first_selection = True

        # traces to show
        self._visible = [True, False, False, False]
        # trace_colors
        self._colors: list[pd.Series | None] = [None, None, None, None]

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
        for i in range(len(self._visible)):
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
        color_serie[~region.mask] = colors['gray']

        # cmap = ['blue', 'green', 'red1']
        # cmap = [colors[c] for c in cmap]
        cmap = 'Portland'

        color_serie[region.mask] = sample_colorscale(cmap,
                                                     y[region.mask],
                                                     low=0,
                                                     high=1)
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

        with self.figure.batch_update():
            trace_id = self.active_trace
            self.figure.data[trace_id].x = projection[0]
            self.figure.data[trace_id].y = projection[1]
            self.figure.data[trace_id].customdata = self.data_store.y[
                self.display_mask]

            colors = self._colors[trace_id]
            if colors is None:
                colors = self.data_store.y
            colors = colors[self.display_mask]

            self.figure.data[trace_id].marker.color = colors
            if self.dim == 3:
                self.figure.data[trace_id].z = projection[2]

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
                    colors = colors[self.display_mask]
                    with self.figure.batch_update():
                        self.figure.data[trace_id].marker.color = colors

    @timeit
    def update_X(self, X: pd.DataFrame):
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
        neighbors_weight = (neighbors_weight.T /
                            neighbors_weight.sum(axis=1)).T
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
        selection = utils.rows_to_mask(self.figure_data[self.display_mask],
                                       row_numbers)
        if not selection.any() or selection.all():
            return utils.boolean_mask(self._get_figure_data(masked=False),
                                      selection.iloc[0])
        if self.display_mask.all():
            return selection

        neighbors_weight, neighbors = self._neighbors_data
        neighbors_label = np.zeros(neighbors.shape)
        for k in range(neighbors.shape[1]):
            neighbors_label[:, k] = selection.iloc[neighbors[:, k]]
        majority_label = (neighbors_label *
                          neighbors_weight).sum(axis=1).round()

        guessed_selection = pd.Series(majority_label,
                                      index=self.figure_data.index)
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
            # selection bug : we need to recreate figure in order to display the selection
            self.create_figure()
            self._refresh_data()
            # self.figure.data[trace_id].update(selectedpoints=[None])
            # self.figure.data[trace_id].selectedpoints = [None]
            self.first_selection |= self.data_store.empty_selection
            stats_logger.log(
                'hde_selection', {
                    'first_selection': str(self.first_selection),
                    'space': str(self.space),
                    'points': self.data_store.selection_mask.mean()
                })
            extrapolated_selection = self.selection_to_mask(points.point_inds)
            self.data_store.selection_mask &= extrapolated_selection
            if not self.data_store.empty_selection:
                self.selection_changed('selection_event')
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
                'hde_deselection', {
                    'first_selection': str(self.first_selection),
                    'space': str(self.space)
                })
            # We tell the GUI
            self.first_selection = False
            self.data_store.selection_mask = utils.boolean_mask(
                self.figure_data, True)
            self.selection_changed('selection_event')

    @timeit
    def display_selection(self):
        """
        display selection on figure
        Returns
        -------

        """
        with Log('display_selection ' + self.space, level=3):
            if self.dim == 2:
                fig = self.figure.data[self.active_trace]

                fig.selectedpoints = utils.mask_to_rows(
                    self.data_store.selection_mask[self.display_mask])
                # fig.update(
                #      selectedpoints=utils.mask_to_rows(self.data_store.selection_mask[self.mask]))

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
                self._display_mask = pd.Series([False] * len(self.figure_data),
                                               index=self.figure_data.index)
                indices = np.random.choice(self.figure_data.index,
                                           size=limit,
                                           replace=False)
                self._display_mask.loc[indices] = True
            else:
                self._display_mask = pd.Series([True] * len(self.figure_data),
                                               index=self.figure_data.index)
        return self._display_mask

    @timeit
    def create_figure(self):
        """
        Builds the FigureWidget for the given dimension with no data
        """
        hde_marker = {'color': self.data_store.y, 'colorscale': "Viridis"}
        if self.dim == 3:
            hde_marker['size'] = 2

        fig_args = {
            'x': [],
            'y': [],
            'mode': "markers",
            'marker': hde_marker,
            'customdata': [],
            'hovertemplate': "%{customdata:.3f}",
        }
        if self.dim == 3:
            fig_args['z'] = []
            fig_builder = Scatter3d
        else:
            fig_builder = Scattergl

        self.figure = FigureWidget(data=[fig_builder(**fig_args)
                                         ])  # Trace 0 for dots
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 1 for rules
        self.figure.add_trace(
            fig_builder(**fig_args))  # Trace 2 for region set
        self.figure.add_trace(fig_builder(**fig_args))  # Trace 3 for region

        self.figure.update_layout(dragmode=self._selection_mode)
        self.figure.update_traces(selected={"marker": {
            "opacity": 1.0
        }},
                                  unselected={"marker": {
                                      "opacity": 0.1
                                  }},
                                  selector={'type': "scatter"})
        self.figure.update_layout(
            autosize=True,
            margin={
                't': 0,
                'b': 0,
                'l': 0,
                'r': 0
            },
        )
        self.figure._config = self.figure._config | {"displaylogo": False}
        self.figure._config = self.figure._config | {'displayModeBar': True}
        # We don't want the name of the trace to appear :
        for trace_id in range(len(self.figure.data)):
            self.figure.data[trace_id].showlegend = False

        if self.dim == 2:
            # selection only on trace 0
            self.figure.data[0].on_selection(partial(self._selection_event, 0))
            self.figure.data[0].on_deselect(partial(self._deselection_event,
                                                    0))
            self.figure.data[1].on_selection(partial(self._selection_event, 1))
            self.figure.data[1].on_deselect(partial(self._deselection_event,
                                                    1))
        self.widget.children = [self.figure]

    @timeit
    def rebuild(self):
        with Log('rebuild ' + self.space, level=3):
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
