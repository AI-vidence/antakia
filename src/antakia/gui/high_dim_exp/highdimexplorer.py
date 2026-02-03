from __future__ import annotations

import logging as logging
from typing import Callable

import pandas as pd
from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia_core.data_handler import ProjectedValues

from antakia.gui.helpers.data import DataStore
from antakia.gui.high_dim_exp.figure_display import FigureDisplay
from antakia.gui.high_dim_exp.map_figure import MapFigureDisplay
from antakia.gui.high_dim_exp.projected_values_selector import ProjectedValuesSelector
from antakia.utils.logging_utils import conf_logger
from antakia.utils.other_utils import NotInitialized

logger = logging.getLogger(__name__)
conf_logger(logger)

# GeoMap method ID
GEOMAP_METHOD = DimReducMethod.dimreduc_method_as_int("GeoMap")


class HighDimExplorer:
    """
    An HighDimExplorer displays one high dim Dataframes on a scatter plot.
    This class is only responsible for displaying the provided projection
    dimensionality reduction is handled by ProjectedValueSelector

    It can display in 3 or 2 dimensions.

    Implementation details :
    It handles projections computation and switch through ProjectedValueSelector
    it can update its high dim dataframe through the update_pv method

    Attributes are mostly privates (underscorred) since they are not meant to be used outside of the class.

    Attributes :

    """

    def __init__(self, data_store: DataStore, selection_changed: Callable, space: str):
        """

        Parameters
        ----------
        pv_bank: projected values storage
        selection_changed : callable called when a selection changed
        """
        self.data_store = data_store
        self.space = space
        self._selection_changed = selection_changed

        # GeoMap only makes sense for Value Space (VS), not Explanation Space (ES)
        # because ES contains SHAP values, not geographic coordinates
        self._geo_available = self._has_geo_columns() and space == "VS"

        # projected values handler & widget
        self.projected_value_selector = ProjectedValuesSelector(
            data_store.pv_bank, self.refresh, space, has_geo_columns=self._geo_available
        )

        # Create both figure types - we'll switch between them
        self._scatter_figure = FigureDisplay(data_store, selection_changed, space)
        self._map_figure = (
            MapFigureDisplay(data_store, selection_changed, space) if self._geo_available else None
        )

        # Active figure (scatter by default, or map if geo columns detected on VS)
        self._use_map = self._geo_available
        self.figure = (
            self._map_figure if self._use_map and self._map_figure else self._scatter_figure
        )

        # Container widget that can be updated when switching figure types
        import ipyvuetify as v

        self._figure_container = v.Container(class_="flex-fill", children=[self.figure.widget])

        self.initialized = False

    @property
    def figure_widget(self):
        """Return the figure container widget."""
        return self._figure_container

    def _has_geo_columns(self) -> bool:
        """Check if data has latitude/longitude columns."""
        X = self.data_store.X
        lat_col = lon_col = None
        for col in X.columns:
            col_lower = col.lower()
            if col_lower in ["latitude", "lat"]:
                lat_col = col
            elif col_lower in ["longitude", "lon", "long"]:
                lon_col = col
        return lat_col is not None and lon_col is not None

    def _switch_to_map(self, use_map: bool):
        """Switch between scatter and map figure."""
        # Can't switch to map if not available (ES or no geo columns)
        if use_map and self._map_figure is None:
            use_map = False

        if use_map == self._use_map:
            return

        self._use_map = use_map
        if use_map and self._map_figure:
            self.figure = self._map_figure
            logger.info(f"[{self.space}] Switched to Map view")
        else:
            self.figure = self._scatter_figure
            logger.info(f"[{self.space}] Switched to Scatter view")

        # Update the container widget
        self._figure_container.children = [self.figure.widget]

    def get_current_X_proj(self, dim=None, progress_callback=None) -> pd.DataFrame | None:
        return self.projected_value_selector.get_current_X_proj(dim, progress_callback)

    def initialize(self, progress_callback, X: pd.DataFrame):
        """
        inital computation (called at startup, after init to compute required values
        Parameters
        ----------
        progress_callback : callable to notify progress

        Returns
        -------

        """
        self.projected_value_selector.initialize(progress_callback, X)
        self.figure.initialize(self.get_current_X_proj())
        self.initialized = True

    def disable(self, disable_figure: bool, disable_projection: bool):
        """
        disable dropdown select
        Parameters
        ----------
        is_disabled: disable value

        Returns
        -------

        """
        self.figure.disable_selection(disable_figure)
        self.projected_value_selector.disable(disable_projection)

    def refresh(self, progress_callback=None):
        self.disable(True, True)

        # Check if we need to switch figure type (only for VS with geo columns)
        is_geomap = (
            self.projected_value_selector.projection_method == GEOMAP_METHOD
            and self._map_figure is not None
        )
        self._switch_to_map(is_geomap)

        # Get projection data
        proj_data = self.get_current_X_proj(progress_callback=progress_callback)

        # Initialize map figure if switching to it for the first time
        if is_geomap and self._map_figure and not self._map_figure.initialized:
            self._map_figure.initialize(proj_data)
        else:
            self.figure.update_X(proj_data)

        self.disable(False, False)

    def update_X(self):
        """
        changes the undelying projected value instance - update the data used in display
        Parameters
        ----------
        pv
        progress_callback

        Returns
        -------

        """
        self.projected_value_selector.update_X(self.data_store.X_exp)

    @property
    def current_pv(self) -> ProjectedValues:
        return self.projected_value_selector.projected_value

    @property
    def current_X(self) -> pd.DataFrame | None:
        """
        return hde current X value (not projected)

        Returns
        -------

        """
        if self.projected_value_selector is None:
            return (
                None  # When we're an ES HDE and no explanation have been imported nor computed yet
            )
        if self.projected_value_selector.projected_value is None:
            raise NotInitialized()
        return self.projected_value_selector.projected_value.X

    def set_tab(self, *args, **kwargs):
        return self.figure.set_tab(*args, **kwargs)

    def display_selection(self, *args, **kwargs):
        return self.figure.display_selection(*args, **kwargs)

    def set_dim(self, dim: int):
        self.projected_value_selector.update_dim(dim)
