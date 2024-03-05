import time

import pandas as pd

from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia_core.compute.dim_reduction.dim_reduction import dim_reduc_factory
from antakia_core.data_handler.projected_values import Proj, ProjectedValues

from antakia import config
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank
from antakia.gui.helpers.progress_bar import ProgressBar
from ipywidgets import widgets
import ipyvuetify as v

from antakia_core.utils import utils

from antakia.utils.stats import stats_logger, log_errors


class ProjectedValuesSelector:
    def __init__(self, pv_bank: ProjectedValueBank, update_callback: callable, space):
        self.widget = None
        self.progress_bar = None
        self.projected_value: ProjectedValues | None = None
        self._proj_params_cards = {}
        self.update_callback = update_callback
        self.pv_bank = pv_bank
        self.space = space

        self.X = None
        self.current_proj = Proj(
            DimReducMethod.dimreduc_method_as_int(config.ATK_DEFAULT_PROJECTION),
            config.ATK_DEFAULT_DIMENSION
        )

        self._build_widget()

    @property
    def current_dim(self):
        return self.current_proj.dimension

    def _build_widget(self):
        self.widget = v.Row(children=[
            v.Select(  # Selection of proj method
                label=f"Projection in the {self.space} :",
                items=DimReducMethod.dimreduc_methods_as_str_list(),
                style_="width: 15%",
                class_="ml-2 mr-2",
            ),
            v.Menu(  # proj settings
                class_="ml-2 mr-2",
                v_slots=[
                    {
                        "name": "activator",
                        "variable": "props",
                        "children": v.Btn(
                            v_on="props.on",
                            icon=True,
                            size="x-large",
                            children=[
                                v.Icon(
                                    children=["mdi-cogs"],
                                    size="large",
                                )
                            ],
                            class_="ma-2 pa-3",
                            elevation="3",
                        ),
                    }
                ],
                children=[
                    v.Card(  # 1410
                        class_="pa-4",
                        rounded=True,
                        children=[
                            widgets.VBox(  # 14100
                                [
                                    v.Slider(  # 141000
                                        class_="ma-8 pa-2",
                                        v_model=10,
                                        min=5,
                                        max=30,
                                        step=1,
                                        label="Number of neighbours",
                                        thumb_label="always",
                                        thumb_size=25,
                                    ),
                                    v.Slider(  # 141001
                                        class_="ma-8 pa-2",
                                        v_model=0.5,
                                        min=0.1,
                                        max=0.9,
                                        step=0.1,
                                        label="MN ratio",
                                        thumb_label="always",
                                        thumb_size=25,
                                    ),
                                    v.Slider(  # 141002
                                        class_="ma-8 pa-2",
                                        v_model=2,
                                        min=0.1,
                                        max=5,
                                        step=0.1,
                                        label="FP ratio",
                                        thumb_label="always",
                                        thumb_size=25,
                                    )
                                ],
                            )
                        ],
                        min_width="500",
                    )
                ],
                v_model=False,
                close_on_content_click=False,
                offset_y=True,
            ),
            v.ProgressCircular(  # progress bar
                indeterminate=True,
                color="blue",
                width="6",
                size="35",
                class_="ml-2 mr-2 mt-2",
            )
        ])
        self.progress_bar = ProgressBar(self.widget.children[2], indeterminate=True)
        self.progress_bar.update(100, 0)
        self.projection_select.on_event("change", self.projection_select_changed)
        self.build_all_proj_param_w()
        self.projection_select.on_event("change", self.projection_select_changed)

    def initialize(self, progress_callback, X: pd.DataFrame):
        self.projected_value = self.pv_bank.get_projected_values(X)
        self.get_current_X_proj(progress_callback=progress_callback)
        self.refresh()

    def refresh(self):
        self.disable(True)
        self.projection_select.v_model = DimReducMethod.dimreduc_method_as_str(
            self.current_proj.reduction_method
        )
        self.update_proj_params_menu()
        self.update_callback()
        self.disable(False)

    def update_X(self, X: pd.DataFrame):
        self.projected_value = self.pv_bank.get_projected_values(X)
        self.refresh()

    def update_dim(self, dim):
        self.current_proj = Proj(self.current_proj.reduction_method, dim)
        self.refresh()

    @property
    def projection_select(self):
        """
        get dim reduc selector
        Returns
        -------

        """
        return self.widget.children[0]

    @property
    def projection_method(self) -> int:
        """
        returns the current projection method
        Returns
        -------

        """
        if self.projection_select.v_model == '!!disabled!!':
            self.projection_select.v_model = config.ATK_DEFAULT_PROJECTION
        return DimReducMethod.dimreduc_method_as_int(
            self.projection_select.v_model
        )

    @log_errors
    def projection_select_changed(self, *args):
        """
        callback called on projection select change
        projection is computed if needed
        Parameters
        ----------
        widget
        event
        data

        Returns
        -------

        """
        self.current_proj = Proj(self.projection_method, self.current_dim)
        self.update_proj_params_menu()
        self.refresh()

    @property
    def proj_param_widget(self):
        """
        get the projection parameter widget
        Returns
        -------

        """
        return self.widget.children[1]

    def build_proj_param_widget(self, dim_reduc) -> list[v.Slider]:
        """
        build widget
        Parameters
        ----------
        dim_reduc

        Returns
        -------

        """
        parameters = dim_reduc_factory[dim_reduc].parameters()
        sliders = []
        for param, info in parameters.items():
            min_, max_, step = utils.compute_step(info['min'], info['max'])
            default_value = info['default']
            if info['type'] == int:
                step = max(round(step), 1)

            slider = v.Slider(  # 15000
                class_="ma-8 pa-2",
                v_model=default_value,
                min=float(min_),
                max=float(max_),
                step=step,
                label=param,
                thumb_label="always",
                thumb_size=25,
            )
            slider.on_event("change", self.params_changed)
            sliders.append(slider)
        return sliders

    def update_proj_param_value(self):
        parameters = self.projected_value.get_parameters(self.current_proj)['current']
        param_widget = self._proj_params_cards[self.projection_method]
        for slider in param_widget:
            slider.v_model = parameters[slider.label]

    def build_all_proj_param_w(self):
        for dim_reduc in DimReducMethod.dimreduc_methods_as_list():
            self._proj_params_cards[dim_reduc] = self.build_proj_param_widget(dim_reduc)

    @log_errors
    def params_changed(self, widget, event, data):
        """
        called when user changes a parameter value
        Parameters
        ----------
        widget: caller widget
        event -
        data : new value

        Returns
        -------

        """
        self.update_params(widget.label, data)

    def update_params(self, parameter, new_value):
        self.projected_value.set_parameters(self.current_proj,
                                            {parameter: new_value})
        self.refresh()

    def update_proj_params_menu(self):
        """
        Called at startup by the GUI
        """
        # We return
        params = self._proj_params_cards[self.projection_method]
        # We neet to set a Card, depending on the projection method
        self.proj_param_widget.children[0].children = [widgets.VBox(params)]
        self.update_proj_param_value()

    def disable_select(self, is_disabled: bool):
        self.projection_select.disabled = is_disabled

    def disable_params(self, is_disabled: bool):
        self.proj_param_widget.disabled = is_disabled

    def disable(self, is_disabled):
        params = self._proj_params_cards[self.projection_method]
        self.disable_select(is_disabled)
        # do not enable proj parama menu if there are no parameters
        is_disabled |= len(params) == 0
        self.disable_params(is_disabled)

    def get_current_X_proj(self, dim=None, progress_callback=None) -> pd.DataFrame | None:
        """
        get current project X
        Parameters
        ----------
        dim: dimension to get, if None use current
        progress_callback: callback to publish progress to, if None use default

        Returns
        -------

        """
        if dim is None:
            dim = self.current_dim
        if progress_callback is None:
            progress_callback = self.progress_bar.update
        is_present = self.projected_value.is_present(Proj(self.current_proj.reduction_method, dim))
        t = time.time()
        X = self.projected_value.get_projection(
            Proj(self.current_proj.reduction_method, dim), progress_callback
        )
        if not is_present:
            stats_logger.log('compute_projection',
                             {'projection_method': self.current_proj.reduction_method, 'dimension': dim,
                              'compute_time': time.time() - t})

        return X

    def is_computed(self, projection_method=None, dim=None) -> bool:
        if projection_method is None:
            projection_method = self.projection_method
        if dim is None:
            dim = self.current_dim
        return self.projected_value.is_present(Proj(projection_method, dim))
