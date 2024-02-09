import pandas as pd

from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.dim_reduction.dim_reduction import dim_reduc_factory
from antakia.data_handler.projected_values import ProjectedValues, Proj
from antakia.gui.progress_bar import ProgressBar
from antakia.gui.widgets import get_widget, app_widget
from ipywidgets import widgets
import ipyvuetify as v

from antakia.utils import utils


class ProjectedValueSelector:
    def __init__(self, is_value_space: bool, dim: int, update_callback: callable, projected_value: ProjectedValues):
        self._proj_params_cards = {}
        self.current_dim = dim
        self.is_value_space = is_value_space
        self.update_callback = update_callback
        self.projected_value = projected_value

        self.progress_bar = ProgressBar(
            get_widget(app_widget.widget, "16" if self.is_value_space else "19"),
            True
        )
        self.progress_bar.update(100, 0)

        self.build_all_proj_param_w()
        self.projection_select.on_event("change", self.projection_select_changed)

    def initialize(self, progress_callback, pv: ProjectedValues | None):
        if pv is not None:
            # for ES space we need to manually provide the pv value in other to not trigger all auto updates
            self.projected_value = pv
        self.get_current_X_proj(progress_callback=progress_callback)
        self.refresh()

    def refresh(self):
        self.disable(True)
        self.projection_select.v_model = DimReducMethod.dimreduc_method_as_str(
            self.projected_value.current_proj.reduction_method
        )
        self.update_proj_params_menu()
        self.update_callback()
        self.disable(False)

    def update_projected_value(self, new_projected_value: ProjectedValues):
        self.projected_value = new_projected_value
        self.refresh()

    def update_dim(self, dim):
        self.current_dim = dim
        self.refresh()

    @property
    def projection_select(self):
        """
        get dim reduc selector
        Returns
        -------

        """
        return get_widget(app_widget.widget, "14" if self.is_value_space else "17")

    @property
    def projection_method(self) -> int:
        """
        returns the current projection method
        Returns
        -------

        """
        if self.projection_select.v_model == '!!disabled!!':
            self.projection_select.v_model = DimReducMethod.default_projection_as_str()
            return DimReducMethod.default_projection_as_int()
        return DimReducMethod.dimreduc_method_as_int(
            self.projection_select.v_model
        )

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
        self.projected_value.current_proj = Proj(self.projection_method, self.current_dim)
        self.update_proj_params_menu()
        self.refresh()

    @property
    def proj_param_widget(self):
        """
        get the projection parameter widget
        Returns
        -------

        """
        return get_widget(app_widget.widget, "15" if self.is_value_space else "18")

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
        parameters = self.projected_value.get_parameters(self.projection_method, self.current_dim)['current']
        param_widget = self._proj_params_cards[self.projection_method]
        for slider in param_widget:
            slider.v_model = parameters[slider.label]

    def build_all_proj_param_w(self):
        for dim_reduc in DimReducMethod.dimreduc_methods_as_list():
            self._proj_params_cards[dim_reduc] = self.build_proj_param_widget(dim_reduc)

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
        self.projected_value.set_parameters(self.projection_method, self.current_dim,
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
        set_current = dim == self.current_dim
        if progress_callback is None:
            progress_callback = self.progress_bar.update
        X = self.projected_value.get_projection(
            self.projection_method, dim, progress_callback, set_current
        )
        return X

    def is_computed(self, projection_method=None, dim=None) -> bool:
        if projection_method is None:
            projection_method = self.projection_method
        if dim is None:
            dim = self.current_dim
        return self.projected_value.is_present(projection_method, dim)
