from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from ipyvue import VueWidget
from ipywidgets.widgets.widget import widget_serialization


class VuetifyWidget(VueWidget):

    _model_name = Unicode('VuetifyWidgetModel').tag(sync=True)

    _view_name = Unicode('VuetifyView').tag(sync=True)

    _view_module = Unicode('jupyter-vuetify').tag(sync=True)

    _model_module = Unicode('jupyter-vuetify').tag(sync=True)

    _view_module_version = Unicode('^1.8.5').tag(sync=True)

    _model_module_version = Unicode('^1.8.5').tag(sync=True)

    _metadata = Dict(default_value=None, allow_none=True).tag(sync=True)


__all__ = ['VuetifyWidget']
