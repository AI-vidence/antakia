from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ToolbarTitle(VuetifyWidget):

    _model_name = Unicode('ToolbarTitleModel').tag(sync=True)


__all__ = ['ToolbarTitle']
