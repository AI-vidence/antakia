from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ListItemIcon(VuetifyWidget):

    _model_name = Unicode('ListItemIconModel').tag(sync=True)


__all__ = ['ListItemIcon']
