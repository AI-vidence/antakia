from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ListItemTitle(VuetifyWidget):

    _model_name = Unicode('ListItemTitleModel').tag(sync=True)


__all__ = ['ListItemTitle']
