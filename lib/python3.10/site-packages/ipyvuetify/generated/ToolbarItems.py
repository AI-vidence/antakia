from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ToolbarItems(VuetifyWidget):

    _model_name = Unicode('ToolbarItemsModel').tag(sync=True)


__all__ = ['ToolbarItems']
