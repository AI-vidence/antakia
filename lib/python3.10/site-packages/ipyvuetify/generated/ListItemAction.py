from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ListItemAction(VuetifyWidget):

    _model_name = Unicode('ListItemActionModel').tag(sync=True)


__all__ = ['ListItemAction']
