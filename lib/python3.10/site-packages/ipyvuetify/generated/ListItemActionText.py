from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ListItemActionText(VuetifyWidget):

    _model_name = Unicode('ListItemActionTextModel').tag(sync=True)


__all__ = ['ListItemActionText']
