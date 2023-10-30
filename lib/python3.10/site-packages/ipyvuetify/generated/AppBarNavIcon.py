from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class AppBarNavIcon(VuetifyWidget):

    _model_name = Unicode('AppBarNavIconModel').tag(sync=True)


__all__ = ['AppBarNavIcon']
