from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ExpandTransition(VuetifyWidget):

    _model_name = Unicode('ExpandTransitionModel').tag(sync=True)

    mode = Unicode(None, allow_none=True).tag(sync=True)


__all__ = ['ExpandTransition']
