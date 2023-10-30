from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Subheader(VuetifyWidget):

    _model_name = Unicode('SubheaderModel').tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    inset = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['Subheader']
