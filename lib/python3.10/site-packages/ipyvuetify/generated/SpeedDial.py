from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class SpeedDial(VuetifyWidget):

    _model_name = Unicode('SpeedDialModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    bottom = Bool(None, allow_none=True).tag(sync=True)

    direction = Unicode(None, allow_none=True).tag(sync=True)

    fixed = Bool(None, allow_none=True).tag(sync=True)

    left = Bool(None, allow_none=True).tag(sync=True)

    mode = Unicode(None, allow_none=True).tag(sync=True)

    open_on_hover = Bool(None, allow_none=True).tag(sync=True)

    origin = Unicode(None, allow_none=True).tag(sync=True)

    right = Bool(None, allow_none=True).tag(sync=True)

    top = Bool(None, allow_none=True).tag(sync=True)

    transition = Unicode(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)


__all__ = ['SpeedDial']
