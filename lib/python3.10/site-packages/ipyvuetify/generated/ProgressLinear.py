from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ProgressLinear(VuetifyWidget):

    _model_name = Unicode('ProgressLinearModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    active = Bool(None, allow_none=True).tag(sync=True)

    background_color = Unicode(None, allow_none=True).tag(sync=True)

    background_opacity = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    bottom = Bool(None, allow_none=True).tag(sync=True)

    buffer_value = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    fixed = Bool(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    indeterminate = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    query = Bool(None, allow_none=True).tag(sync=True)

    rounded = Bool(None, allow_none=True).tag(sync=True)

    stream = Bool(None, allow_none=True).tag(sync=True)

    striped = Bool(None, allow_none=True).tag(sync=True)

    top = Bool(None, allow_none=True).tag(sync=True)

    value = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['ProgressLinear']
