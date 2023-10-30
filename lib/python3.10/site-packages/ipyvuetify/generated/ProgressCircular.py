from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ProgressCircular(VuetifyWidget):

    _model_name = Unicode('ProgressCircularModel').tag(sync=True)

    button = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    indeterminate = Bool(None, allow_none=True).tag(sync=True)

    rotate = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    size = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    value = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['ProgressCircular']
