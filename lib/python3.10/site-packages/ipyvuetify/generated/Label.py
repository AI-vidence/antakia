from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Label(VuetifyWidget):

    _model_name = Unicode('LabelModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    focused = Bool(None, allow_none=True).tag(sync=True)

    for_ = Unicode(None, allow_none=True).tag(sync=True)

    left = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    right = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    value = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['Label']
