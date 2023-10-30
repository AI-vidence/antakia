from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Col(VuetifyWidget):

    _model_name = Unicode('ColModel').tag(sync=True)

    align_self = Unicode(None, allow_none=True).tag(sync=True)

    cols = Union([
        Bool(),
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    lg = Union([
        Bool(),
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    md = Union([
        Bool(),
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    offset = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    offset_lg = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    offset_md = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    offset_sm = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    offset_xl = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    order = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    order_lg = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    order_md = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    order_sm = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    order_xl = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    sm = Union([
        Bool(),
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    tag = Unicode(None, allow_none=True).tag(sync=True)

    xl = Union([
        Bool(),
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['Col']
