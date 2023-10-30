from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Pagination(VuetifyWidget):

    _model_name = Unicode('PaginationModel').tag(sync=True)

    circle = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    length = Float(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    next_icon = Unicode(None, allow_none=True).tag(sync=True)

    prev_icon = Unicode(None, allow_none=True).tag(sync=True)

    total_visible = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    value = Float(None, allow_none=True).tag(sync=True)


__all__ = ['Pagination']
