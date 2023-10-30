from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Badge(VuetifyWidget):

    _model_name = Unicode('BadgeModel').tag(sync=True)

    avatar = Bool(None, allow_none=True).tag(sync=True)

    bordered = Bool(None, allow_none=True).tag(sync=True)

    bottom = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    content = Any(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dot = Bool(None, allow_none=True).tag(sync=True)

    icon = Unicode(None, allow_none=True).tag(sync=True)

    inline = Bool(None, allow_none=True).tag(sync=True)

    label = Unicode(None, allow_none=True).tag(sync=True)

    left = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    mode = Unicode(None, allow_none=True).tag(sync=True)

    offset_x = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    offset_y = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    origin = Unicode(None, allow_none=True).tag(sync=True)

    overlap = Bool(None, allow_none=True).tag(sync=True)

    tile = Bool(None, allow_none=True).tag(sync=True)

    transition = Unicode(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)


__all__ = ['Badge']
