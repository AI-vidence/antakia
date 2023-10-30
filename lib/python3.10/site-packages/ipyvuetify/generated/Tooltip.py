from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Tooltip(VuetifyWidget):

    _model_name = Unicode('TooltipModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    activator = Any(None, allow_none=True).tag(sync=True)

    allow_overflow = Bool(None, allow_none=True).tag(sync=True)

    attach = Any(None, allow_none=True).tag(sync=True)

    bottom = Bool(None, allow_none=True).tag(sync=True)

    close_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    content_class = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    eager = Bool(None, allow_none=True).tag(sync=True)

    fixed = Bool(None, allow_none=True).tag(sync=True)

    internal_activator = Bool(None, allow_none=True).tag(sync=True)

    left = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    max_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    min_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    nudge_bottom = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    nudge_left = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    nudge_right = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    nudge_top = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    nudge_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    offset_overflow = Bool(None, allow_none=True).tag(sync=True)

    open_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    open_on_click = Bool(None, allow_none=True).tag(sync=True)

    open_on_hover = Bool(None, allow_none=True).tag(sync=True)

    position_x = Float(None, allow_none=True).tag(sync=True)

    position_y = Float(None, allow_none=True).tag(sync=True)

    right = Bool(None, allow_none=True).tag(sync=True)

    tag = Unicode(None, allow_none=True).tag(sync=True)

    top = Bool(None, allow_none=True).tag(sync=True)

    transition = Unicode(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)

    z_index = Any(None, allow_none=True).tag(sync=True)


__all__ = ['Tooltip']
