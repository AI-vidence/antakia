from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class AppBar(VuetifyWidget):

    _model_name = Unicode('AppBarModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    app = Bool(None, allow_none=True).tag(sync=True)

    bottom = Bool(None, allow_none=True).tag(sync=True)

    clipped_left = Bool(None, allow_none=True).tag(sync=True)

    clipped_right = Bool(None, allow_none=True).tag(sync=True)

    collapse = Bool(None, allow_none=True).tag(sync=True)

    collapse_on_scroll = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    elevate_on_scroll = Bool(None, allow_none=True).tag(sync=True)

    elevation = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    extended = Bool(None, allow_none=True).tag(sync=True)

    extension_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    fade_img_on_scroll = Bool(None, allow_none=True).tag(sync=True)

    fixed = Bool(None, allow_none=True).tag(sync=True)

    flat = Bool(None, allow_none=True).tag(sync=True)

    floating = Bool(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_on_scroll = Bool(None, allow_none=True).tag(sync=True)

    inverted_scroll = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    max_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    max_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    min_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    min_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    prominent = Bool(None, allow_none=True).tag(sync=True)

    scroll_off_screen = Bool(None, allow_none=True).tag(sync=True)

    scroll_target = Unicode(None, allow_none=True).tag(sync=True)

    scroll_threshold = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    short = Bool(None, allow_none=True).tag(sync=True)

    shrink_on_scroll = Bool(None, allow_none=True).tag(sync=True)

    src = Union([
        Unicode(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    tag = Unicode(None, allow_none=True).tag(sync=True)

    tile = Bool(None, allow_none=True).tag(sync=True)

    value = Bool(None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['AppBar']
