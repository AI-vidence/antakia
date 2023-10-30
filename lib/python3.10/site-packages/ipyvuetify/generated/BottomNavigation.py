from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class BottomNavigation(VuetifyWidget):

    _model_name = Unicode('BottomNavigationModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    active_class = Unicode(None, allow_none=True).tag(sync=True)

    app = Bool(None, allow_none=True).tag(sync=True)

    background_color = Unicode(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    fixed = Bool(None, allow_none=True).tag(sync=True)

    grow = Bool(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_on_scroll = Bool(None, allow_none=True).tag(sync=True)

    horizontal = Bool(None, allow_none=True).tag(sync=True)

    input_value = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    mandatory = Bool(None, allow_none=True).tag(sync=True)

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

    scroll_target = Unicode(None, allow_none=True).tag(sync=True)

    scroll_threshold = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    shift = Bool(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['BottomNavigation']
