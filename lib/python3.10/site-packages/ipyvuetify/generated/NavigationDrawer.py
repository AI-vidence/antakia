from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class NavigationDrawer(VuetifyWidget):

    _model_name = Unicode('NavigationDrawerModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    app = Bool(None, allow_none=True).tag(sync=True)

    bottom = Bool(None, allow_none=True).tag(sync=True)

    clipped = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disable_resize_watcher = Bool(None, allow_none=True).tag(sync=True)

    disable_route_watcher = Bool(None, allow_none=True).tag(sync=True)

    expand_on_hover = Bool(None, allow_none=True).tag(sync=True)

    fixed = Bool(None, allow_none=True).tag(sync=True)

    floating = Bool(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_overlay = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    mini_variant = Bool(None, allow_none=True).tag(sync=True)

    mini_variant_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    mobile_break_point = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    overlay_color = Unicode(None, allow_none=True).tag(sync=True)

    overlay_opacity = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    permanent = Bool(None, allow_none=True).tag(sync=True)

    right = Bool(None, allow_none=True).tag(sync=True)

    src = Union([
        Unicode(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    stateless = Bool(None, allow_none=True).tag(sync=True)

    tag = Unicode(None, allow_none=True).tag(sync=True)

    temporary = Bool(None, allow_none=True).tag(sync=True)

    touchless = Bool(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['NavigationDrawer']
