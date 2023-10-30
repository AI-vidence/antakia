from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Dialog(VuetifyWidget):

    _model_name = Unicode('DialogModel').tag(sync=True)

    activator = Any(None, allow_none=True).tag(sync=True)

    attach = Any(None, allow_none=True).tag(sync=True)

    close_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    content_class = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    eager = Bool(None, allow_none=True).tag(sync=True)

    fullscreen = Bool(None, allow_none=True).tag(sync=True)

    hide_overlay = Bool(None, allow_none=True).tag(sync=True)

    internal_activator = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    max_width = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    no_click_animation = Bool(None, allow_none=True).tag(sync=True)

    open_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    open_on_hover = Bool(None, allow_none=True).tag(sync=True)

    origin = Unicode(None, allow_none=True).tag(sync=True)

    overlay_color = Unicode(None, allow_none=True).tag(sync=True)

    overlay_opacity = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    persistent = Bool(None, allow_none=True).tag(sync=True)

    retain_focus = Bool(None, allow_none=True).tag(sync=True)

    return_value = Any(None, allow_none=True).tag(sync=True)

    scrollable = Bool(None, allow_none=True).tag(sync=True)

    transition = Union([
        Unicode(),
        Bool()
    ], default_value=None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)

    width = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['Dialog']
