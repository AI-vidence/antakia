from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class RangeSlider(VuetifyWidget):

    _model_name = Unicode('RangeSliderModel').tag(sync=True)

    append_icon = Unicode(None, allow_none=True).tag(sync=True)

    background_color = Unicode(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    error = Bool(None, allow_none=True).tag(sync=True)

    error_count = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    error_messages = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_details = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hint = Unicode(None, allow_none=True).tag(sync=True)

    id = Unicode(None, allow_none=True).tag(sync=True)

    inverse_label = Bool(None, allow_none=True).tag(sync=True)

    label = Unicode(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    loader_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    loading = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    max = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    messages = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    min = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    persistent_hint = Bool(None, allow_none=True).tag(sync=True)

    prepend_icon = Unicode(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    rules = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    step = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    success = Bool(None, allow_none=True).tag(sync=True)

    success_messages = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    thumb_color = Unicode(None, allow_none=True).tag(sync=True)

    thumb_label = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    thumb_size = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    tick_labels = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    tick_size = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    ticks = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    track_color = Unicode(None, allow_none=True).tag(sync=True)

    track_fill_color = Unicode(None, allow_none=True).tag(sync=True)

    validate_on_blur = Bool(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)

    vertical = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['RangeSlider']
