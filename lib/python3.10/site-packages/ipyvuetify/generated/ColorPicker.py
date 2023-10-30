from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ColorPicker(VuetifyWidget):

    _model_name = Unicode('ColorPickerModel').tag(sync=True)

    canvas_height = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    dot_size = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    flat = Bool(None, allow_none=True).tag(sync=True)

    hide_canvas = Bool(None, allow_none=True).tag(sync=True)

    hide_inputs = Bool(None, allow_none=True).tag(sync=True)

    hide_mode_switch = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    mode = Unicode(None, allow_none=True).tag(sync=True)

    show_swatches = Bool(None, allow_none=True).tag(sync=True)

    swatches = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    swatches_max_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    value = Union([
        Dict(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['ColorPicker']
