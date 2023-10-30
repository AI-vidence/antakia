from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ColorPickerCanvas(VuetifyWidget):

    _model_name = Unicode('ColorPickerCanvasModel').tag(sync=True)

    color = Dict(default_value=None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    dot_size = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['ColorPickerCanvas']
