from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ColorPickerSwatches(VuetifyWidget):

    _model_name = Unicode('ColorPickerSwatchesModel').tag(sync=True)

    color = Dict(default_value=None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    max_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    max_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    swatches = List(Any(), default_value=None, allow_none=True).tag(sync=True)


__all__ = ['ColorPickerSwatches']
