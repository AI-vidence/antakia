from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class SystemBar(VuetifyWidget):

    _model_name = Unicode('SystemBarModel').tag(sync=True)

    absolute = Bool(None, allow_none=True).tag(sync=True)

    app = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    fixed = Bool(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    lights_out = Bool(None, allow_none=True).tag(sync=True)

    window = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['SystemBar']
