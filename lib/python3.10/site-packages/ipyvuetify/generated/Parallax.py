from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Parallax(VuetifyWidget):

    _model_name = Unicode('ParallaxModel').tag(sync=True)

    alt = Unicode(None, allow_none=True).tag(sync=True)

    height = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    src = Unicode(None, allow_none=True).tag(sync=True)


__all__ = ['Parallax']
