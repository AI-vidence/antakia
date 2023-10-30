from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Img(VuetifyWidget):

    _model_name = Unicode('ImgModel').tag(sync=True)

    alt = Unicode(None, allow_none=True).tag(sync=True)

    aspect_ratio = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    contain = Bool(None, allow_none=True).tag(sync=True)

    eager = Bool(None, allow_none=True).tag(sync=True)

    gradient = Unicode(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    lazy_src = Unicode(None, allow_none=True).tag(sync=True)

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

    options = Dict(default_value=None, allow_none=True).tag(sync=True)

    position = Unicode(None, allow_none=True).tag(sync=True)

    sizes = Unicode(None, allow_none=True).tag(sync=True)

    src = Union([
        Unicode(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    srcset = Unicode(None, allow_none=True).tag(sync=True)

    transition = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['Img']
