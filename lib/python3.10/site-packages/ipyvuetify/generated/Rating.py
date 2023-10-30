from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Rating(VuetifyWidget):

    _model_name = Unicode('RatingModel').tag(sync=True)

    background_color = Unicode(None, allow_none=True).tag(sync=True)

    clearable = Bool(None, allow_none=True).tag(sync=True)

    close_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    empty_icon = Unicode(None, allow_none=True).tag(sync=True)

    full_icon = Unicode(None, allow_none=True).tag(sync=True)

    half_icon = Unicode(None, allow_none=True).tag(sync=True)

    half_increments = Bool(None, allow_none=True).tag(sync=True)

    hover = Bool(None, allow_none=True).tag(sync=True)

    large = Bool(None, allow_none=True).tag(sync=True)

    length = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    open_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    ripple = Union([
        Bool(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    size = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    small = Bool(None, allow_none=True).tag(sync=True)

    value = Float(None, allow_none=True).tag(sync=True)

    x_large = Bool(None, allow_none=True).tag(sync=True)

    x_small = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['Rating']
