from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ListItem(VuetifyWidget):

    _model_name = Unicode('ListItemModel').tag(sync=True)

    active_class = Unicode(None, allow_none=True).tag(sync=True)

    append = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    exact = Bool(None, allow_none=True).tag(sync=True)

    exact_active_class = Unicode(None, allow_none=True).tag(sync=True)

    href = Union([
        Unicode(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    inactive = Bool(None, allow_none=True).tag(sync=True)

    input_value = Any(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    link = Bool(None, allow_none=True).tag(sync=True)

    nuxt = Bool(None, allow_none=True).tag(sync=True)

    replace = Bool(None, allow_none=True).tag(sync=True)

    ripple = Union([
        Bool(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    selectable = Bool(None, allow_none=True).tag(sync=True)

    tag = Unicode(None, allow_none=True).tag(sync=True)

    target = Unicode(None, allow_none=True).tag(sync=True)

    three_line = Bool(None, allow_none=True).tag(sync=True)

    to = Union([
        Unicode(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    two_line = Bool(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)


__all__ = ['ListItem']
