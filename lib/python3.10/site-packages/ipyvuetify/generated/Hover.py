from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Hover(VuetifyWidget):

    _model_name = Unicode('HoverModel').tag(sync=True)

    close_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    open_delay = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    value = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['Hover']
