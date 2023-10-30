from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ListGroup(VuetifyWidget):

    _model_name = Unicode('ListGroupModel').tag(sync=True)

    active_class = Unicode(None, allow_none=True).tag(sync=True)

    append_icon = Unicode(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    eager = Bool(None, allow_none=True).tag(sync=True)

    group = Unicode(None, allow_none=True).tag(sync=True)

    no_action = Bool(None, allow_none=True).tag(sync=True)

    prepend_icon = Unicode(None, allow_none=True).tag(sync=True)

    ripple = Union([
        Bool(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    sub_group = Bool(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)


__all__ = ['ListGroup']
