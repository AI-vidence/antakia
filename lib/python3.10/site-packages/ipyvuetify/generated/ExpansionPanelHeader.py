from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ExpansionPanelHeader(VuetifyWidget):

    _model_name = Unicode('ExpansionPanelHeaderModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    disable_icon_rotate = Bool(None, allow_none=True).tag(sync=True)

    expand_icon = Unicode(None, allow_none=True).tag(sync=True)

    hide_actions = Bool(None, allow_none=True).tag(sync=True)

    ripple = Union([
        Bool(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['ExpansionPanelHeader']
