from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ExpansionPanel(VuetifyWidget):

    _model_name = Unicode('ExpansionPanelModel').tag(sync=True)

    active_class = Unicode(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['ExpansionPanel']
