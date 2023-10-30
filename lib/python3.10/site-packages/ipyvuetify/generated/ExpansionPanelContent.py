from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ExpansionPanelContent(VuetifyWidget):

    _model_name = Unicode('ExpansionPanelContentModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    eager = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['ExpansionPanelContent']
