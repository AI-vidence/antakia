from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Timeline(VuetifyWidget):

    _model_name = Unicode('TimelineModel').tag(sync=True)

    align_top = Bool(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    reverse = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['Timeline']
