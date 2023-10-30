from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class TimelineItem(VuetifyWidget):

    _model_name = Unicode('TimelineItemModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    fill_dot = Bool(None, allow_none=True).tag(sync=True)

    hide_dot = Bool(None, allow_none=True).tag(sync=True)

    icon = Unicode(None, allow_none=True).tag(sync=True)

    icon_color = Unicode(None, allow_none=True).tag(sync=True)

    large = Bool(None, allow_none=True).tag(sync=True)

    left = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    right = Bool(None, allow_none=True).tag(sync=True)

    small = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['TimelineItem']
