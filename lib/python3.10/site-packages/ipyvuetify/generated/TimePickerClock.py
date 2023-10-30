from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class TimePickerClock(VuetifyWidget):

    _model_name = Unicode('TimePickerClockModel').tag(sync=True)

    ampm = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    double = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    max = Float(None, allow_none=True).tag(sync=True)

    min = Float(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    rotate = Float(None, allow_none=True).tag(sync=True)

    scrollable = Bool(None, allow_none=True).tag(sync=True)

    step = Float(None, allow_none=True).tag(sync=True)

    value = Float(None, allow_none=True).tag(sync=True)


__all__ = ['TimePickerClock']
