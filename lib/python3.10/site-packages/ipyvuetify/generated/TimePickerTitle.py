from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class TimePickerTitle(VuetifyWidget):

    _model_name = Unicode('TimePickerTitleModel').tag(sync=True)

    ampm = Bool(None, allow_none=True).tag(sync=True)

    ampm_readonly = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    hour = Float(None, allow_none=True).tag(sync=True)

    minute = Float(None, allow_none=True).tag(sync=True)

    period = Unicode(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    second = Float(None, allow_none=True).tag(sync=True)

    selecting = Float(None, allow_none=True).tag(sync=True)

    use_seconds = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['TimePickerTitle']
