from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class TimePicker(VuetifyWidget):

    _model_name = Unicode('TimePickerModel').tag(sync=True)

    allowed_hours = Union([
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    allowed_minutes = Union([
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    allowed_seconds = Union([
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    ampm_in_title = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    format = Unicode(None, allow_none=True).tag(sync=True)

    full_width = Bool(None, allow_none=True).tag(sync=True)

    header_color = Unicode(None, allow_none=True).tag(sync=True)

    landscape = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    max = Unicode(None, allow_none=True).tag(sync=True)

    min = Unicode(None, allow_none=True).tag(sync=True)

    no_title = Bool(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    scrollable = Bool(None, allow_none=True).tag(sync=True)

    use_seconds = Bool(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['TimePicker']
