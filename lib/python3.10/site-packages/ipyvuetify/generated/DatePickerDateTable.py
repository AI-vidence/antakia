from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class DatePickerDateTable(VuetifyWidget):

    _model_name = Unicode('DatePickerDateTableModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    current = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    event_color = Union([
        List(Any()),
        Dict(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    events = Union([
        List(Any()),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    first_day_of_week = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    locale = Unicode(None, allow_none=True).tag(sync=True)

    locale_first_day_of_year = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    max = Unicode(None, allow_none=True).tag(sync=True)

    min = Unicode(None, allow_none=True).tag(sync=True)

    range = Bool(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    scrollable = Bool(None, allow_none=True).tag(sync=True)

    show_week = Bool(None, allow_none=True).tag(sync=True)

    table_date = Unicode(None, allow_none=True).tag(sync=True)

    value = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['DatePickerDateTable']
