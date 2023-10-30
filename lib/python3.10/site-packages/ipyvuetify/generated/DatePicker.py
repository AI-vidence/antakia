from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class DatePicker(VuetifyWidget):

    _model_name = Unicode('DatePickerModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

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

    full_width = Bool(None, allow_none=True).tag(sync=True)

    header_color = Unicode(None, allow_none=True).tag(sync=True)

    landscape = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    locale = Unicode(None, allow_none=True).tag(sync=True)

    locale_first_day_of_year = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    max = Unicode(None, allow_none=True).tag(sync=True)

    min = Unicode(None, allow_none=True).tag(sync=True)

    multiple = Bool(None, allow_none=True).tag(sync=True)

    next_icon = Unicode(None, allow_none=True).tag(sync=True)

    no_title = Bool(None, allow_none=True).tag(sync=True)

    picker_date = Unicode(None, allow_none=True).tag(sync=True)

    prev_icon = Unicode(None, allow_none=True).tag(sync=True)

    range = Bool(None, allow_none=True).tag(sync=True)

    reactive = Bool(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    scrollable = Bool(None, allow_none=True).tag(sync=True)

    selected_items_text = Unicode(None, allow_none=True).tag(sync=True)

    show_current = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    show_week = Bool(None, allow_none=True).tag(sync=True)

    type = Unicode(None, allow_none=True).tag(sync=True)

    value = Union([
        List(Any()),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    year_icon = Unicode(None, allow_none=True).tag(sync=True)


__all__ = ['DatePicker']
