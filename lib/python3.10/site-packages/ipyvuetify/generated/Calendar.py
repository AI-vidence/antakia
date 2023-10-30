from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Calendar(VuetifyWidget):

    _model_name = Unicode('CalendarModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    end = Unicode(None, allow_none=True).tag(sync=True)

    event_color = Union([
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    event_end = Unicode(None, allow_none=True).tag(sync=True)

    event_height = Float(None, allow_none=True).tag(sync=True)

    event_margin_bottom = Float(None, allow_none=True).tag(sync=True)

    event_more = Bool(None, allow_none=True).tag(sync=True)

    event_more_text = Unicode(None, allow_none=True).tag(sync=True)

    event_name = Union([
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    event_overlap_mode = Union([
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    event_overlap_threshold = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    event_ripple = Union([
        Bool(),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    event_start = Unicode(None, allow_none=True).tag(sync=True)

    event_text_color = Union([
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    events = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    first_interval = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_header = Bool(None, allow_none=True).tag(sync=True)

    interval_count = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    interval_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    interval_minutes = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    interval_width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    locale = Unicode(None, allow_none=True).tag(sync=True)

    max_days = Float(None, allow_none=True).tag(sync=True)

    min_weeks = Any(None, allow_none=True).tag(sync=True)

    now = Unicode(None, allow_none=True).tag(sync=True)

    short_intervals = Bool(None, allow_none=True).tag(sync=True)

    short_months = Bool(None, allow_none=True).tag(sync=True)

    short_weekdays = Bool(None, allow_none=True).tag(sync=True)

    show_month_on_first = Bool(None, allow_none=True).tag(sync=True)

    start = Unicode(None, allow_none=True).tag(sync=True)

    type = Unicode(None, allow_none=True).tag(sync=True)

    value = Unicode(None, allow_none=True).tag(sync=True)

    weekdays = Union([
        List(Any()),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['Calendar']
