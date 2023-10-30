from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class CalendarMonthly(VuetifyWidget):

    _model_name = Unicode('CalendarMonthlyModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    end = Unicode(None, allow_none=True).tag(sync=True)

    hide_header = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    locale = Unicode(None, allow_none=True).tag(sync=True)

    min_weeks = Any(None, allow_none=True).tag(sync=True)

    now = Unicode(None, allow_none=True).tag(sync=True)

    short_months = Bool(None, allow_none=True).tag(sync=True)

    short_weekdays = Bool(None, allow_none=True).tag(sync=True)

    show_month_on_first = Bool(None, allow_none=True).tag(sync=True)

    start = Unicode(None, allow_none=True).tag(sync=True)

    weekdays = Union([
        List(Any()),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['CalendarMonthly']
