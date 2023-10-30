from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class DatePickerTitle(VuetifyWidget):

    _model_name = Unicode('DatePickerTitleModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    date = Unicode(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    selecting_year = Bool(None, allow_none=True).tag(sync=True)

    value = Unicode(None, allow_none=True).tag(sync=True)

    year = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    year_icon = Unicode(None, allow_none=True).tag(sync=True)


__all__ = ['DatePickerTitle']
