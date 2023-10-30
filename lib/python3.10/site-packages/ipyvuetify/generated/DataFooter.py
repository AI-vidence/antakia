from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class DataFooter(VuetifyWidget):

    _model_name = Unicode('DataFooterModel').tag(sync=True)

    disable_items_per_page = Bool(None, allow_none=True).tag(sync=True)

    disable_pagination = Bool(None, allow_none=True).tag(sync=True)

    first_icon = Unicode(None, allow_none=True).tag(sync=True)

    items_per_page_all_text = Unicode(None, allow_none=True).tag(sync=True)

    items_per_page_options = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    items_per_page_text = Unicode(None, allow_none=True).tag(sync=True)

    last_icon = Unicode(None, allow_none=True).tag(sync=True)

    next_icon = Unicode(None, allow_none=True).tag(sync=True)

    options = Dict(default_value=None, allow_none=True).tag(sync=True)

    page_text = Unicode(None, allow_none=True).tag(sync=True)

    pagination = Dict(default_value=None, allow_none=True).tag(sync=True)

    prev_icon = Unicode(None, allow_none=True).tag(sync=True)

    show_current_page = Bool(None, allow_none=True).tag(sync=True)

    show_first_last_page = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['DataFooter']
