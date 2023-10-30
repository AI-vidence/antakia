from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class DataIterator(VuetifyWidget):

    _model_name = Unicode('DataIteratorModel').tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disable_filtering = Bool(None, allow_none=True).tag(sync=True)

    disable_pagination = Bool(None, allow_none=True).tag(sync=True)

    disable_sort = Bool(None, allow_none=True).tag(sync=True)

    expanded = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    footer_props = Dict(default_value=None, allow_none=True).tag(sync=True)

    group_by = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    group_desc = Union([
        Bool(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_default_footer = Bool(None, allow_none=True).tag(sync=True)

    item_key = Unicode(None, allow_none=True).tag(sync=True)

    items = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    items_per_page = Float(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    loading = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    loading_text = Unicode(None, allow_none=True).tag(sync=True)

    locale = Unicode(None, allow_none=True).tag(sync=True)

    mobile_breakpoint = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    multi_sort = Bool(None, allow_none=True).tag(sync=True)

    must_sort = Bool(None, allow_none=True).tag(sync=True)

    no_data_text = Unicode(None, allow_none=True).tag(sync=True)

    no_results_text = Unicode(None, allow_none=True).tag(sync=True)

    options = Dict(default_value=None, allow_none=True).tag(sync=True)

    page = Float(None, allow_none=True).tag(sync=True)

    search = Unicode(None, allow_none=True).tag(sync=True)

    selectable_key = Unicode(None, allow_none=True).tag(sync=True)

    server_items_length = Float(None, allow_none=True).tag(sync=True)

    single_expand = Bool(None, allow_none=True).tag(sync=True)

    single_select = Bool(None, allow_none=True).tag(sync=True)

    sort_by = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    sort_desc = Union([
        Bool(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    value = List(Any(), default_value=None, allow_none=True).tag(sync=True)


__all__ = ['DataIterator']
