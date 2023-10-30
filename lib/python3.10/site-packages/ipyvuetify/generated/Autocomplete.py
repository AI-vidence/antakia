from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Autocomplete(VuetifyWidget):

    _model_name = Unicode('AutocompleteModel').tag(sync=True)

    allow_overflow = Bool(None, allow_none=True).tag(sync=True)

    append_icon = Unicode(None, allow_none=True).tag(sync=True)

    append_outer_icon = Unicode(None, allow_none=True).tag(sync=True)

    attach = Any(None, allow_none=True).tag(sync=True)

    auto_select_first = Bool(None, allow_none=True).tag(sync=True)

    autofocus = Bool(None, allow_none=True).tag(sync=True)

    background_color = Unicode(None, allow_none=True).tag(sync=True)

    cache_items = Bool(None, allow_none=True).tag(sync=True)

    chips = Bool(None, allow_none=True).tag(sync=True)

    clear_icon = Unicode(None, allow_none=True).tag(sync=True)

    clearable = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    counter = Union([
        Bool(),
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    deletable_chips = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    disable_lookup = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    eager = Bool(None, allow_none=True).tag(sync=True)

    error = Bool(None, allow_none=True).tag(sync=True)

    error_count = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    error_messages = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    filled = Bool(None, allow_none=True).tag(sync=True)

    flat = Bool(None, allow_none=True).tag(sync=True)

    full_width = Bool(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_details = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    hide_no_data = Bool(None, allow_none=True).tag(sync=True)

    hide_selected = Bool(None, allow_none=True).tag(sync=True)

    hint = Unicode(None, allow_none=True).tag(sync=True)

    id = Unicode(None, allow_none=True).tag(sync=True)

    item_color = Unicode(None, allow_none=True).tag(sync=True)

    item_disabled = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    item_text = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    item_value = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    items = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    label = Unicode(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    loader_height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    loading = Union([
        Bool(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    menu_props = Union([
        Unicode(),
        List(Any()),
        Dict()
    ], default_value=None, allow_none=True).tag(sync=True)

    messages = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    multiple = Bool(None, allow_none=True).tag(sync=True)

    no_data_text = Unicode(None, allow_none=True).tag(sync=True)

    no_filter = Bool(None, allow_none=True).tag(sync=True)

    open_on_clear = Bool(None, allow_none=True).tag(sync=True)

    outlined = Bool(None, allow_none=True).tag(sync=True)

    persistent_hint = Bool(None, allow_none=True).tag(sync=True)

    placeholder = Unicode(None, allow_none=True).tag(sync=True)

    prefix = Unicode(None, allow_none=True).tag(sync=True)

    prepend_icon = Unicode(None, allow_none=True).tag(sync=True)

    prepend_inner_icon = Unicode(None, allow_none=True).tag(sync=True)

    readonly = Bool(None, allow_none=True).tag(sync=True)

    return_object = Bool(None, allow_none=True).tag(sync=True)

    reverse = Bool(None, allow_none=True).tag(sync=True)

    rounded = Bool(None, allow_none=True).tag(sync=True)

    rules = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    search_input = Unicode(None, allow_none=True).tag(sync=True)

    shaped = Bool(None, allow_none=True).tag(sync=True)

    single_line = Bool(None, allow_none=True).tag(sync=True)

    small_chips = Bool(None, allow_none=True).tag(sync=True)

    solo = Bool(None, allow_none=True).tag(sync=True)

    solo_inverted = Bool(None, allow_none=True).tag(sync=True)

    success = Bool(None, allow_none=True).tag(sync=True)

    success_messages = Union([
        Unicode(),
        List(Any())
    ], default_value=None, allow_none=True).tag(sync=True)

    suffix = Unicode(None, allow_none=True).tag(sync=True)

    type = Unicode(None, allow_none=True).tag(sync=True)

    validate_on_blur = Bool(None, allow_none=True).tag(sync=True)

    value = Any(None, allow_none=True).tag(sync=True)


__all__ = ['Autocomplete']
