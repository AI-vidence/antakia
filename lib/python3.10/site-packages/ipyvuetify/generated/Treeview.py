from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Treeview(VuetifyWidget):

    _model_name = Unicode('TreeviewModel').tag(sync=True)

    activatable = Bool(None, allow_none=True).tag(sync=True)

    active = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    active_class = Unicode(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    expand_icon = Unicode(None, allow_none=True).tag(sync=True)

    hoverable = Bool(None, allow_none=True).tag(sync=True)

    indeterminate_icon = Unicode(None, allow_none=True).tag(sync=True)

    item_children = Unicode(None, allow_none=True).tag(sync=True)

    item_disabled = Unicode(None, allow_none=True).tag(sync=True)

    item_key = Unicode(None, allow_none=True).tag(sync=True)

    item_text = Unicode(None, allow_none=True).tag(sync=True)

    items = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    loading_icon = Unicode(None, allow_none=True).tag(sync=True)

    multiple_active = Bool(None, allow_none=True).tag(sync=True)

    off_icon = Unicode(None, allow_none=True).tag(sync=True)

    on_icon = Unicode(None, allow_none=True).tag(sync=True)

    open_ = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    open_all = Bool(None, allow_none=True).tag(sync=True)

    open_on_click = Bool(None, allow_none=True).tag(sync=True)

    return_object = Bool(None, allow_none=True).tag(sync=True)

    rounded = Bool(None, allow_none=True).tag(sync=True)

    search = Unicode(None, allow_none=True).tag(sync=True)

    selectable = Bool(None, allow_none=True).tag(sync=True)

    selected_color = Unicode(None, allow_none=True).tag(sync=True)

    selection_type = Unicode(None, allow_none=True).tag(sync=True)

    shaped = Bool(None, allow_none=True).tag(sync=True)

    transition = Bool(None, allow_none=True).tag(sync=True)

    value = List(Any(), default_value=None, allow_none=True).tag(sync=True)


__all__ = ['Treeview']
