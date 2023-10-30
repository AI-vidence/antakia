from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class VirtualTable(VuetifyWidget):

    _model_name = Unicode('VirtualTableModel').tag(sync=True)

    chunk_size = Float(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    fixed_header = Bool(None, allow_none=True).tag(sync=True)

    header_height = Float(None, allow_none=True).tag(sync=True)

    height = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    items = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    row_height = Float(None, allow_none=True).tag(sync=True)


__all__ = ['VirtualTable']
