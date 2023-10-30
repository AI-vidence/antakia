from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class DataTableHeader(VuetifyWidget):

    _model_name = Unicode('DataTableHeaderModel').tag(sync=True)

    mobile = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['DataTableHeader']
