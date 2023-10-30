from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Row(VuetifyWidget):

    _model_name = Unicode('RowModel').tag(sync=True)

    align = Unicode(None, allow_none=True).tag(sync=True)

    align_content = Unicode(None, allow_none=True).tag(sync=True)

    align_content_lg = Unicode(None, allow_none=True).tag(sync=True)

    align_content_md = Unicode(None, allow_none=True).tag(sync=True)

    align_content_sm = Unicode(None, allow_none=True).tag(sync=True)

    align_content_xl = Unicode(None, allow_none=True).tag(sync=True)

    align_lg = Unicode(None, allow_none=True).tag(sync=True)

    align_md = Unicode(None, allow_none=True).tag(sync=True)

    align_sm = Unicode(None, allow_none=True).tag(sync=True)

    align_xl = Unicode(None, allow_none=True).tag(sync=True)

    dense = Bool(None, allow_none=True).tag(sync=True)

    justify = Unicode(None, allow_none=True).tag(sync=True)

    justify_lg = Unicode(None, allow_none=True).tag(sync=True)

    justify_md = Unicode(None, allow_none=True).tag(sync=True)

    justify_sm = Unicode(None, allow_none=True).tag(sync=True)

    justify_xl = Unicode(None, allow_none=True).tag(sync=True)

    no_gutters = Bool(None, allow_none=True).tag(sync=True)

    tag = Unicode(None, allow_none=True).tag(sync=True)


__all__ = ['Row']
