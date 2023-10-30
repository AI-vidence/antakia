from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class EditDialog(VuetifyWidget):

    _model_name = Unicode('EditDialogModel').tag(sync=True)

    cancel_text = Any(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    eager = Bool(None, allow_none=True).tag(sync=True)

    large = Bool(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    persistent = Bool(None, allow_none=True).tag(sync=True)

    return_value = Any(None, allow_none=True).tag(sync=True)

    save_text = Any(None, allow_none=True).tag(sync=True)

    transition = Unicode(None, allow_none=True).tag(sync=True)


__all__ = ['EditDialog']
