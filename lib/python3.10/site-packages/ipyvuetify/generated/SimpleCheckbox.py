from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class SimpleCheckbox(VuetifyWidget):

    _model_name = Unicode('SimpleCheckboxModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    dark = Bool(None, allow_none=True).tag(sync=True)

    disabled = Bool(None, allow_none=True).tag(sync=True)

    indeterminate = Bool(None, allow_none=True).tag(sync=True)

    indeterminate_icon = Unicode(None, allow_none=True).tag(sync=True)

    light = Bool(None, allow_none=True).tag(sync=True)

    off_icon = Unicode(None, allow_none=True).tag(sync=True)

    on_icon = Unicode(None, allow_none=True).tag(sync=True)

    ripple = Bool(None, allow_none=True).tag(sync=True)

    value = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['SimpleCheckbox']
