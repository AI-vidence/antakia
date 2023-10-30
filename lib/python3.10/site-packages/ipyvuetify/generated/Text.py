from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Text(VuetifyWidget):

    _model_name = Unicode('TextModel').tag(sync=True)

    value = Unicode('').tag(sync=True)


__all__ = ['Text']
