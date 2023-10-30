from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Spacer(VuetifyWidget):

    _model_name = Unicode('SpacerModel').tag(sync=True)


__all__ = ['Spacer']
