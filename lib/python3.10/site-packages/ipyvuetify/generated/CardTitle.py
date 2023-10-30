from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class CardTitle(VuetifyWidget):

    _model_name = Unicode('CardTitleModel').tag(sync=True)


__all__ = ['CardTitle']
