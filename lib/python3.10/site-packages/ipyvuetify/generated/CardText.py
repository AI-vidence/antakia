from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class CardText(VuetifyWidget):

    _model_name = Unicode('CardTextModel').tag(sync=True)


__all__ = ['CardText']
