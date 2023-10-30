from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class CardActions(VuetifyWidget):

    _model_name = Unicode('CardActionsModel').tag(sync=True)


__all__ = ['CardActions']
