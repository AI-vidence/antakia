from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class CardSubtitle(VuetifyWidget):

    _model_name = Unicode('CardSubtitleModel').tag(sync=True)


__all__ = ['CardSubtitle']
