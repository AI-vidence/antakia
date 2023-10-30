from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class ListItemSubtitle(VuetifyWidget):

    _model_name = Unicode('ListItemSubtitleModel').tag(sync=True)


__all__ = ['ListItemSubtitle']
