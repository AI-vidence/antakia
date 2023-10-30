from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class TabsSlider(VuetifyWidget):

    _model_name = Unicode('TabsSliderModel').tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)


__all__ = ['TabsSlider']
