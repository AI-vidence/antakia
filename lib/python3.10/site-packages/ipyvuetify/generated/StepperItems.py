from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class StepperItems(VuetifyWidget):

    _model_name = Unicode('StepperItemsModel').tag(sync=True)


__all__ = ['StepperItems']
