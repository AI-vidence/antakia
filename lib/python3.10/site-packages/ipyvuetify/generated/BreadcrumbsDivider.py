from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class BreadcrumbsDivider(VuetifyWidget):

    _model_name = Unicode('BreadcrumbsDividerModel').tag(sync=True)


__all__ = ['BreadcrumbsDivider']
