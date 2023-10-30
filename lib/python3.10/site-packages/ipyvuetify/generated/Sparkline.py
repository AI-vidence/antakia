from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Sparkline(VuetifyWidget):

    _model_name = Unicode('SparklineModel').tag(sync=True)

    auto_draw = Bool(None, allow_none=True).tag(sync=True)

    auto_draw_duration = Float(None, allow_none=True).tag(sync=True)

    auto_draw_easing = Unicode(None, allow_none=True).tag(sync=True)

    auto_line_width = Bool(None, allow_none=True).tag(sync=True)

    color = Unicode(None, allow_none=True).tag(sync=True)

    fill = Bool(None, allow_none=True).tag(sync=True)

    gradient = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    gradient_direction = Unicode(None, allow_none=True).tag(sync=True)

    height = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    label_size = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    labels = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    line_width = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    padding = Union([
        Unicode(),
        Float()
    ], default_value=None, allow_none=True).tag(sync=True)

    show_labels = Bool(None, allow_none=True).tag(sync=True)

    smooth = Union([
        Bool(),
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)

    type = Unicode(None, allow_none=True).tag(sync=True)

    value = List(Any(), default_value=None, allow_none=True).tag(sync=True)

    width = Union([
        Float(),
        Unicode()
    ], default_value=None, allow_none=True).tag(sync=True)


__all__ = ['Sparkline']
