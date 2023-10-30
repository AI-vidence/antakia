from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Content(VuetifyWidget):

    _model_name = Unicode('ContentModel').tag(sync=True)

    tag = Unicode(None, allow_none=True).tag(sync=True)

    mt_auto = Bool(None, allow_none=True).tag(sync=True)

    mt_0 = Bool(None, allow_none=True).tag(sync=True)

    mt_1 = Bool(None, allow_none=True).tag(sync=True)

    mt_2 = Bool(None, allow_none=True).tag(sync=True)

    mt_3 = Bool(None, allow_none=True).tag(sync=True)

    mt_4 = Bool(None, allow_none=True).tag(sync=True)

    mt_5 = Bool(None, allow_none=True).tag(sync=True)

    mb_auto = Bool(None, allow_none=True).tag(sync=True)

    mb_0 = Bool(None, allow_none=True).tag(sync=True)

    mb_1 = Bool(None, allow_none=True).tag(sync=True)

    mb_2 = Bool(None, allow_none=True).tag(sync=True)

    mb_3 = Bool(None, allow_none=True).tag(sync=True)

    mb_4 = Bool(None, allow_none=True).tag(sync=True)

    mb_5 = Bool(None, allow_none=True).tag(sync=True)

    ml_auto = Bool(None, allow_none=True).tag(sync=True)

    ml_0 = Bool(None, allow_none=True).tag(sync=True)

    ml_1 = Bool(None, allow_none=True).tag(sync=True)

    ml_2 = Bool(None, allow_none=True).tag(sync=True)

    ml_3 = Bool(None, allow_none=True).tag(sync=True)

    ml_4 = Bool(None, allow_none=True).tag(sync=True)

    ml_5 = Bool(None, allow_none=True).tag(sync=True)

    mr_auto = Bool(None, allow_none=True).tag(sync=True)

    mr_0 = Bool(None, allow_none=True).tag(sync=True)

    mr_1 = Bool(None, allow_none=True).tag(sync=True)

    mr_2 = Bool(None, allow_none=True).tag(sync=True)

    mr_3 = Bool(None, allow_none=True).tag(sync=True)

    mr_4 = Bool(None, allow_none=True).tag(sync=True)

    mr_5 = Bool(None, allow_none=True).tag(sync=True)

    mx_auto = Bool(None, allow_none=True).tag(sync=True)

    mx_0 = Bool(None, allow_none=True).tag(sync=True)

    mx_1 = Bool(None, allow_none=True).tag(sync=True)

    mx_2 = Bool(None, allow_none=True).tag(sync=True)

    mx_3 = Bool(None, allow_none=True).tag(sync=True)

    mx_4 = Bool(None, allow_none=True).tag(sync=True)

    mx_5 = Bool(None, allow_none=True).tag(sync=True)

    my_auto = Bool(None, allow_none=True).tag(sync=True)

    my_0 = Bool(None, allow_none=True).tag(sync=True)

    my_1 = Bool(None, allow_none=True).tag(sync=True)

    my_2 = Bool(None, allow_none=True).tag(sync=True)

    my_3 = Bool(None, allow_none=True).tag(sync=True)

    my_4 = Bool(None, allow_none=True).tag(sync=True)

    my_5 = Bool(None, allow_none=True).tag(sync=True)

    ma_auto = Bool(None, allow_none=True).tag(sync=True)

    ma_0 = Bool(None, allow_none=True).tag(sync=True)

    ma_1 = Bool(None, allow_none=True).tag(sync=True)

    ma_2 = Bool(None, allow_none=True).tag(sync=True)

    ma_3 = Bool(None, allow_none=True).tag(sync=True)

    ma_4 = Bool(None, allow_none=True).tag(sync=True)

    ma_5 = Bool(None, allow_none=True).tag(sync=True)

    pt_auto = Bool(None, allow_none=True).tag(sync=True)

    pt_0 = Bool(None, allow_none=True).tag(sync=True)

    pt_1 = Bool(None, allow_none=True).tag(sync=True)

    pt_2 = Bool(None, allow_none=True).tag(sync=True)

    pt_3 = Bool(None, allow_none=True).tag(sync=True)

    pt_4 = Bool(None, allow_none=True).tag(sync=True)

    pt_5 = Bool(None, allow_none=True).tag(sync=True)

    pb_auto = Bool(None, allow_none=True).tag(sync=True)

    pb_0 = Bool(None, allow_none=True).tag(sync=True)

    pb_1 = Bool(None, allow_none=True).tag(sync=True)

    pb_2 = Bool(None, allow_none=True).tag(sync=True)

    pb_3 = Bool(None, allow_none=True).tag(sync=True)

    pb_4 = Bool(None, allow_none=True).tag(sync=True)

    pb_5 = Bool(None, allow_none=True).tag(sync=True)

    pl_auto = Bool(None, allow_none=True).tag(sync=True)

    pl_0 = Bool(None, allow_none=True).tag(sync=True)

    pl_1 = Bool(None, allow_none=True).tag(sync=True)

    pl_2 = Bool(None, allow_none=True).tag(sync=True)

    pl_3 = Bool(None, allow_none=True).tag(sync=True)

    pl_4 = Bool(None, allow_none=True).tag(sync=True)

    pl_5 = Bool(None, allow_none=True).tag(sync=True)

    pr_auto = Bool(None, allow_none=True).tag(sync=True)

    pr_0 = Bool(None, allow_none=True).tag(sync=True)

    pr_1 = Bool(None, allow_none=True).tag(sync=True)

    pr_2 = Bool(None, allow_none=True).tag(sync=True)

    pr_3 = Bool(None, allow_none=True).tag(sync=True)

    pr_4 = Bool(None, allow_none=True).tag(sync=True)

    pr_5 = Bool(None, allow_none=True).tag(sync=True)

    px_auto = Bool(None, allow_none=True).tag(sync=True)

    px_0 = Bool(None, allow_none=True).tag(sync=True)

    px_1 = Bool(None, allow_none=True).tag(sync=True)

    px_2 = Bool(None, allow_none=True).tag(sync=True)

    px_3 = Bool(None, allow_none=True).tag(sync=True)

    px_4 = Bool(None, allow_none=True).tag(sync=True)

    px_5 = Bool(None, allow_none=True).tag(sync=True)

    py_auto = Bool(None, allow_none=True).tag(sync=True)

    py_0 = Bool(None, allow_none=True).tag(sync=True)

    py_1 = Bool(None, allow_none=True).tag(sync=True)

    py_2 = Bool(None, allow_none=True).tag(sync=True)

    py_3 = Bool(None, allow_none=True).tag(sync=True)

    py_4 = Bool(None, allow_none=True).tag(sync=True)

    py_5 = Bool(None, allow_none=True).tag(sync=True)

    pa_auto = Bool(None, allow_none=True).tag(sync=True)

    pa_0 = Bool(None, allow_none=True).tag(sync=True)

    pa_1 = Bool(None, allow_none=True).tag(sync=True)

    pa_2 = Bool(None, allow_none=True).tag(sync=True)

    pa_3 = Bool(None, allow_none=True).tag(sync=True)

    pa_4 = Bool(None, allow_none=True).tag(sync=True)

    pa_5 = Bool(None, allow_none=True).tag(sync=True)


__all__ = ['Content']
