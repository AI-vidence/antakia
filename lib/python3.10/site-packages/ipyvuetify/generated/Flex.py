from traitlets import (
    Unicode, Enum, Instance, Union, Float, Int, List, Tuple, Dict,
    Undefined, Bool, Any
)

from .VuetifyWidget import VuetifyWidget


class Flex(VuetifyWidget):

    _model_name = Unicode('FlexModel').tag(sync=True)

    xs1 = Bool(None, allow_none=True).tag(sync=True)

    xs2 = Bool(None, allow_none=True).tag(sync=True)

    xs3 = Bool(None, allow_none=True).tag(sync=True)

    xs4 = Bool(None, allow_none=True).tag(sync=True)

    xs5 = Bool(None, allow_none=True).tag(sync=True)

    xs6 = Bool(None, allow_none=True).tag(sync=True)

    xs7 = Bool(None, allow_none=True).tag(sync=True)

    xs8 = Bool(None, allow_none=True).tag(sync=True)

    xs9 = Bool(None, allow_none=True).tag(sync=True)

    xs10 = Bool(None, allow_none=True).tag(sync=True)

    xs11 = Bool(None, allow_none=True).tag(sync=True)

    xs12 = Bool(None, allow_none=True).tag(sync=True)

    sm1 = Bool(None, allow_none=True).tag(sync=True)

    sm2 = Bool(None, allow_none=True).tag(sync=True)

    sm3 = Bool(None, allow_none=True).tag(sync=True)

    sm4 = Bool(None, allow_none=True).tag(sync=True)

    sm5 = Bool(None, allow_none=True).tag(sync=True)

    sm6 = Bool(None, allow_none=True).tag(sync=True)

    sm7 = Bool(None, allow_none=True).tag(sync=True)

    sm8 = Bool(None, allow_none=True).tag(sync=True)

    sm9 = Bool(None, allow_none=True).tag(sync=True)

    sm10 = Bool(None, allow_none=True).tag(sync=True)

    sm11 = Bool(None, allow_none=True).tag(sync=True)

    sm12 = Bool(None, allow_none=True).tag(sync=True)

    md1 = Bool(None, allow_none=True).tag(sync=True)

    md2 = Bool(None, allow_none=True).tag(sync=True)

    md3 = Bool(None, allow_none=True).tag(sync=True)

    md4 = Bool(None, allow_none=True).tag(sync=True)

    md5 = Bool(None, allow_none=True).tag(sync=True)

    md6 = Bool(None, allow_none=True).tag(sync=True)

    md7 = Bool(None, allow_none=True).tag(sync=True)

    md8 = Bool(None, allow_none=True).tag(sync=True)

    md9 = Bool(None, allow_none=True).tag(sync=True)

    md10 = Bool(None, allow_none=True).tag(sync=True)

    md11 = Bool(None, allow_none=True).tag(sync=True)

    md12 = Bool(None, allow_none=True).tag(sync=True)

    lg1 = Bool(None, allow_none=True).tag(sync=True)

    lg2 = Bool(None, allow_none=True).tag(sync=True)

    lg3 = Bool(None, allow_none=True).tag(sync=True)

    lg4 = Bool(None, allow_none=True).tag(sync=True)

    lg5 = Bool(None, allow_none=True).tag(sync=True)

    lg6 = Bool(None, allow_none=True).tag(sync=True)

    lg7 = Bool(None, allow_none=True).tag(sync=True)

    lg8 = Bool(None, allow_none=True).tag(sync=True)

    lg9 = Bool(None, allow_none=True).tag(sync=True)

    lg10 = Bool(None, allow_none=True).tag(sync=True)

    lg11 = Bool(None, allow_none=True).tag(sync=True)

    lg12 = Bool(None, allow_none=True).tag(sync=True)

    xl1 = Bool(None, allow_none=True).tag(sync=True)

    xl2 = Bool(None, allow_none=True).tag(sync=True)

    xl3 = Bool(None, allow_none=True).tag(sync=True)

    xl4 = Bool(None, allow_none=True).tag(sync=True)

    xl5 = Bool(None, allow_none=True).tag(sync=True)

    xl6 = Bool(None, allow_none=True).tag(sync=True)

    xl7 = Bool(None, allow_none=True).tag(sync=True)

    xl8 = Bool(None, allow_none=True).tag(sync=True)

    xl9 = Bool(None, allow_none=True).tag(sync=True)

    xl10 = Bool(None, allow_none=True).tag(sync=True)

    xl11 = Bool(None, allow_none=True).tag(sync=True)

    xl12 = Bool(None, allow_none=True).tag(sync=True)

    align_self_baseline = Bool(None, allow_none=True).tag(sync=True)

    align_self_center = Bool(None, allow_none=True).tag(sync=True)

    align_self_end = Bool(None, allow_none=True).tag(sync=True)

    align_self_start = Bool(None, allow_none=True).tag(sync=True)

    grow = Bool(None, allow_none=True).tag(sync=True)

    id = Unicode(None, allow_none=True).tag(sync=True)

    offset_xs0 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs1 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs2 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs3 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs4 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs5 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs6 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs7 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs8 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs9 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs10 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs11 = Bool(None, allow_none=True).tag(sync=True)

    offset_xs12 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm0 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm1 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm2 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm3 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm4 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm5 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm6 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm7 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm8 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm9 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm10 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm11 = Bool(None, allow_none=True).tag(sync=True)

    offset_sm12 = Bool(None, allow_none=True).tag(sync=True)

    offset_md0 = Bool(None, allow_none=True).tag(sync=True)

    offset_md1 = Bool(None, allow_none=True).tag(sync=True)

    offset_md2 = Bool(None, allow_none=True).tag(sync=True)

    offset_md3 = Bool(None, allow_none=True).tag(sync=True)

    offset_md4 = Bool(None, allow_none=True).tag(sync=True)

    offset_md5 = Bool(None, allow_none=True).tag(sync=True)

    offset_md6 = Bool(None, allow_none=True).tag(sync=True)

    offset_md7 = Bool(None, allow_none=True).tag(sync=True)

    offset_md8 = Bool(None, allow_none=True).tag(sync=True)

    offset_md9 = Bool(None, allow_none=True).tag(sync=True)

    offset_md10 = Bool(None, allow_none=True).tag(sync=True)

    offset_md11 = Bool(None, allow_none=True).tag(sync=True)

    offset_md12 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg0 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg1 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg2 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg3 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg4 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg5 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg6 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg7 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg8 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg9 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg10 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg11 = Bool(None, allow_none=True).tag(sync=True)

    offset_lg12 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl0 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl1 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl2 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl3 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl4 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl5 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl6 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl7 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl8 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl9 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl10 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl11 = Bool(None, allow_none=True).tag(sync=True)

    offset_xl12 = Bool(None, allow_none=True).tag(sync=True)

    order_xs1 = Bool(None, allow_none=True).tag(sync=True)

    order_xs2 = Bool(None, allow_none=True).tag(sync=True)

    order_xs3 = Bool(None, allow_none=True).tag(sync=True)

    order_xs4 = Bool(None, allow_none=True).tag(sync=True)

    order_xs5 = Bool(None, allow_none=True).tag(sync=True)

    order_xs6 = Bool(None, allow_none=True).tag(sync=True)

    order_xs7 = Bool(None, allow_none=True).tag(sync=True)

    order_xs8 = Bool(None, allow_none=True).tag(sync=True)

    order_xs9 = Bool(None, allow_none=True).tag(sync=True)

    order_xs10 = Bool(None, allow_none=True).tag(sync=True)

    order_xs11 = Bool(None, allow_none=True).tag(sync=True)

    order_xs12 = Bool(None, allow_none=True).tag(sync=True)

    order_sm1 = Bool(None, allow_none=True).tag(sync=True)

    order_sm2 = Bool(None, allow_none=True).tag(sync=True)

    order_sm3 = Bool(None, allow_none=True).tag(sync=True)

    order_sm4 = Bool(None, allow_none=True).tag(sync=True)

    order_sm5 = Bool(None, allow_none=True).tag(sync=True)

    order_sm6 = Bool(None, allow_none=True).tag(sync=True)

    order_sm7 = Bool(None, allow_none=True).tag(sync=True)

    order_sm8 = Bool(None, allow_none=True).tag(sync=True)

    order_sm9 = Bool(None, allow_none=True).tag(sync=True)

    order_sm10 = Bool(None, allow_none=True).tag(sync=True)

    order_sm11 = Bool(None, allow_none=True).tag(sync=True)

    order_sm12 = Bool(None, allow_none=True).tag(sync=True)

    order_md1 = Bool(None, allow_none=True).tag(sync=True)

    order_md2 = Bool(None, allow_none=True).tag(sync=True)

    order_md3 = Bool(None, allow_none=True).tag(sync=True)

    order_md4 = Bool(None, allow_none=True).tag(sync=True)

    order_md5 = Bool(None, allow_none=True).tag(sync=True)

    order_md6 = Bool(None, allow_none=True).tag(sync=True)

    order_md7 = Bool(None, allow_none=True).tag(sync=True)

    order_md8 = Bool(None, allow_none=True).tag(sync=True)

    order_md9 = Bool(None, allow_none=True).tag(sync=True)

    order_md10 = Bool(None, allow_none=True).tag(sync=True)

    order_md11 = Bool(None, allow_none=True).tag(sync=True)

    order_md12 = Bool(None, allow_none=True).tag(sync=True)

    order_lg1 = Bool(None, allow_none=True).tag(sync=True)

    order_lg2 = Bool(None, allow_none=True).tag(sync=True)

    order_lg3 = Bool(None, allow_none=True).tag(sync=True)

    order_lg4 = Bool(None, allow_none=True).tag(sync=True)

    order_lg5 = Bool(None, allow_none=True).tag(sync=True)

    order_lg6 = Bool(None, allow_none=True).tag(sync=True)

    order_lg7 = Bool(None, allow_none=True).tag(sync=True)

    order_lg8 = Bool(None, allow_none=True).tag(sync=True)

    order_lg9 = Bool(None, allow_none=True).tag(sync=True)

    order_lg10 = Bool(None, allow_none=True).tag(sync=True)

    order_lg11 = Bool(None, allow_none=True).tag(sync=True)

    order_lg12 = Bool(None, allow_none=True).tag(sync=True)

    order_xl1 = Bool(None, allow_none=True).tag(sync=True)

    order_xl2 = Bool(None, allow_none=True).tag(sync=True)

    order_xl3 = Bool(None, allow_none=True).tag(sync=True)

    order_xl4 = Bool(None, allow_none=True).tag(sync=True)

    order_xl5 = Bool(None, allow_none=True).tag(sync=True)

    order_xl6 = Bool(None, allow_none=True).tag(sync=True)

    order_xl7 = Bool(None, allow_none=True).tag(sync=True)

    order_xl8 = Bool(None, allow_none=True).tag(sync=True)

    order_xl9 = Bool(None, allow_none=True).tag(sync=True)

    order_xl10 = Bool(None, allow_none=True).tag(sync=True)

    order_xl11 = Bool(None, allow_none=True).tag(sync=True)

    order_xl12 = Bool(None, allow_none=True).tag(sync=True)

    shrink = Bool(None, allow_none=True).tag(sync=True)

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


__all__ = ['Flex']
