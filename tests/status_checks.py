from functools import wraps

from antakia import config
from antakia_core.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia_core.compute.dim_reduction.dim_reduction import dim_reduc_factory
from antakia.gui.widget_utils import get_widget


def check_widget(gui):
    assert get_widget(gui.widget, '120') == gui.exp_values.widget
    assert get_widget(gui.widget,
                      '130') == gui.vs_hde.projected_value_selector.widget
    assert get_widget(gui.widget, '201') == gui.vs_hde.figure.widget
    assert get_widget(gui.widget,
                      '140') == gui.es_hde.projected_value_selector.widget
    assert get_widget(gui.widget, '211') == gui.es_hde.figure.widget


def check_dim(gui):
    dim = 3 if get_widget(gui.widget, '100').v_model else 2
    assert gui.vs_hde.figure.dim == dim
    assert gui.vs_hde.projected_value_selector.current_proj.dimension == dim
    assert gui.es_hde.figure.dim == dim
    assert gui.es_hde.projected_value_selector.current_proj.dimension == dim


def check_hde_color(gui):
    if gui.tab == 0:
        mode = get_widget(gui.widget, '11').v_model
        if mode == 'y':
            c = gui.y
        elif mode == 'y^':
            c = gui.y_pred
        else:
            c = gui.y - gui.y_pred
        if gui.vs_hde.figure._colors[gui.tab] is not None:
            assert (gui.vs_hde.figure._colors[gui.tab] == c).all()
            assert (gui.es_hde.figure._colors[gui.tab] == c).all()
        assert gui.vs_hde.figure.active_trace == 0
        assert gui.es_hde.figure.active_trace == 0
        assert gui.vs_hde.figure._visible == [1, 0, 0, 0]
    elif gui.tab == 1:
        assert gui.vs_hde.figure.active_trace == 1
        assert gui.es_hde.figure.active_trace == 1
        assert gui.vs_hde.figure._visible == [0, 1, 0, 0]
        selection = gui.selection_mask
        assert selection.mean() not in (0, 1)
        color = gui.vs_hde.figure._colors[gui.tab]
        if color is not None:
            assert len(color[selection].unique()) <= 2
            assert len(color[~selection].unique()) <= 2
            assert len(color.unique()) <= 4
            assert (gui.vs_hde.figure._colors[gui.tab] ==
                    gui.es_hde.figure._colors[gui.tab]).all()
    elif gui.tab == 2:
        assert gui.vs_hde.figure.active_trace == 2
        assert gui.es_hde.figure.active_trace == 2
        assert gui.vs_hde.figure._visible == [0, 0, 1, 0]

        assert (gui.region_set.get_color_serie() ==
                gui.vs_hde.figure._colors[2]).all()
        assert (gui.vs_hde.figure._colors[gui.tab] ==
                gui.es_hde.figure._colors[gui.tab]).all()
    elif gui.tab == 3:
        assert gui.vs_hde.figure.active_trace == 3
        assert gui.es_hde.figure.active_trace == 3
        assert gui.vs_hde.figure._visible == [0, 0, 0, 1]

        color = gui.vs_hde.figure._colors[gui.tab]
        if color is not None:
            assert len(color.unique()) <= 2
            assert (gui.vs_hde.figure._colors[gui.tab] ==
                    gui.es_hde.figure._colors[gui.tab]).all()


def check_exp_menu(gui):
    # assert value displayed
    assert gui.exp_values.current_exp == gui.exp_values.get_explanation_select(
    ).v_model
    # assert value in hde up to date
    assert gui.es_hde.current_X is gui.exp_values.current_exp_df

    # assert value enabled if computed
    # select_options = gui.exp_values.get_explanation_select().items
    # assert select_options[1]['disabled'] == (gui.exp_values.explanations['SHAP'] is None)
    # assert select_options[2]['disabled'] == (gui.exp_values.explanations['LIME'] is None)

    # assert tab and button in same state
    # assert get_widget(gui.widget, '130000').disabled == get_widget(gui.widget, '13000203').disabled
    # assert get_widget(gui.widget, '130001').disabled == get_widget(gui.widget, '13000303').disabled

    # assert tab disabled if computed
    # assert get_widget(gui.widget, '130000').disabled == (gui.exp_values.explanations['SHAP'] is not None)
    # assert get_widget(gui.widget, '130001').disabled == (gui.exp_values.explanations['LIME'] is not None)


def check_proj_menu(gui):
    vs_pvs = gui.vs_hde.projected_value_selector
    vs_widget = vs_pvs.widget
    assert vs_pvs.proj_param_widget.disabled == (
        (vs_pvs.projection_select.v_model == 'PCA')
        or (not gui.selection_mask.all() and gui.tab == 0))
    assert len(get_widget(vs_widget, '100').children) == len(
        dim_reduc_factory[DimReducMethod.dimreduc_method_as_int(
            vs_pvs.projection_select.v_model)].parameters())

    es_pvs = gui.es_hde.projected_value_selector
    es_widget = es_pvs.widget
    assert es_pvs.proj_param_widget.disabled == (
        (es_pvs.projection_select.v_model == 'PCA')
        or (not gui.selection_mask.all() and gui.tab == 0))
    assert len(get_widget(es_widget, '100').children) == len(
        dim_reduc_factory[DimReducMethod.dimreduc_method_as_int(
            es_pvs.projection_select.v_model)].parameters())


def check_tab_1_btn(gui):
    # data table
    tab1 = gui.tab1

    empty_rule_set = len(tab1.vs_rules_wgt.current_rules_set) == 0
    empty_history = tab1.vs_rules_wgt.history_size <= 1

    # data table
    assert tab1.data_table.disabled == (not tab1.valid_selection)
    # self.widget[2].children[0].disabled = not self.valid_selection
    assert tab1.find_rules_btn.disabled == (not tab1.valid_selection) or (not tab1.selection_changed)
    assert tab1.undo_btn.disabled == empty_history
    assert tab1.cancel_btn.disabled == empty_rule_set and empty_history

    has_modif = (
                    tab1.vs_rules_wgt.history_size > 1
                ) or (
                    tab1.es_rules_wgt.history_size == 1 and tab1.region.num < 0  # do not validate a empty modif
                )
    assert tab1.validate_btn.disabled == (not has_modif) or empty_rule_set


def check_tab_2_btn(gui):
    # auto-cluster button
    assert gui.tab2.auto_cluster_btn.disabled == False
    # auto number == num slider disabled
    assert gui.tab2.cluster_num_wgt.disabled == gui.tab2.auto_cluster_checkbox.v_model
    # substitute
    assert gui.tab2.substitute_btn.disabled == (len(gui.tab2.selected_regions) != 1)
    assert gui.tab2.edit_btn.disabled == (len(gui.tab2.selected_regions) != 1)
    # subdivide
    if gui.tab2.selected_regions:
        first_region = gui.region_set.get(
            gui.tab2.selected_regions[0]['Region'])
    else:
        first_region = None
    enable_sub = (len(gui.tab2.selected_regions) == 1) and bool(first_region.num_points() >= config.ATK_MIN_POINTS_NUMBER)
    assert gui.tab2.divide_btn.disabled == (not enable_sub)

    enable_merge = (len(gui.tab2.selected_regions) > 1)
    gui.tab2.merge_btn.disabled = not enable_merge
    # delete
    assert gui.tab2.delete_btn.disabled == (len(gui.tab2.selected_regions) == 0)


def check_tab_3_btn(gui):
    assert gui.tab3.validate_model_btn.disabled == (
            (gui.tab3.region is None) or
            len(gui.tab3.selected_sub_model) == 0
    )


def check_all(gui):
    check_widget(gui)
    check_dim(gui)
    check_hde_color(gui)
    check_exp_menu(gui)
    check_proj_menu(gui)
    check_tab_1_btn(gui)
    check_tab_2_btn(gui)
    check_tab_3_btn(gui)


def check(method):

    @wraps(method)
    def check(gui, *args, check=False, **kw):
        result = method(gui, *args, **kw)
        if check:
            check_all(gui)
            return result

    return check
