from functools import wraps

from antakia import config
from antakia.compute.dim_reduction.dim_reduc_method import DimReducMethod
from antakia.compute.dim_reduction.dim_reduction import dim_reduc_factory
from antakia.gui.widgets import get_widget, app_widget


def check_dim(gui):
    dim = 3 if get_widget(app_widget.widget, '100').v_model else 2
    assert gui.vs_hde.dim == dim
    assert gui.es_hde.dim == dim


def check_hde_color(gui):
    if gui.tab == 0:
        mode = get_widget(app_widget.widget, '11').v_model
        if mode == 'y':
            c = gui.y
        elif mode == 'y^':
            c = gui.y_pred
        else:
            c = gui.y - gui.y_pred
        if gui.vs_hde._colors[gui.tab] is not None:
            assert (gui.vs_hde._colors[gui.tab] == c).all()
            assert (gui.es_hde._colors[gui.tab] == c).all()
        assert gui.vs_hde.active_tab == 0
        assert gui.es_hde.active_tab == 0
        assert gui.vs_hde._visible == [1, 0, 0, 0]
    elif gui.tab == 1:
        assert gui.vs_hde.active_tab == 1
        assert gui.es_hde.active_tab == 1
        assert gui.vs_hde._visible == [0, 1, 0, 0]
        selection = gui.selection_mask
        assert selection.mean() not in (0, 1)
        color = gui.vs_hde._colors[gui.tab]
        if color is not None:
            assert len(color[selection].unique()) <= 2
            assert len(color[~selection].unique()) <= 2
            assert len(color.unique()) <= 4
            assert (gui.vs_hde._colors[gui.tab] == gui.es_hde._colors[gui.tab]).all()
    elif gui.tab == 2:
        assert gui.vs_hde.active_tab == 2
        assert gui.es_hde.active_tab == 2
        assert gui.vs_hde._visible == [0, 0, 1, 0]

        assert (gui.region_set.get_color_serie() == gui.vs_hde._colors[2]).all()
        assert (gui.vs_hde._colors[gui.tab] == gui.es_hde._colors[gui.tab]).all()
    elif gui.tab == 3:
        assert gui.vs_hde.active_tab == 3
        assert gui.es_hde.active_tab == 3
        assert gui.vs_hde._visible == [0, 0, 0, 1]

        color = gui.vs_hde._colors[gui.tab]
        if color is not None:
            assert len(color.unique()) <= 2
            assert (gui.vs_hde._colors[gui.tab] == gui.es_hde._colors[gui.tab]).all()


def check_exp_menu(gui):
    # assert value displayed
    assert gui.exp_values.current_exp == gui.exp_values.get_explanation_select().v_model
    # assert value in hde up to date
    assert gui.es_hde.projected_value_selector.projected_value == gui.exp_values.current_pv

    # assert value enabled if computed
    select_options = gui.exp_values.get_explanation_select().items
    assert select_options[1]['disabled'] == (gui.exp_values.explanations['SHAP'] is None)
    assert select_options[2]['disabled'] == (gui.exp_values.explanations['LIME'] is None)

    # assert tab and button in same state
    assert get_widget(app_widget.widget, '130000').disabled == get_widget(app_widget.widget, '13000203').disabled
    assert get_widget(app_widget.widget, '130001').disabled == get_widget(app_widget.widget, '13000303').disabled

    # assert tab disabled if computed
    assert get_widget(app_widget.widget, '130000').disabled == (gui.exp_values.explanations['SHAP'] is not None)
    assert get_widget(app_widget.widget, '130001').disabled == (gui.exp_values.explanations['LIME'] is not None)


def check_proj_menu(gui):
    vs_pvs = gui.vs_hde.projected_value_selector
    assert vs_pvs.projection_method == vs_pvs.projected_value.current_proj.reduction_method
    assert vs_pvs.current_dim == vs_pvs.projected_value.current_proj.dimension
    assert vs_pvs.proj_param_widget.disabled == (
            (vs_pvs.projection_select.v_model == 'PCA') or (not gui.selection_mask.all() and gui.tab == 0)
    )
    assert len(get_widget(app_widget.widget, "1500").children) == len(
        dim_reduc_factory[DimReducMethod.dimreduc_method_as_int(vs_pvs.projection_select.v_model)].parameters())

    es_pvs = gui.es_hde.projected_value_selector
    assert es_pvs.projection_method == es_pvs.projected_value.current_proj.reduction_method
    assert es_pvs.current_dim == es_pvs.projected_value.current_proj.dimension
    assert es_pvs.proj_param_widget.disabled == (
            (es_pvs.projection_select.v_model == 'PCA') or (not gui.selection_mask.all() and gui.tab == 0)
    )
    assert len(get_widget(app_widget.widget, "1800").children) == len(
        dim_reduc_factory[DimReducMethod.dimreduc_method_as_int(es_pvs.projection_select.v_model)].parameters())


def check_tab_1_btn(gui):
    # data table
    assert get_widget(app_widget.widget, "4320").disabled == bool(gui.selection_mask.all())
    # skope_rule
    assert get_widget(app_widget.widget, "43010").disabled == (not gui.new_selection or bool(gui.selection_mask.all()))
    # undo
    assert get_widget(app_widget.widget, "4302").disabled == (not (gui.vs_rules_wgt.rules_num > 1))
    # validate rule
    assert get_widget(app_widget.widget, "43030").disabled == (not (gui.vs_rules_wgt.rules_num > 0))


def check_tab_2_btn(gui):
    # auto-cluster button
    assert get_widget(app_widget.widget, "4402000").disabled == False
    # auto number == num slider disabled
    assert get_widget(app_widget.widget, "4402100").disabled == get_widget(app_widget.widget, "440211").v_model
    # substitute
    assert get_widget(app_widget.widget, "4401000").disabled == (len(gui.selected_regions) != 1)
    # subdivide
    if gui.selected_regions:
        first_region = gui.region_set.get(gui.selected_regions[0]['Region'])
    else:
        first_region = None
    enable_sub = (len(gui.selected_regions) == 1) and bool(first_region.num_points() >= config.MIN_POINTS_NUMBER)
    assert get_widget(app_widget.widget, "4401100").disabled == (not enable_sub)

    enable_merge = (len(gui.selected_regions) > 1)
    get_widget(app_widget.widget, "4401200").disabled = not enable_merge
    # delete
    assert get_widget(app_widget.widget, "4401300").disabled == (len(gui.selected_regions) == 0)


def check_tab_3_btn(gui):
    assert get_widget(app_widget.widget, "450100").disabled == (
            (gui.substitute_region is None) or
            len(gui.selected_sub_model) == 0
    )


def check_all(gui):
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

