from collections import namedtuple

from antakia.gui.widgets import get_widget, app_widget
from antakia_core.utils.utils import mask_to_rows
from tests.status_checks import check


class InteractionError(Exception):
    pass


@check
def select_dim(gui, dim):
    gui.set_dimension(dim)


@check
def set_color(gui, color):
    colors = ['y', 'y^', 'residual']
    get_widget(gui.widget, '11').v_model = colors[color]
    get_widget(gui.widget, '11').fire_event('change', get_widget(gui.widget, '11').v_model)


@check
def set_exp_method(gui, method):
    widget = gui.exp_values.get_explanation_select()
    methods = list(map(lambda x:x['text'], widget.items))
    exp = methods[method]
    if widget.disabled:
        raise InteractionError('exp menu disabled')

    def get_item(item_list, value):
        for item in item_list:
            if item['text'] == value:
                return item

    item = get_item(widget.items, exp)
    if item['disabled']:
        raise InteractionError('explanation method not available')
    widget.v_model = exp
    widget.fire_event('change', exp)


@check
def set_proj_method(gui, is_value_space, method):
    methods = ['PCA', 'UMAP', 'PaCMAP']
    proj = methods[method]
    if is_value_space:
        selector = gui.vs_hde.projected_value_selector
    else:
        selector = gui.vs_hde.projected_value_selector
    widget = selector.projection_select
    if widget.disabled:
        raise InteractionError('proj menu disabled')
    widget.v_model = proj
    widget.fire_event('change', proj)


@check
def edit_parameter(gui, is_value_space):
    if is_value_space:
        selector = gui.vs_hde.projected_value_selector
    else:
        selector = gui.vs_hde.projected_value_selector
    projection_param = selector.proj_param_widget
    if projection_param.disabled:
        raise InteractionError('param widget disabled')
    widget_param_list = projection_param.children[0].children[0].children
    for widget_p in widget_param_list:
        widget_p.v_model += 0.1
        widget_p.fire_event('change', widget_p.v_model)


@check
def change_tab(gui, tab):
    adresses = ['40', '41', '42']
    widget = get_widget(gui.widget, adresses[tab])
    widget.click()


@check
def select_points(gui, is_value_space, q=(1, 1)):
    if gui.tab > 1:
        raise InteractionError('wrong tab')
    X = gui.vs_hde.figure.get_X(masked=True)
    std = X.std().replace(0, 1)
    X_scaled = (X - X.mean()) / std

    b = X_scaled.iloc[:, 0] * q[0] > 0
    b &= X_scaled.iloc[:, 1] * q[1] > 0
    if is_value_space:
        hde = gui.vs_hde
    else:
        hde = gui.es_hde
    points = namedtuple('points', ['point_inds'])
    hde.figure._selection_event('', points(mask_to_rows(b)))


@check
def unselect(gui, is_value_space):
    if gui.tab > 1:
        raise InteractionError('wrong tab')
    if is_value_space:
        hde = gui.vs_hde
    else:
        hde = gui.es_hde
    hde.figure._deselection_event('', )


@check
def find_rules(gui):
    if gui.tab > 1:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "43010")
    if btn.disabled:
        raise InteractionError('skr button disabled')
    btn.click()


@check
def undo(gui):
    if gui.tab > 1:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "4302")
    if btn.disabled:
        raise InteractionError('undo button disabled')
    btn.click()


@check
def validate_rules(gui):
    if gui.tab > 1:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "43030")
    if btn.disabled:
        raise InteractionError('validate_rules button disabled')
    btn.click()


@check
def auto_cluster(gui):
    if gui.tab != 2:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "4402000")
    if btn.disabled:
        raise InteractionError('auto_cluster button disabled')
    btn.click()


@check
def clear_region_selection(gui):
    if gui.tab != 2:
        raise InteractionError('wrong tab')
    for region in gui.selected_regions.copy():
        toggle_select_region(gui, region['Region'], check=False)


@check
def toggle_select_region(gui, region_num):
    if gui.tab != 2:
        raise InteractionError('wrong tab')
    if gui.region_set.get(region_num) is None:
        raise InteractionError('unknown region')
    value = len(list(filter(lambda r: r['Region'] == region_num, gui.selected_regions))) == 0
    data = {
        'value': value,
        'item': {'Region': region_num}
    }
    gui.region_selected(data)
    if value:
        gui.selected_regions += [data['item']]
    else:
        gui.selected_regions = list(filter(lambda x: x['Region'] != region_num, gui.selected_regions))


@check
def substitute(gui):
    if gui.tab != 2:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "4401000")
    if btn.disabled:
        raise InteractionError('substitute button disabled')
    btn.click()


@check
def subdivide(gui):
    if gui.tab != 2:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "4401100")
    if btn.disabled:
        raise InteractionError('subdivide button disabled')
    btn.click()


@check
def merge(gui):
    if gui.tab != 2:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "4401200")
    if btn.disabled:
        raise InteractionError('merge button disabled')
    btn.click()


@check
def delete(gui):
    if gui.tab != 2:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "4401300")
    if btn.disabled:
        raise InteractionError('delete button disabled')
    btn.click()


@check
def select_model(gui, model):
    if gui.tab != 3:
        raise InteractionError('wrong tab')
    if len(gui.selected_regions) == 0:
        raise InteractionError('no region selected')
    region = gui.region_set.get(gui.selected_regions[0]['Region'])
    if model >= len(region.perfs):
        raise InteractionError('unknown model')
    model = region.perfs.index[model]
    data = {
        'value': True,
        'item': {'Sub-model': model}
    }
    gui.sub_model_selected_callback(data)


@check
def validate_model(gui):
    if gui.tab != 3:
        raise InteractionError('wrong tab')
    btn = get_widget(gui.widget, "450100")
    if btn.disabled:
        raise InteractionError('validate button disabled')
    btn.click()
