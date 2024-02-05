from collections import namedtuple

from antakia.gui.widgets import get_widget, app_widget
from antakia.utils.utils import mask_to_rows


def select_dim(gui, dim):
    gui.set_dimension(dim)


def set_color(gui, color):
    colors = ['y', 'y^', 'residual']
    get_widget(app_widget, '11').v_model = colors[color]
    get_widget(app_widget, '11').fire_event('change')


def set_exp_method(gui, method):
    methods = ['Imported', 'SHAP', 'LIME']
    exp = methods[method]
    widget = gui.exp_values.get_explanation_select()

    def get_item(item_list, value):
        for item in item_list:
            if item['text'] == value:
                return item

    item = get_item(widget.items, exp)
    if item['disabled']:
        raise ValueError('explanation method not available')
    widget.v_model = exp
    widget.fire_event('change', exp)


def compute_exp_method(gui, method):
    methods = ['Imported', 'SHAP', 'LIME']
    exp = methods[method]
    if exp == 'Imported':
        raise ValueError('cannot compute imported')
    if exp == "SHAP":
        btn = '13000203'
    else:
        btn = '13000303'
    btn = get_widget(app_widget, btn)
    if btn.disabled:
        raise ValueError('value already computed')
    btn.click()


def set_proj_method(gui, is_value_space, method):
    methods = ['PCA', 'UMAP', 'PaCMAP']
    proj = methods[method]
    if is_value_space:
        widget = get_widget(app_widget, '14')
    else:
        widget = get_widget(app_widget, '17')
    widget.v_model = proj
    widget.fire_event('change')


def edit_parameter(gui, is_value_space):
    if is_value_space:
        param = get_widget(app_widget, '15')
    else:
        param = get_widget(app_widget, '18')
    widget_param_list = param.children[0].children[0].children
    for widget_p in widget_param_list:
        widget_p.v_model += 0.1
        widget_p.fire_event('change')


def change_tab(gui, tab):
    adresses = ['40', '41', '42']
    widget = get_widget(app_widget, adresses[tab])
    widget.click()


def select(gui, is_value_space, q=(1, 1)):
    X = gui.vs_hde.get_current_X_proj(dim=2)
    b = (X.iloc[:, 0] / X.iloc[:, 0].max()) * q[0] > 0.5
    b &= (X.iloc[:, 1] / X.iloc[:, 1].max()) * q[1] > 0.5
    if is_value_space:
        hde = gui.vs_hde
    else:
        hde = gui.es_hde
    points = namedtuple('points', ['point_inds'])
    hde._selection_event('', points(mask_to_rows(b)))


def unselect(gui, is_value_space):
    if is_value_space:
        hde = gui.vs_hde
    else:
        hde = gui.es_hde
    hde._deselection_event('', )


def find_rules(gui):
    btn = get_widget(app_widget, "43010")
    if btn.disabled:
        raise ValueError('skr button disabled')
    btn.click()


def undo(gui):
    btn = get_widget(app_widget, "4302")
    if btn.disabled:
        raise ValueError('undo button disabled')
    btn.click()


def validate_rules(gui):
    btn = get_widget(app_widget, "43030")
    if btn.disabled:
        raise ValueError('validate_rules button disabled')
    btn.click()


def auto_cluster(gui):
    btn = get_widget(app_widget, "4402000")
    if btn.disabled:
        raise ValueError('auto_cluster button disabled')
    btn.click()


def select_region(gui, region_num):
    data = {
        'value': True,
        'item': {'Region': region_num}
    }
    gui.region_selected(data)


def substitute(gui):
    btn = get_widget(app_widget, "4401000")
    if btn.disabled:
        raise ValueError('substitute button disabled')
    btn.click()


def subdivide(gui):
    btn = get_widget(app_widget, "440110")
    if btn.disabled:
        raise ValueError('subdivide button disabled')
    btn.click()


def delete(gui):
    btn = get_widget(app_widget, "440120")
    if btn.disabled:
        raise ValueError('delete button disabled')
    btn.click()


def select_model(gui, model):
    gui.sub_model_selected = {
        'value': True,
        'item': {'Sub-model': model}
    }

def validate(gui):
    btn = get_widget(app_widget, "450100")
    if btn.disabled:
        raise ValueError('validate button disabled')
    btn.click()
