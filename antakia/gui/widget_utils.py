from ipywidgets.widgets import Widget
import ipyvuetify as v


def get_widget(root_widget: Widget, address: str | list[int]) -> Widget:
    """
    Returns a sub widget of root_widget. Address is a sequence of childhood ranks as a string
    Return sub_widget may be modified, it's still the same sub_widget of the root_widget
    get_widget(root_widget, '0') returns root_widgetn first child
    TODO : allow childhood rank > 9
    """
    if not address:
        return root_widget

    try:
        address = [int(i) for i in address]
    except ValueError:
        raise ValueError(address, "must be a string composed of digits")
    try:
        return recursive_get_widget(root_widget, address)
    except:
        raise IndexError(f"Nothing found @{address} in this {root_widget.__class__.__name__}")


def recursive_get_widget(root_widget: Widget, address: list[int]):
    if not address:
        return root_widget

    if isinstance(root_widget, v.Tooltip):
        assert address.pop(0) == 0
        new_root = root_widget.v_slots[0]["children"]
    else:
        new_root = root_widget.children[address.pop(0)]
    return recursive_get_widget(new_root, address)


def change_widget(root_widget: Widget, address: str, new_widget):
    """
    Substitutes a sub_widget in a root_widget.
    Address is a sequence of childhood ranks as a string, root_widget first child address is  '0'
    The root_widget is altered but the object remains the same
    """

    if not address:
        raise ValueError('must provide at least the parent widget')

    parent_widget = get_widget(root_widget, address[:-1])
    new_children = list(parent_widget.children)
    new_children[int(address[-1])] = new_widget
    parent_widget.children = new_children
