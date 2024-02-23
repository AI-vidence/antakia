from antakia.gui.widget_utils import get_widget, change_widget as cw_new
from antakia.gui.widgets import change_widget as cw_obs
from tests.test_antakia import TestAntakia
from unittest import TestCase
import pytest

import ipyvuetify as v


# class = v.Row(children=[v.Col(children=[])])

def test_get_widget():
    widget = v.Row(children = [v.Col(children = [])])
    a1 = get_widget(widget, '4400')
    a2 = get_widget(widget, '440')

    assert get_widget(widget, '4400') is (get_widget(get_widget(widget, '44'), '00'))

    with pytest.raises(ValueError):
        get_widget(widget, 'a')

    with pytest.raises(IndexError):
        get_widget(widget, '789456')

    assert get_widget(widget, '') is widget


def test_change_widget():

    with pytest.raises(ValueError):
        cw_new(widget, False, None)
    assert not get_widget(widget, '4400') is get_widget(widget, '4401')

    cw_new(widget, '44', get_widget(widget, '440'))
    assert len(get_widget(AppWidget().widget, '440').children) == len(get_widget(widget,'44').children)
