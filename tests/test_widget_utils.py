from antakia.gui.widget_utils import get_widget, change_widget
from antakia.gui.widgets import change_widget as cw_obs
from tests.test_antakia import TestAntakia
from unittest import TestCase
import pytest

import ipyvuetify as v


class TestWidgetUtils(TestCase):
    def setUp(self):
        self.widget = v.Row(children=[v.Col(children=[])])
        #CREER UN WIDGET PLUS COMPLEXE POUR Y NAVIGUER

    def test_get_widget(self):
        widget = self.widget
        a1 = get_widget(widget, '4400')
        a2 = get_widget(widget, '440')
        assert get_widget(widget, '4400') is (get_widget(get_widget(widget, '44'), '00'))
        with pytest.raises(ValueError):
            get_widget(widget, 'a')
        with pytest.raises(IndexError):
            get_widget(widget, '789456')
        assert get_widget(widget, '') is widget

    def test_change_widget(self):
        with pytest.raises(ValueError):
            change_widget(self.widget, False, None)
        assert not get_widget(self.widget, '4400') is get_widget(self.widget, '4401')
        change_widget(self.widget, '44', get_widget(self.widget, '440'))
        assert (get_widget(self.widget, '440').children) is (get_widget(self.widget, '44').children)