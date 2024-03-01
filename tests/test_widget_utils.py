from antakia.gui.widget_utils import get_widget, change_widget
from unittest import TestCase
import pytest

import ipyvuetify as v


class TestWidgetUtils(TestCase):
    def setUp(self):
        self.widget = v.Row(children=[
            v.Col(children=[]),
            v.Col(children=[]),
            v.Col(children=[]),
            v.Col(children=[]),
            v.Col(children=[  # 4
                v.Col(children=[
                ]),
                v.Col(children=[
                ]),
                v.Col(children=[
                ]),
                v.Col(children=[
                ]),
                v.Col(children=[  # 44
                    v.Col(children=[  # 440
                        v.Col(children=[  # 4400
                        ]),
                    ]),
                ]),
            ]),
        ])

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
        new_widget = v.Col()
        # test no adress
        with pytest.raises(ValueError):
            change_widget(self.widget, '', new_widget)
        # test change 1
        change_widget(self.widget, '0', new_widget)
        assert get_widget(self.widget, '0') is new_widget

        # test change 2
        change_widget(self.widget, '440', new_widget)
        assert get_widget(self.widget, '440') is new_widget