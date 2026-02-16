from functools import partial

import ipyvuetify as v
import numpy as np
from antakia_core.utils import timeit

from antakia.utils.logging_utils import Log


class RuleSlider:

    def __init__(self,
                 range_min: float,
                 range_max: float,
                 step: float,
                 value_min: float | None = None,
                 value_max: float | None = None,
                 change_callback=None):
        self.range = range_min, range_max
        if value_max is None or value_max >= range_max:
            value_max = None
        if value_min is None or value_min <= range_min:
            value_min = None
        self.value = [value_min, value_max]
        self.step = step
        if change_callback is None:
            change_callback = lambda *args: None
        self.change_callback = partial(change_callback, self, 'change')
        self.build_widget()

    def build_widget(self):
        self.min_input = v.TextField(class_='ml-2 px-2',
                                     style_="max-width:100px",
                                     v_model=self.value[0],
                                     placeholder='',
                                     label='',
                                     disabled=False,
                                     type_="number")
        self.max_input = v.TextField(class_='px-2',
                                     style_="max-width:100px",
                                     v_model=self.value[1],
                                     placeholder='',
                                     label='',
                                     disabled=False)
        self.range_slider = v.RangeSlider(
            class_='px-2',
            thumb_label='always',
            thumb_size=30,
            thumb_color='blue',
            # style_="max-width:500px",
            height=90,
            v_model=[
                self.range[0] if self.value[0] is None else self.value[0],
                self.range[1] if self.value[1] is None else self.value[1]
            ],
            min=self.range[0],
            max=self.range[1],
            step=self.step,
            vertical=False,
            color='green',
            track_color='red')

        # self.range_slider.on_event('start', self._update_from_min_slider)
        self.range_slider.on_event('end', self._update_from_slider)

        self.min_input.on_event('blur', self._update_from_min_txt)
        self.min_input.on_event('keyup.enter', self._update_from_min_txt)
        self.max_input.on_event('blur', self._update_from_max_txt)
        self.max_input.on_event('keyup.enter', self._update_from_max_txt)

        self.widget = v.Row(
            children=[self.min_input, self.range_slider, self.max_input],
            align="center")

    def _display(self):
        self.min_input.v_model = self.value[0]
        self.max_input.v_model = self.value[1]

        min_val = self.value[0] if self.value[0] is not None else self.range[0]
        max_val = self.value[1] if self.value[1] is not None else self.range[1]
        self.range_slider.v_model = (min_val, max_val)

    def _update_min_txt(self):
        min_val = self.min_input.v_model
        try:
            if not min_val:
                min_val = None
            else:
                min_val = float(min_val)
                if min_val < self.range[0]:
                    min_val = None
            self.value[0] = min_val
        except ValueError:
            pass

    def _update_max_txt(self):
        max_val = self.max_input.v_model
        try:
            if not max_val:
                max_val = None
            else:
                max_val = float(max_val)
                if max_val > self.range[1]:
                    max_val = None
            self.value[1] = max_val
        except ValueError:
            pass

    def _update_from_min_txt(self, *args):
        self._update_min_txt()
        self._display()
        self.change_callback(self.value)

    def _update_from_max_txt(self, *args):
        self._update_max_txt()
        self._display()
        self.change_callback(self.value)

    def _update_min_slider(self):
        min_val = self.range_slider.v_model[0]
        try:
            min_val = float(min_val)
            if min_val <= self.range[0]:
                min_val = None
            self.value[0] = min_val
        except ValueError:
            pass

    def _update_max_slider(self):
        max_val = self.range_slider.v_model[1]
        try:
            max_val = float(max_val)
            if max_val >= self.range[1]:
                max_val = None
            self.value[1] = max_val
        except ValueError:
            pass

    def _update_from_slider(self, *args):
        if self.value[0] is not None and self.value[
                0] != self.range_slider.v_model[0]:
            self._update_min_slider()
        elif self.value[0] is None and self.range_slider.v_model[
                0] >= self.range[0]:
            self._update_min_slider()
        if self.value[1] is not None and self.value[
                1] != self.range_slider.v_model[1]:
            self._update_max_slider()
        elif self.value[1] is None and self.range_slider.v_model[
                1] <= self.range[1]:
            self._update_max_slider()
        self._display()
        self.change_callback(self.value)

    @timeit
    def set_value(self, min_val=None, max_val=None):
        """

        Parameters
        ----------
        min_val
        max_val

        Returns
        -------

        """
        if min_val <= self.range[0]:
            min_val = None
        if max_val >= self.range[1]:
            max_val = None
        self.value = [min_val, max_val]
        self._display()
