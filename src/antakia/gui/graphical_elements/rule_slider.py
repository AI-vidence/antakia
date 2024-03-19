from functools import partial

import ipyvuetify as v
import numpy as np


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

        self.range_slider.on_event('end', self._update_txt)
        self.range_slider.on_event('start', self._update_txt)

        self.max_input.on_event('blur', self._update_slider_and_txt)
        self.max_input.on_event('keyup.enter', self._update_slider_and_txt)
        self.min_input.on_event('blur', self._update_slider_and_txt)
        self.min_input.on_event('keyup.enter', self._update_slider_and_txt)

        self.widget = v.Row(
            children=[self.min_input, self.range_slider, self.max_input],
            align="center")

    def _update_txt(self,
                    wgt=None,
                    event: str | None = None,
                    data: tuple[float, float] | None = None,
                    *,
                    callback=True):
        """
        method to update text field values - called by slider
        Parameters
        ----------
        wgt
        event
        data : new range value from slider - data contains always two float

        Returns
        -------

        """
        if data is not None:
            if data[1] >= self.range[1]:
                self.value[1] = None
            else:
                self.value[1] = float(data[1])
            if data[0] <= self.range[0]:
                self.value[0] = None
            else:
                self.value[0] = float(data[0])
        self.max_input.v_model = self.value[1]
        self.min_input.v_model = self.value[0]
        if callback:
            self.change_callback(self.value)

    def _update_slider(self, *args, callback=True):
        """
        method to update slider values - called by txt fields
        Parameters
        ----------
        args

        Returns
        -------

        """
        # update self.value
        # update min
        min_val = self.min_input.v_model
        if not min_val:
            min_val = None
        else:
            try:
                min_val = float(min_val)
                if min_val < self.range[0]:
                    min_val = None
                    self.min_input.v_model = None
            except ValueError:
                min_val = self.value[0]
        self.value[0] = min_val
        # update max
        max_val = self.max_input.v_model
        if not max_val:
            max_val = None
        else:
            try:
                max_val = float(max_val)
                if max_val > self.range[1]:
                    max_val = None
                    self.max_input.v_model = None
            except ValueError:
                max_val = self.value[1]
        self.value[1] = max_val
        # compute range
        range_ = self.value.copy()
        if range_[0] is None:
            range_[0] = self.range[0]
        if range_[1] is None:
            range_[1] = self.range[1]
        # update slider
        self.range_slider.v_model = range_
        if callback:
            self.change_callback(self.value)

    def _update_slider_and_txt(self, *args, callback=True):
        """
        method to update slider and txt fields
        Returns
        -------

        """
        self._update_slider(callback=False)
        self._update_txt(callback=False)
        if callback:
            self.change_callback(self.value)

    def set_value(self, min_val=None, max_val=None):
        """

        Parameters
        ----------
        min_val
        max_val

        Returns
        -------

        """
        if min_val == -np.inf:
            min_val = None
        if max_val == np.inf:
            max_val = None

        self.max_input.v_model = max_val
        self.min_input.v_model = min_val
        self._update_slider_and_txt(callback=False)
