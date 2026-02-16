import math

import ipyvuetify as v
import pandas as pd
from antakia_core.utils.splittable_callback import ProgressCallback

from antakia.utils.logging_utils import Log


class ProgressBar(ProgressCallback):

    def __init__(self,
                 widget,
                 indeterminate: bool = False,
                 active_color='blue',
                 unactive_color='grey',
                 reset_at_end=True,
                 min: float = 0,
                 max: float = 100,
                 log: Log | None = None):
        """
        generic progress bar update
        Parameters
        ----------
        widget : widget element
        indeterminate : whether the progress is indeterminate or finegrained
        active_color
        unactive_color
        reset_at_end : should we reset widget when 100% is reached
        """
        assert min <= max
        self.active_color = active_color
        self.unactive_color = unactive_color
        assert isinstance(widget, (v.ProgressLinear, v.ProgressCircular))
        self.widget = widget
        self.indeterminate = indeterminate
        self.reset_at_end = reset_at_end
        self.min = min
        self.max = max
        self._log = log
        self.sub_progress_bar: list[ProgressBar] = []

    def update(self, progress: float, time_elapsed=None):
        """

        Parameters
        ----------
        progress: progress value between 0 and 100
        time_elapsed : duration since start

        Returns
        -------

        """
        if progress > 100 or progress < 0:
            raise ValueError("progress should be between 0 and 100")
        if self._log is not None:
            self._log.percent(progress)
        progress = self.min + (self.max - self.min) / 100 * progress
        self.progress = progress
        self.widget.color = self.active_color
        self.widget.indeterminate = self.indeterminate

        if math.ceil(progress) >= 100 and self.reset_at_end:
            self.reset_progress_bar()

    def reset_progress_bar(self):
        self.progress = 100
        self.widget.indeterminate = False
        self.widget.color = self.unactive_color

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    @property
    def progress(self):
        if self.widget.v_model == '!!disabled!!':
            self.progress = 0
        return self.widget.v_model

    @progress.setter
    def progress(self, value: float):
        if value is None or pd.isna(value):
            self.widget.indeterminate = True
        else:
            if self.indeterminate and value <= 99:
                self.widget.indeterminate = True
            else:
                self.widget.indeterminate = False
            self.widget.v_model = value

    def _split_val(self, value: float):
        new_value = self.min + value / 100 * (self.max - self.min)
        first = ProgressBar(self.widget,
                            self.indeterminate,
                            self.active_color,
                            self.unactive_color,
                            self.reset_at_end,
                            min=self.min,
                            max=new_value,
                            log=self._log)
        second = ProgressBar(self.widget,
                             self.indeterminate,
                             self.active_color,
                             self.unactive_color,
                             self.reset_at_end,
                             min=new_value,
                             max=self.max,
                             log=self._log)
        return [first, second]

    def split_list(self, value_list: list[float]):
        if len(value_list) == 0:
            return self
        if len(value_list) == 1:
            return self._split_val(value_list[0])
        progress_bars = self.split_list(value_list[1:])
        progress_bars = progress_bars[0]._split_val(
            value_list[0]) + progress_bars[1:]
        return progress_bars

    def split(self, value: float | list[float]) -> list['ProgressBar']:
        if isinstance(value, list):
            value.sort()
            self.sub_progress_bar = self.split_list(value)
        else:
            self.sub_progress_bar = self._split_val(value)
        return self.sub_progress_bar

    def set_log(self, log):
        self._log = log
        for pr in self.sub_progress_bar:
            pr.set_log(self._log)
