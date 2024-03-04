import math

import ipyvuetify as v
import pandas as pd


class ProgressBar:
    def __init__(self, widget, indeterminate: bool = False, active_color='blue', unactive_color='grey',
                 reset_at_end=True):
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
        self.active_color = active_color
        self.unactive_color = unactive_color
        assert isinstance(widget, (v.ProgressLinear, v.ProgressCircular))
        self.widget = widget
        self.indeterminate = indeterminate
        self.progress = 0
        self.reset_at_end = reset_at_end

    def update(self, progress: float, time_elapsed=None):
        """

        Parameters
        ----------
        progress: progress value between 0 and 100
        time_elapsed : duration since start

        Returns
        -------

        """
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


class MultiStepProgressBar:
    def __init__(self, widget, steps=1, active_color='blue', unactive_color='grey', reset_at_end=True):
        """
        generic progress bar update

        Parameters
        ----------
        widget : widget element
        steps
        active_color
        unactive_color
        reset_at_end : should we reset widget when 100% is reached
        """
        self.steps = steps
        self.widget = widget
        self.active_color = active_color
        self.unactive_color = unactive_color
        self.widget.v_model = 0
        self.reset_at_end = reset_at_end

    def get_update(self, step):
        """
        returns the progress updater for the provided step
        Parameters
        ----------
        step

        Returns
        -------

        """
        if step == 0 or step > self.steps:
            raise ValueError('step should be between 1 and self.steps')

        def update_ac_progress_bar(progress: float, duration: float):
            """
            Called by the AutoCluster to update the progress bar
            """
            self.widget.color = self.active_color
            progress = round(((step - 1) * 100 + progress) / self.steps)
            self.set_progress(progress)

        return update_ac_progress_bar

    def set_progress(self, progress: float):
        """
        force progress value
        Parameters
        ----------
        progress

        Returns
        -------

        """
        self.widget.v_model = progress
        if progress >= 100 and self.reset_at_end:
            self.reset_progress_bar()

    @property
    def progress(self):
        return self.widget.v_model

    @progress.setter
    def progress(self, value):
        self.set_progress(value)

    def reset_progress_bar(self):
        self.progress = 0
        self.widget.color = self.unactive_color
