import ipyvuetify as v


class ProgressBar:
    def __init__(self, widget, indeterminate: bool = False, active_color='blue', unactive_color='grey',
                 reset_at_end=True):
        """
        generic progress bar update
        Parameters
        ----------
        widget : widget element
        indeterminate : whether the progress is indeterminate or finegrained
        """
        self.active_color = active_color
        self.unactive_color = unactive_color
        assert isinstance(widget, (v.ProgressLinear, v.ProgressCircular))
        self.widget = widget
        self.indeterminate = indeterminate
        self.progress = 0
        self.reset_at_end = reset_at_end

    def update(self, progress: float, time_elapsed):
        """

        Parameters
        ----------
        progress: progress value between 0 and 100
        time_elapsed : duration since start

        Returns
        -------

        """
        self.widget.color = self.active_color
        self.widget.indeterminate = self.indeterminate

        self.progress = progress
        self.widget.v_model = round(progress)

        if round(progress) >= 100 and self.reset_at_end:
            self.reset_progress_bar()

    def reset_progress_bar(self):
        self.progress = 0
        self.widget.indeterminate = False
        self.widget.color = self.unactive_color
        self.widget.disabled = False


class MultiStepProgressBar:
    def __init__(self, widget, steps=1, active_color='blue', unactive_color='grey', reset_at_end=True):
        """
        generic progress bar update
        Parameters
        ----------
        widget : widget element
        indeterminate : whether the progress is indeterminate or finegrained
        """
        self.steps = steps
        self.widget = widget
        self.active_color = active_color
        self.unactive_color = unactive_color
        self.progress = 0
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
            self.widget.v_model = progress
            if progress >= 100 and self.reset_at_end:
                self.reset_progress_bar()

        return update_ac_progress_bar

    def reset_progress_bar(self):
        self.progress = 0
        self.widget.color = self.unactive_color
        self.widget.disabled = False
