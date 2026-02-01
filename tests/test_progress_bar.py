from unittest import TestCase

import ipyvuetify as v

from antakia.gui.helpers.progress_bar import ProgressBar


class TestProgressBar(TestCase):
    def setUp(self):
        self.progress_bar = ProgressBar(v.ProgressLinear())

    def test_update_reset(self):
        pb = self.progress_bar
        assert pb.progress == 0
        pb.update(56.3, None)
        assert pb.progress == 56.3
        pb.update(99.8, None)
        assert pb.progress == 100
        assert pb.widget.color == pb.unactive_color

    def test_progress(self):
        pb = self.progress_bar
        assert pb.progress == 0
