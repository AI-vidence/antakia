import unittest
import pytest
import ipyvuetify as v

from antakia.gui.progress_bar import ProgressBar, MultiStepProgressBar


class TestProgressBar(unittest.TestCase):
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


class TestMultiStepProgressBar(unittest.TestCase):
    def setUp(self):
        self.widget = v.Row(children=[v.Col(children=[])])

    def test_init_MSPB(self):
        mspb = MultiStepProgressBar(self.widget, steps=20)
        assert mspb.steps == 20
        assert mspb.widget is self.widget
        assert mspb.active_color == 'blue'
        assert mspb.unactive_color == 'grey'
        assert mspb.widget.v_model == 0
        assert mspb.reset_at_end

    def test_get_update_MSPB(self):
        mspb = MultiStepProgressBar(self.widget, steps=20)
        with pytest.raises(ValueError):
            mspb.get_update(0)
        with pytest.raises(ValueError):
            mspb.get_update(30)

    def test_set_progress_reset(self):
        mspb = MultiStepProgressBar(self.widget, steps=20)
        assert mspb.widget.v_model == 0
        mspb.set_progress(45.2)
        assert mspb.widget.v_model == 45.2
        mspb.set_progress(100)
        assert mspb.progress == 0
        assert mspb.widget.color == mspb.unactive_color
