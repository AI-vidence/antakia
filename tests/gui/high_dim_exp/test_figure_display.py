import pandas as pd
import ipyvuetify as v

import antakia.config as config
import antakia_core.utils.utils as utils

from antakia.gui.high_dim_exp.figure_display import FigureDisplay

trace_name = FigureDisplay.trace_name


def test_trace_name():
    assert trace_name(0) == 'values trace'
    assert trace_name(1) == 'rules trace'
    assert trace_name(2) == 'regionset trace'
    assert trace_name(3) == 'region trace'
    assert trace_name(8) == 'unknown trace'


def test_init():
    X = pd.DataFrame([[4, 7, 10],
                      [5, 8, 11],
                      [6, 9, 12]],
                     index=[1, 2, 3],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    fd = FigureDisplay(X, y, lambda x: x + 1)
    assert fd.active_tab == 0
    assert fd._mask is None
    # assert fd.selection_changed
    assert fd.X.equals(X)
    assert fd.y.equals(y)
    # assert fd.widget == v.Container()
    assert fd.widget.class_ == "flex-fill"
    assert fd.fig_width == config.INIT_FIG_WIDTH / 2
    assert fd.fig_height == config.INIT_FIG_WIDTH / 4
    assert fd._selection_mode == 'lasso'
    assert fd._current_selection.equals(utils.boolean_mask(fd.X, True))
    assert fd.first_selection
    assert fd._visible == [True, False, False, False]
    assert fd._colors == [None, None, None, None]
    assert fd.figure_2D is None
    assert fd.figure_3D is None
    assert not fd.initialized

    fd = FigureDisplay(None, y, lambda x: x + 1)
    assert fd._current_selection is None


def test_set_get_figure():
    X = pd.DataFrame([[4, 7, 10],
                      [5, 8, 11],
                      [6, 9, 12]],
                     index=[1, 2, 3],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    fd = FigureDisplay(X, y, lambda x: x + 1)
    fd.figure = 5

    fd1 = FigureDisplay(None, y, lambda x: x + 1)
    fd1.figure = 5


def test_initialize_create_figure():
    X = pd.DataFrame([[4, 7],
                      [5, 8]],
                     index=[0, 1],
                     columns=[0, 1])
    y = pd.Series([1, 2])

    fd = FigureDisplay(X, y, lambda x: x + 1)

    fd.initialize()

    assert fd.initialized
    z = 1
    # test avec X = None


def test_get_X():
    X = pd.DataFrame([[4, 7, 10],
                      [5, 8, 11],
                      [6, 9, 12]],
                     index=[0, 1, 2],
                     columns=['a', 'b', 'c'])
    y = pd.Series([1, 2, 3])

    fd = FigureDisplay(X, y, lambda x: x + 1)
    fd.get_X(True)

