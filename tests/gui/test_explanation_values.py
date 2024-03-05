import mock
import numpy as np
import pandas as pd
import pytest
from unittest import TestCase

from antakia.utils.dummy_datasets import generate_corner_dataset
from antakia_core.utils.utils import ProblemCategory
from antakia.gui.app_bar.explanation_values import ExplanationValues
from tests.test_antakia import dummy_exp
from tests.utils_fct import test_progress_bar, dummy_callable, DummyModel


class TestExplanationValues(TestCase):
    def setUp(self) -> None:
        self.X, self.y = generate_corner_dataset(10)
        self.callable = dummy_callable
        self.model = DummyModel

        def on_change_callback(pv, progress=None):
            if progress is None:
                progress = test_progress_bar.update
            progress(100, 0)

        self.on_change_callback = on_change_callback

    def test_init(self):  # ajouter test click

        exp_val = ExplanationValues(self.X, self.y, self.model, ProblemCategory.regression, self.on_change_callback,
                                    self.callable)
        np.testing.assert_array_equal(exp_val.X, self.X)
        np.testing.assert_array_equal(exp_val.y, self.y)
        assert exp_val.model is self.model
        assert exp_val.explanations == {'Imported': None, 'SHAP': None, 'LIME': None}
        assert exp_val.current_exp == 'SHAP'
        assert exp_val.on_change_callback is self.on_change_callback

        X_exp = generate_corner_dataset(10)[0]
        exp_val = ExplanationValues(self.X, self.y, self.model, ProblemCategory.regression, self.on_change_callback,
                                    self.callable, X_exp=X_exp)
        assert exp_val.current_exp == 'Imported'
        np.testing.assert_array_equal(exp_val.explanations['Imported'], X_exp)
        assert exp_val.explanations['SHAP'] is None
        assert exp_val.explanations['LIME'] is None

        # get_widget(exp_val.widget, "13000203").click()
        # get_widget(exp_val.widget, "13000303").click()

    @mock.patch('antakia.gui.app_bar.explanation_values.compute_explanations')
    def test_initialize(self, cpt_exp):
        X_exp = pd.DataFrame(generate_corner_dataset(10)[0])
        cpt_exp.return_value = X_exp
        exp_val = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                    self.on_change_callback, self.callable, X_exp=X_exp)
        exp_val.initialize(self.callable)
        assert exp_val.current_exp == exp_val.available_exp[0]
        assert exp_val.get_explanation_select().v_model == 'Imported'

        exp_val1 = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                     self.on_change_callback, self.callable)
        exp_val1.initialize(test_progress_bar.update)
        assert exp_val1.get_explanation_select().v_model == 'SHAP'
        assert test_progress_bar.progress == 100

    def test_current_pv(self):
        exp_val = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                    self.on_change_callback, self.callable)
        assert exp_val.current_exp_df == exp_val.explanations[exp_val.current_exp]

    def test_has_user_exp(self):
        X_exp = pd.DataFrame(generate_corner_dataset(10)[0])
        exp_val = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                    self.on_change_callback, self.callable, X_exp=X_exp)
        assert exp_val.has_user_exp

        exp_val = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                    self.on_change_callback, self.callable)
        assert not exp_val.has_user_exp

    @mock.patch('antakia.gui.app_bar.explanation_values.compute_explanations')
    def test_update_explanation_select(self, cpt_exp):
        exp_val = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                    self.on_change_callback, self.callable)

        cpt_exp.return_value = self.X
        assert exp_val.get_explanation_select().items == [{'disabled': True, "text": 'Imported'},
                                                          {'disabled': False, "text": 'SHAP (compute)'},
                                                          {'disabled': False, "text": 'LIME (compute)'}]

    @mock.patch('antakia.gui.app_bar.explanation_values.compute_explanations', wraps=dummy_exp)
    def test_compute_explanation(self, _):
        exp_val = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                    self.on_change_callback, self.callable)
        # cpt_exp.return_value = pd.DataFrame(self.X)
        assert exp_val.current_exp == exp_val.available_exp[1]

        assert exp_val.explanations['Imported'] is None
        assert exp_val.explanations['SHAP'] is None
        assert exp_val.explanations['LIME'] is None

        exp_val.compute_explanation(1, test_progress_bar.update)

        assert test_progress_bar.progress == 100
        assert exp_val.get_explanation_select().items == [{"text": 'Imported', 'disabled': True},
                                                          {"text": 'SHAP', 'disabled': False},
                                                          {"text": 'LIME (compute)', 'disabled': False}]

    def test_compute_btn_clicked(self):  # à compléter
        pass
        # exp_val.compute_btn_clicked(get_widget(exp_val.widget, "130000"), None, None)
        # exp_val.compute_btn_clicked(None, None, None)
        # assert get_widget(exp_val.widget, "13000203").disabled
        # assert exp_val.current_exp == 1

    def test_disable_selection(self):
        exp0 = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                 self.on_change_callback, self.callable)
        exp0.disable_selection(True)
        assert exp0.get_explanation_select().disabled

        exp2 = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                 self.on_change_callback, self.callable)
        exp2.disable_selection(False)
        assert not exp2.get_explanation_select().disabled

    @mock.patch('antakia.gui.app_bar.explanation_values.compute_explanations')
    def test_explanation_select_changed(self, cpt_exp):
        data = 'SHAP'
        cpt_exp.return_value = pd.DataFrame(self.X)
        exp_val = ExplanationValues(pd.DataFrame(self.X), self.y, self.model, ProblemCategory.regression,
                                 self.on_change_callback, self.callable)

        exp_val.explanation_select_changed(None, None, data)
        assert exp_val.current_exp == data
        with pytest.raises(KeyError):
            exp_val.explanation_select_changed(None, None, None)
        exp_val.explanation_select_changed(None, None, 'LIME')
        assert exp_val.current_exp == 'LIME'
        # tester l'appel de on_change_callback
