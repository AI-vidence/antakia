from typing import Callable

import numpy as np
import pandas as pd
from antakia_core.compute.model_subtitution.model_class import MLModel
from antakia_core.data_handler import ModelRegionSet, ModelRegion, RegionSet, Region
from antakia_core.utils import ProblemCategory, boolean_mask, get_mask_comparison_color, timeit, DataVariables, \
    BASE_COLOR

from antakia.config import AppConfig
from antakia.gui.high_dim_exp.projected_value_bank import ProjectedValueBank

from antakia.utils.logging_utils import conf_logger, Log
from antakia.utils.stats import stats_logger, log_errors


class DataStore:

    @timeit
    def __init__(self, X: pd.DataFrame, y: pd.Series, variables: DataVariables,
                 X_exp: pd.DataFrame | None, X_test: pd.DataFrame | None,
                 y_test: pd.Series | None, model,
                 problem_category: ProblemCategory | str, score: Callable | str):
        self.X = X
        self.X_scaled: pd.DataFrame | None = None
        self.user_x_exp = X_exp
        self._X_exp = X_exp
        self.variables = variables
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.problem_category = problem_category
        self._y_pred = None
        self._y_pred_submodels = {}  # dictionary containing {'model_name' : y_predict_submodel}
        self.score = score
        self.color = 'y' # par défault lors du build on affiche les y target
        self.color_series = y

        self._selection_mask = boolean_mask(X, True)
        self.empty_selection = True
        self._rules_mask = boolean_mask(X, True)
        self._rule_selection_color = get_mask_comparison_color(
            self._rules_mask, self._selection_mask)
        self.highlighted_mask = boolean_mask(X, True)
        self.region_set = ModelRegionSet(self.X, self.y, self.X_test,
                                         self.y_test, self.model, self.score)
        self.pv_bank = ProjectedValueBank(self.y)
        self._display_mask: pd.Series | None = None

    @property
    def y_pred(self):
        if self._y_pred is None:
            pred = self.model.predict(self.X)
            if self.problem_category in [
                ProblemCategory.classification_with_proba
            ]:
                pred = self.model.predict_proba(self.X)

            if len(pred.shape) > 1:
                if pred.shape[1] == 1:
                    pred = pred.squeeze()
                if pred.shape[1] == 2:
                    pred = np.array(pred)[:, 1]
                else:
                    pred = pred.argmax(axis=1)
            self._y_pred = pd.Series(pred, index=self.X.index)
        return self._y_pred

    def y_pred_submodels(self, model: MLModel) -> pd.Series:
        """
        directly returns the prediction from the dictionary
        computes the required prediction if needed

        Parameters
        ----------
        model : model used for the predictions

        Returns
        the series containing the colors (values) predicted by the model
        -------

        """
        if model.name == 'Original Model':
            return self.y_pred
        if model.name not in self._y_pred_submodels:
            self._y_pred_submodels[model.name] = model.predict(self.X)
        return self._y_pred_submodels[model.name]

    @property
    def X_exp(self):
        return self._X_exp

    @X_exp.setter
    def X_exp(self, explanation_dataframe):
        self._X_exp = explanation_dataframe
        self._selection_mask = boolean_mask(self.X, True)
        self.empty_selection = True
        self._rules_mask = boolean_mask(self.X, True)
        self._compute_rule_selection_color()

    @property
    def selection_mask(self):
        return self._selection_mask

    @selection_mask.setter
    def selection_mask(self, mask: pd.Series):
        if not mask.any():
            mask = ~mask
        self.empty_selection = mask.all()
        self._selection_mask = mask
        self._compute_rule_selection_color()

    def _compute_rule_selection_color(self):
        if self.empty_selection:
            self._rule_selection_color = get_mask_comparison_color(
                self._rules_mask, self._rules_mask)
        else:
            self._rule_selection_color = get_mask_comparison_color(
                self._rules_mask, self._selection_mask)

    @property
    def rules_mask(self) -> pd.Series:
        return self._rules_mask

    @rules_mask.setter
    def rules_mask(self, mask: pd.Series):
        self._rules_mask = mask
        self._compute_rule_selection_color()

    @property
    def rule_selection_color(self):
        return self._rule_selection_color[0]

    @property
    def rule_selection_color_legend(self):
        return self._rule_selection_color[1]

    def reset_rules_mask(self):
        self.rules_mask = boolean_mask(self.X)

    def reset_selection_mask(self):
        self.selection_mask = boolean_mask(self.X)

    def reset_rules_and_selection(self):
        self._rules_mask = boolean_mask(self.X)
        self.selection_mask = boolean_mask(self.X)

    @log_errors
    def switch_color(self, region_list: [int], model: MLModel | None, viewmode=AppConfig.ATK_REGION_VIEWMODE):
        """
        functions that updates the color_series attribute and the mask attribute
        Parameters
        ----------
        region_list : list of regions to show
        model
        viewmode

        Returns
        pd.Series of colors to display(values)
        -------

        """
        with (Log('switch_color', 2)):
            stats_logger.log('color_changed', {'color': self.color})

            self.highlighted_mask = self.get_selected_mask(region_list)

            # ATTRIBUTION DE LA COULEUR
            match self.color :
                case "y":
                    self.color_series = self.y
                case "y^":
                    self.color_series = self.y_pred
                case "residual":
                    self.color_series = self.y - self.y_pred
                case "all_regions":
                    self.color_series = self.region_set.get_color_serie()
                case "region_selection":
                    self.color_series = self.region_set.get_color_serie()
                    if viewmode == 'grey mask':
                        if region_list and region_list is not None:
                            region_set_selected = RegionSet(self.X)
                            for i in region_list:
                                region_set_selected.add(self.region_set.get(i))
                            self.color_series = region_set_selected.get_color_serie()

                case 'y^model':
                    self.color_series = self.y_pred_submodels(model)
                case 'residual_sub':
                    self.color_series = self.y - self.y_pred_submodels(model)

                case 'rule_selection':
                    self.color_series = self.y
                    self.highlighted_mask = self.selection_mask.copy()
                case 'rule':
                    self.color_series = self.rule_selection_color
                    self.highlighted_mask = self.rule_selection_color != BASE_COLOR  # We highlight every point exept those in BASECOLOR (those not included in either selection mask or rule mask)

        return self.color


    def get_selected_mask(self, region_list: [Region]) -> pd.Series:
        """

        Parameters
        ----------
        region_list : list of regions to display

        Returns
        a mask of the regions to display
        -------

        """
        mask = boolean_mask(self.X, True)
        if region_list and region_list is not None:
            region_set_selected = RegionSet(self.X)
            for i in region_list:
                region_set_selected.add(self.region_set.get(i))
            mask = region_set_selected.mask

        return mask


    @property
    @timeit
    def display_mask(self) -> pd.Series:
        """
        mask should be applied on each display (x,y,z,color, selection)
        """
        if self._display_mask is None:
            limit = int(AppConfig.ATK_MAX_DOTS)
            if len(self.X) > limit:
                self._display_mask = pd.Series([False] * len(self.X),
                                               index=self.X.index)
                indices = np.random.choice(self.X.index,
                                           size=limit,
                                           replace=False)
                self._display_mask.loc[indices] = True
            else:
                self._display_mask = pd.Series([True] * len(self.X),
                                               index=self.X.index)
        return self._display_mask
