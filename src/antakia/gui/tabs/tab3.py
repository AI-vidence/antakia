from typing import Callable

import ipyvuetify as v
import pandas as pd
from antakia_core.data_handler import ModelRegion
from antakia_core.utils import BASE_COLOR

from antakia.config import AppConfig
from antakia.gui.helpers.data import DataStore
from antakia.gui.graphical_elements.sub_model_table import SubModelTable
from antakia.gui.helpers.progress_bar import ProgressBar
from antakia.gui.tabs.model_explorer import ModelExplorer
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors, stats_logger


class Tab3:
    headers = [
        {
            "text": column,
            "sortable": True,
            "value": column,
            # "class": "primary white--text",\
        } for column in ['Sub-model', 'MSE', 'MAE', 'R2', 'delta']
    ]

    def __init__(self, data_store: DataStore, validate_callback: Callable,
                 display_model_data: Callable):
        self.data_store = data_store
        self.validate_callback = validate_callback
        self.display_model_data = display_model_data
        self.model_explorer = ModelExplorer(self.data_store.X)
        self.region: ModelRegion | None = None
        self.substitution_model_training = False  # tab 3 : training flag

        self._build_widget()
        self.progress_bar = ProgressBar(self.progress_wgt,
                                        indeterminate=True,
                                        reset_at_end=True)
        self.progress_bar(100)

    def _build_widget(self):
        """
        build the tab3 widget - part of init method
        Returns
        -------

        """
        self.validate_model_btn = v.Btn(
            v_on='tooltip.on',
            class_="ma-1 green white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=["mdi-check"],
                ),
                "Validate sub-model",
            ],
        )
        self.model_table = SubModelTable(
            headers=self.headers,
            items=[],
        )
        self.region_prefix_wgt = v.Html(class_="mr-2",
                                        tag="h3",
                                        children=["Region"])
        self.region_chip_wgt = v.Chip(
            color=BASE_COLOR,
            children=["-"],
        )  # 450001
        self.region_title = v.Html(
            class_="ml-2",
            tag="h3",
            children=["No region selected for substitution"])  # 450002
        self.progress_wgt = v.ProgressLinear(  # 450110
            style_="width: 100%",
            class_="mt-4",
            v_model=0,
            height="15",
            indeterminate=True,
            color="blue",
        )
        self.widget = [
            v.Col(children=[
                v.Row(  #Row1 : Title and validate button
                    class_="d-flex",
                    children=[
                        v.Col(  # Col1 - region table
                            class_="col-9",
                            children=[
                                v.Sheet(
                                    class_="ma-1 d-flex flex-row align-center",
                                    children=[
                                        self.region_prefix_wgt,
                                        self.region_chip_wgt, self.region_title
                                    ])
                            ]),
                        v.Col(  # Col2 - buttons
                            class_="col-3",
                            children=[
                                v.Row(class_="flex-column",
                                      children=[
                                          v.Tooltip(
                                              bottom=True,
                                              v_slots=[{
                                                  'name':
                                                  'activator',
                                                  'variable':
                                                  'tooltip',
                                                  'children':
                                                  self.validate_model_btn,
                                              }],
                                              children=['Chose this submodel'])
                                      ])
                            ]),
                    ]),
                v.Row(  #Row2 : Progress bar
                    class_=' flex-column align-center',
                    children=[
                        v.Col(class_="col-5", children=[self.progress_wgt])
                    ]),
                v.Row(  #Row3 : Model table and explanations table
                    children=[
                        v.Col(class_="col-6", children=[self.model_table]),
                        v.Col(class_="col-6",
                              children=[self.model_explorer.widget])
                    ])
            ])
        ]
        self.progressbar_widget = self.widget[0].children[1]
        self.model_table_widget = self.widget[0].children[2]

        self.model_table_widget.hide()

        # We wire a select event on the 'substitution table' :
        self.model_table.set_callback(self._sub_model_selected_callback)

        # We wire a click event on the "validate sub-model" button :
        self.validate_model_btn.on_event("click", self._validate_sub_model)
        self.update()

    @property
    def selected_sub_model(self):
        return self.model_table.selected

    @selected_sub_model.setter
    def selected_sub_model(self, value):
        self.model_table.selected = value

    def update_region(self, region: ModelRegion, train=True):
        """
        method to update the region of substitution
        Parameters
        ----------
        region: region to substitute

        Returns
        -------

        """
        self.region = region
        if self.region is not None and train:
            # We update the substitution table once to show the name of the region
            self.substitution_model_training = True
            self.progress_bar(0)
            self.update()
            # show tab 3 (and update)
            self.region.train_substitution_models(
                task_type=self.data_store.problem_category)
            self.progressbar_widget.hide(
            )  #hides the progress bar widget once submodels trained
            self.model_table_widget.show(
            )  #displays the submodel table once submodels trained

            self.progress_bar(100)
            self.substitution_model_training = False
            self.update()
        else:
            self.update()

    def update(self):
        self._update_substitution_prefix()
        self._update_substitution_title()
        self._update_model_table()
        self._update_selected()
        self._update_validate_btn()

    def _update_substitution_prefix(self):
        # Region prefix text
        self.region_prefix_wgt.class_ = "mr-2 black--text" if self.region else "mr-2 grey--text"
        # v.Chip
        self.region_chip_wgt.color = self.region.color if self.region else BASE_COLOR
        self.region_chip_wgt.children = [str(self.region.num)
                                         ] if self.region else ["-"]

    def _update_model_table(self):
        if (self.substitution_model_training or not self.region
                or self.region.num_points() < AppConfig.ATK_MIN_POINTS_NUMBER
                or len(self.region.perfs) == 0):
            self.model_table.items = []
        else:

            def series_to_str(series: pd.Series) -> pd.Series:
                return series.apply(lambda x: f"{x:.2f}")

            perfs = self.region.perfs
            stats_logger.log('substitute_model',
                             {'best_perf': perfs['delta'].min()})
            for col in perfs.columns:
                if col != 'delta_color':
                    perfs[col] = series_to_str(perfs[col])
            perfs = perfs.reset_index().rename(columns={"index": "Sub-model"})
            headers = [{
                "text": column,
                "sortable": False,
                "value": column,
            } for column in perfs.drop('delta_color', axis=1).columns]
            self.model_table.headers = headers
            self.model_table.items = perfs.to_dict("records")

    def _update_selected(self):
        if self.region and self.region.interpretable_models.selected_model:
            # we set to selected model if any
            self.model_table.selected = [{
                'Sub-model':
                self.region.interpretable_models.selected_model
            }]
            self.model_explorer.update_selected_model(
                self.region.get_selected_model(), self.region)
        else:
            # clear selection if new region:
            self.model_explorer.reset()
            self.model_table.selected = []

    def _update_substitution_title(self):
        title = self.region_title
        title.tag = "h3"
        if self.substitution_model_training:
            # We tell to wait ...
            title.class_ = "ml-2 grey--text italic "
            title.children = [f"Sub-models are being evaluated ..."]
            # We clear items int the SubModelTable
        elif not self.region:  # no region provided
            title.class_ = "ml-2 grey--text italic "
            title.children = [f"No region selected for substitution"]
        elif self.region.num_points(
        ) < AppConfig.ATK_MIN_POINTS_NUMBER:  # region is too small
            title.class_ = "ml-2 red--text"
            title.children = ["Region too small for substitution !"]
        elif len(self.region.perfs) == 0:  # model not trained
            title.class_ = "ml-2 red--text"
            title.children = [
                "click on substitute button to train substitution models"
            ]
        else:
            # We have results
            title.class_ = "ml-2 black--text"
            title.children = [
                f"{self.region.name}, "
                f"{self.region.num_points()} points, {100 * self.region.dataset_cov():.1f}% of the dataset"
            ]

    def _update_validate_btn(self):
        self.validate_model_btn.disabled = len(self.selected_sub_model) == 0

    def _sub_model_selected_callback(self, data):
        """
        callback on model selection - updates the model explorer
        Parameters
        ----------
        data

        Returns
        -------

        """
        with Log('_sub_model_selected_callback', 2):
            is_selected = bool(data["value"])
            # We use this GUI attribute to store the selected sub-model
            self.selected_sub_model = [data['item']]
            model_name = data['item']['Sub-model']
            self.validate_model_btn.disabled = not is_selected
            if is_selected:
                self.model_explorer.update_selected_model(
                    self.region.get_model(model_name), self.region)
                self.display_model_data(
                    self.region, self.region.train_residuals(model_name))
            else:
                self.display_model_data(self.region, None)
                self.model_explorer.reset()

    @log_errors
    def _validate_sub_model(self, *args):
        """
        callback called on model validation
        Parameters
        ----------
        args

        Returns
        -------

        """
        # We get the sub-model data from the SubModelTable:
        # get_widget(self.widget,"45001").items[self.validated_sub_model]
        with Log('_validate_sub_model', 2):
            self.validate_model_btn.disabled = True

            stats_logger.log(
                'validate_sub_model',
                {'model': self.selected_sub_model[0]['Sub-model']})

            # We udpate the region
            self.region.select_model(self.selected_sub_model[0]['Sub-model'])
            self.region.validate()
            # empty selected region
            self.region = None
            self.selected_sub_model = []
            # Show tab 2
            self.validate_callback()
            self.progressbar_widget.show(
            )  # displays the progress bar widget in preparation for the next submodel training
            self.model_table_widget.hide()  # hides the submodel table
