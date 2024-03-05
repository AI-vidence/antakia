import ipyvuetify as v
import pandas as pd
from antakia_core.data_handler.region import ModelRegion
from antakia_core.utils.utils import ProblemCategory

from antakia import config
from antakia.gui.graphical_elements.sub_model_table import SubModelTable
from antakia.gui.helpers.progress_bar import ProgressBar
from antakia.gui.tabs.model_explorer import ModelExplorer
from antakia.utils.stats import log_errors, stats_logger


class Tab3:
    headers = [
        {
            "text": column,
            "sortable": True,
            "value": column,
            # "class": "primary white--text",\
        }
        for column in ['Sub-model', 'MSE', 'MAE', 'R2', 'delta']
    ]

    def __init__(self, X: pd.DataFrame, problem_category: ProblemCategory, validate_callback: callable):
        self.X = X
        self.problem_category = problem_category
        self.validate_callback = validate_callback
        self.model_explorer = ModelExplorer(self.X)
        self.region: ModelRegion | None = None
        self.substitution_model_training = False  # tab 3 : training flag

        self._build_widget()
        self.progress_bar = ProgressBar(
            self.progress_wgt, indeterminate=True, reset_at_end=True
        )
        self.progress_bar(100)

    def _build_widget(self):
        """
        build the tab3 widget - part of init method
        Returns
        -------

        """
        self.validate_model_btn = v.Btn(  # 4501000
            v_on='tooltip.on',
            class_="ma-1 mt-12 green white--text",
            children=[
                v.Icon(
                    class_="mr-2",
                    children=[
                        "mdi-check"
                    ],
                ),
                "Validate sub-model",
            ],
        )
        self.model_table = SubModelTable(  # 45001
            headers=self.headers,
            items=[],
        )
        self.region_prefix_wgt = v.Html(class_="mr-2", tag="h3", children=["Region"])  # 450000
        self.region_chip_wgt = v.Chip(color="red", children=["1"], )  # 450001
        self.region_title = v.Html(class_="ml-2", tag="h3", children=[""])  # 450002
        self.progress_wgt = v.ProgressLinear(  # 450110
            style_="width: 100%",
            class_="mt-4",
            v_model=0,
            height="15",
            indeterminate=True,
            color="blue",
        )
        self.widget = [
            v.Row(  # 450
                class_="d-flex",
                children=[
                    v.Col(  # Col1 - sub model table #4500
                        class_="col-5",
                        children=[
                            v.Sheet(  # 45000
                                class_="ma-1 d-flex flex-row align-center",
                                children=[
                                    self.region_prefix_wgt,
                                    self.region_chip_wgt,
                                    self.region_title
                                ]
                            ),
                            self.model_table
                        ]
                    ),
                    v.Col(  # Col2 - buttons #4501
                        class_="col-2",
                        children=[
                            v.Row(
                                class_="flex-column",
                                children=[
                                    v.Tooltip(  # 45010
                                        bottom=True,
                                        v_slots=[
                                            {
                                                'name': 'activator',
                                                'variable': 'tooltip',
                                                'children':
                                                    self.validate_model_btn,
                                            }
                                        ],
                                        children=['Chose this submodel']
                                    )
                                ]
                            ),
                            v.Row(
                                class_="flex-column",
                                children=[
                                    self.progress_wgt
                                ]
                            )
                        ]
                    ),
                    v.Col(  # Col3 - model explorer #4502
                        class_="col-5",
                        children=[
                            self.model_explorer.widget
                        ]
                    ),
                ]
            )
        ]
        # We wire a select event on the 'substitution table' :
        self.model_table.set_callback(self._sub_model_selected_callback)

        # We wire a ckick event on the "validate sub-model" button :
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
            self.region.train_substitution_models(task_type=self.problem_category)

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
        self.region_chip_wgt.color = self.region.color if self.region else "grey"
        self.region_chip_wgt.children = [str(self.region.num)] if self.region else ["-"]

    def _update_model_table(self):
        if (
            self.substitution_model_training or
            not self.region or
            self.region.num_points() < config.ATK_MIN_POINTS_NUMBER or
            len(self.region.perfs) == 0
        ):
            self.model_table.items = []
        else:
            def series_to_str(series: pd.Series) -> str:
                return series.apply(lambda x: f"{x:.2f}")

            perfs = self.region.perfs.copy()
            stats_logger.log('substitute_model', {'best_perf': perfs['delta'].min()})
            for col in perfs.columns:
                if col != 'delta_color':
                    perfs[col] = series_to_str(perfs[col])
            perfs = perfs.reset_index().rename(columns={"index": "Sub-model"})
            headers = [
                {
                    "text": column,
                    "sortable": False,
                    "value": column,
                }
                for column in perfs.drop('delta_color', axis=1).columns
            ]
            self.model_table.headers = headers
            self.model_table.items = perfs.to_dict("records")

    def _update_selected(self):
        if self.region and self.region.interpretable_models.selected_model:
            # we set to selected model if any
            self.model_table.selected = [{'Sub-model': self.region.interpretable_models.selected_model}]
            self.model_explorer.update_selected_model(self.region.get_selected_model(), self.region)
        else:
            # clear selection if new region:
            self.model_explorer.reset()
            self.model_table.selected = []

    def _update_substitution_title(self):
        title = self.region_title
        title.tag = "h3"
        table = self.model_table  # subModel table
        if self.substitution_model_training:
            # We tell to wait ...
            title.class_ = "ml-2 grey--text italic "
            title.children = [f"Sub-models are being evaluated ..."]
            # We clear items int the SubModelTable
        elif not self.region:  # no region provided
            title.class_ = "ml-2 grey--text italic "
            title.children = [f"No region selected for substitution"]
        elif self.region.num_points() < config.ATK_MIN_POINTS_NUMBER:  # region is too small
            title.class_ = "ml-2 red--text"
            title.children = ["Region too small for substitution !"]
        elif len(self.region.perfs) == 0:  # model not trained
            title.class_ = "ml-2 red--text"
            title.children = ["click on substitute button to train substitution models"]
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
        is_selected = bool(data["value"])
        # We use this GUI attribute to store the selected sub-model
        self.selected_sub_model = [data['item']]
        self.validate_model_btn.disabled = not is_selected
        if is_selected:
            self.model_explorer.update_selected_model(self.region.get_model(data['item']['Sub-model']), self.region)
        else:
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

        self.validate_model_btn.disabled = True

        stats_logger.log('validate_sub_model', {'model': self.selected_sub_model[0]['Sub-model']})

        # We udpate the region
        self.region.select_model(self.selected_sub_model[0]['Sub-model'])
        self.region.validate()
        # empty selected region
        self.region = None
        self.selected_sub_model = []
        # Show tab 2
        self.validate_callback()
