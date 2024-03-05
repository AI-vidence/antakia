import ipyvuetify as v
import pandas as pd
from antakia_core.compute.model_subtitution.model_class import MLModel
from antakia_core.data_handler.region import Region, ModelRegion
from plotly.graph_objects import FigureWidget, Bar

from pdpbox.pdp import PDPIsolate, PDPInteract

from antakia.utils.stats import log_errors


class ModelExplorer:
    """
    This is ModelExplorer docstring.
    """

    def __init__(self, X: pd.DataFrame):
        self._build_widget()
        self.model: MLModel | None = None
        self.region: ModelRegion | None = None
        self.X = X
        pass

    def _build_widget(self):
        self.feature_importance_tab = v.TabItem(  # Tab 1) feature importances # 43
            class_="mt-2",
            children=[]
        )
        self.pdp_feature_select = v.Select()
        self.pdp_figure = v.Container()
        self.widget = v.Tabs(
            v_model=0,  # default active tab
            children=[
                         v.Tab(children=["Feature Importance"]),
                         v.Tab(children=["Partial Dependency"]),
                     ]
                     +
                     [
                         self.feature_importance_tab,
                         v.TabItem(  # Tab 2) Partial dependence
                             children=[
                                 v.Col(
                                     children=[
                                         self.pdp_feature_select,
                                         self.pdp_figure
                                     ]
                                 )
                             ]
                         ),  # End of v.TabItem #2
                     ]
        )
        self.pdp_feature_select.on_event('change', self.display_pdp)

    def update_selected_model(self, model: MLModel, region: ModelRegion):
        self.model = model
        self.region = region
        self.update_feature_importances()
        self.update_pdp_tab()

    def update_feature_importances(self):
        if self.model is not None:
            feature_importances = self.model.feature_importances_.sort_values(ascending=True)
            fig = Bar(x=feature_importances, y=feature_importances.index, orientation='h')
            self.figure_fi = FigureWidget(data=[fig])
            self.figure_fi.update_layout(
                autosize=True,
                margin={
                    't': 0,
                    'b': 0,
                    'l': 0,
                    'r': 0
                },
            )
            self.figure_fi._config = self.figure_fi._config | {"displaylogo": False}

            self.feature_importance_tab.children = [self.figure_fi]
        else:
            self.feature_importance_tab.children = []

    def update_pdp_tab(self):
        if self.pdp_feature_select.v_model not in self.X.columns and self.model is not None:
            features = list(self.model.feature_importances_.sort_values(ascending=False).index)
            self.pdp_feature_select.items = features
            self.pdp_feature_select.v_model = features[0]
        self.display_pdp()

    @log_errors
    def display_pdp(self, *args):
        if self.model is not None:
            selected_feature = self.pdp_feature_select.v_model
            if self.X[self.region.mask][selected_feature].nunique() > 1:
                predict_func = self.model.__class__.predict
                figure = PDPIsolate(
                    df=self.X.copy(), feature=selected_feature, feature_name=selected_feature,
                    model=self.model, model_features=self.X.columns, pred_func=predict_func,
                    n_classes=0  # regression
                ).plot()[0]
                self.figure_pdp = FigureWidget(figure)
                self.figure_pdp.update_layout(
                    autosize=True, width=None, height=None,
                    margin={
                        't': 0,
                        'b': 0,
                        'l': 0,
                        'r': 0
                    },
                )
                self.figure_pdp._config = self.figure_pdp._config | {"displaylogo": False}

                self.pdp_figure.children = [self.figure_pdp]
            else:
                self.pdp_figure.children = ['only one feature value, no pdp to display']
        else:
            self.pdp_figure.children = []

    def reset(self):
        self.model = None
        self.region = None
        self.update_feature_importances()
        self.update_pdp_tab()
