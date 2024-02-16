import ipyvuetify as v
import pandas as pd
from antakia_core.compute.model_subtitution.model_class import MLModel
from antakia_core.data_handler.region import ModelRegion
from plotly.graph_objects import FigureWidget, Bar
import plotly.express as px

from pdpbox.pdp import PDPIsolate, PDPInteract


class ModelExplorer:
    def __init__(self, X: pd.DataFrame):
        self.build_widget()
        self.model: MLModel | None = None
        self.X = X
        pass

    def build_widget(self):
        self.widget = v.Tabs(  # 4
            v_model=0,  # default active tab
            children=[
                         v.Tab(children=["Feature Importance"]),
                         v.Tab(children=["Partial Dependency"]),
                         v.Tab(children=["qq plot"]),
                     ]
                     +
                     [
                         v.TabItem(  # Tab 1) feature importances # 43
                             class_="mt-2",
                             children=[]
                         ),
                         v.TabItem(  # Tab 2) Partial dependency #44
                             children=[
                                 v.Col(
                                     children=[
                                         v.Select(),
                                         v.Container()
                                     ]
                                 )
                             ]
                         ),  # End of v.TabItem #2
                         v.TabItem(  # TabItem #3 qq plot #45
                             children=[]
                         )
                     ]
        )
        self.pdp_feature_select.on_event('change', self.display_pdp)

    @property
    def feature_importance_tab(self) -> v.TabItem:
        return self.widget.children[3]

    @property
    def pdp_feature_select(self) -> v.TabItem:
        return self.widget.children[4].children[0].children[0]

    @property
    def pdp_figure(self) -> v.TabItem:
        return self.widget.children[4].children[0].children[1]

    def update_selected_model(self, model: MLModel):
        self.model = model
        self.update_feature_importances()
        self.update_pdp_tab()

    def update_feature_importances(self):
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

    def update_pdp_tab(self):
        features = list(self.model.feature_importances_.sort_values(ascending=False).index)
        self.pdp_feature_select.items = features
        self.pdp_feature_select.v_model = features[0]
        self.display_pdp()

    def display_pdp(self, *args):
        selected_feature = self.pdp_feature_select.v_model
        predict_func = self.model.__class__.predict
        figure = PDPIsolate(
            df=self.X.copy(), feature=selected_feature, feature_name=selected_feature,
            model=self.model, model_features=self.X.columns, pred_func=predict_func,
            n_classes=0  # regression
        ).plot()[0]
        self.figure_pdp = FigureWidget(figure)
        self.figure_pdp.update_layout(
            autosize=True,
            margin={
                't': 0,
                'b': 0,
                'l': 0,
                'r': 0
            },
        )
        self.figure_pdp._config = self.figure_pdp._config | {"displaylogo": False}

        self.pdp_figure.children = [self.figure_pdp]
