import numpy as np
import pandas as pd

from ipywidgets import Layout, widgets
from ipywidgets.widgets import Widget
from IPython.display import display
import ipyvuetify as v
from traitlets import HasTraits, TraitError
from plotly.graph_objects import FigureWidget, Histogram, Scatter, Scatter3d
import seaborn as sns

from antakia.data import ExplanationMethod, DimReducMethod, Variable, ProjectedValues
import antakia.compute as compute
from antakia.utils import confLogger
from antakia.selection import Selection, Rules
import antakia.config as config

from copy import deepcopy, copy
from importlib.resources import files

import logging
logger = logging.getLogger(__name__)
handler = confLogger(logger)
handler.clear_logs()
handler.show_logs()

def get_widget_at_address(root_widget: Widget, address: str) -> Widget:
    """
    Returns a sub widget of root_widget. Address is a sequence of childhood ranks as a string
    Return sub_widget may be modified, it's still the same sub_widget of the root_widget
    TODO : allow childhood rank > 9
    """

    try: 
        int(address)
    except ValueError:
        raise ValueError(address, " must be a string composed of digits")
    
    if len(address) > 1:
        try:
            return get_widget_at_address(
                root_widget.children[int(address[0])],
                address[1:]
            )
        except IndexError:
            raise IndexError(f"Nothing found @{address} in this {type(root_widget)}")
    else:
        return root_widget.children[int(address[0])]


def _get_parent(widget:Widget, address: str) -> Widget:
    return get_widget_at_address(widget, address[:-1])

def change_widget(root_widget: Widget, address: str, sub_widget: Widget):
    """
    Substitutes a sub_widget in a root_widget.
    Address is a sequence of childhood ranks as a string
    The root_widget is altered but the object remains the same
    """
    if len(address) > 1:
        parent_widget = _get_parent(root_widget, address)

        # Because ipywidgets store their children in a tuple (vs list):
        if isinstance(parent_widget, Widget):
            parent_children = copy(list(parent_widget.children))
        else:
            parent_children = copy(parent_widget.children)

        try: 
            parent_children[int(address[-1])] = sub_widget
        except IndexError:
            raise IndexError(f"Nothing found @{address} in this {type(root_widget)}")

        if isinstance(parent_widget, Widget):
            parent_children = tuple(parent_children)
        try:
            parent_widget.children = parent_children
        except TraitError:
            raise TraitError(f"{type(parent_widget)} cannot have for children {parent_children}")
        
        change_widget(root_widget, address[:-1], parent_widget)
    else:
        parent_children = copy(list(root_widget.children))
        parent_children[int(address[-1])] = sub_widget
        root_widget.children = tuple(parent_children)


def create_rule_card(object) -> list:
    return None

def datatable_from_Selection(sel: list, length: int) -> v.Row:

    """ Returns a DataTable from a list of Selections
    """
    new_df = []
    
    for i in range(len(sel)):
        new_df.append(
            [
                i + 1,
                sel[i].size(),
                str(
                    round(
                        sel[i].size()
                        / length
                        * 100,
                        2,
                    )
                )
                + "%",
            ]
        )
    new_df = pd.DataFrame(
        new_df,
        columns=["Region #", "Number of points", "Percentage of the dataset"],
    )
    data = new_df.to_dict("records")
    columns = [
        {"text": c, "sortable": False, "value": c} for c in new_df.columns
    ]
    datatable = v.DataTable(
        class_="w-100",
        style_="width : 100%",
        show_select=False,
        single_select=True,
        v_model=[],
        headers=columns,
        explanationsMenuDict=data,
        item_value="Region #",
        item_key="Region #",
        hide_default_footer=True,
    )
    all_chips = []
    all_radio = []
    size = len(sel)
    coeff = 100
    start = 0
    end = (size * coeff - 1) * (1 + 1 / (size - 1))
    step = (size * coeff - 1) / (size - 1)
    scale_colors = np.arange(start, end, step)
    a = 0
    for i in scale_colors:
        color = sns.color_palette(
            "viridis", size * coeff
        ).as_hex()[round(i)]
        all_chips.append(v.Chip(class_="rounded-circle", color=color))
        all_radio.append(v.Radio(class_="mt-4", value=str(a)))
        a += 1
    all_radio[-1].class_ = "mt-4 mb-0 pb-0"
    radio_group = v.RadioGroup(
        v_model=None,
        class_="mt-10 ml-7",
        style_="width : 10%",
        children=all_radio,
    )
    chips_col = v.Col(
        class_="mt-10 mb-2 ml-0 d-flex flex-column justify-space-between",
        style_="width : 10%",
        children=all_chips,
    )
    return v.Row(
        children=[
            v.Layout(
                class_="flex-grow-0 flex-shrink-0", children=[radio_group]
            ),
            v.Layout(
                class_="flex-grow-0 flex-shrink-0", children=[chips_col]
            ),
            v.Layout(
                class_="flex-grow-1 flex-shrink-0",
                children=[datatable],
            ),
        ],
    )



class HighDimExplorer :
    """
        An HighDimExplorer displays one or several high dim Dataframes on a scatter plot.
        It uses several dimension reduction techniques, through the DimReduction class.
        It can display in or 2 dimensions.

        Implemntation details :
        It handes projections computation itself when needed.
        But, it asks GUI when another dataframe is asked for.
        It stores dataframes with the ProjectedValues class.
        It stored the current projection method (in widget) but not the dimension
        Attributes are mostly privates (underscorred) since they are not meant to be used outside of the class.

        Attributes :
        pv_list: list # a list of one or several ProjectedValues (PV)
        current_pv : int, stores the index of current PV in the _pv_list
        _y : pd.Series
        _pacmap_params : dictionnary containing the parameters for the PaCMAP projection
            nested keys are "previous" / "current", then "VS" / "ES", then "n_neighbors" / "MN_ratio" / "FP_ratio"
        _current_dim 
        selection_changed : callable (from GUI)
        new_values_wanted : callable (from GUI)

        Widgets :
        _values_select: v.Select
            None if len(_pv_list) = 1
            The labels of its items are transmitted by GUI at construction
        _compute_menu : v.Menu
            None if len(_pv_list) = 1
            Triggers the provision of other dataframes
        _figure_2D and _figure_3D : FigureWidget
            Plotly scatter plot
        _VBoxGraph : the WidgetGraph that contains a VBox and the figure
        _projection_select : v.Select, with mutliple dimreduc methods
        _progress_circular : v.ProgressCircular, used to display the progress of the computation
        _projection_slider_VBoxes : dict of VBox,  parameters for dimreduc methods
        fig_size

    """

    def __init__(self, space_name: str, dataframes_list: list, labels_list: list, is_computable_list: list, y:pd.Series, init_proj: int, init_dim: int, fig_size: int, border_size: int,  selection_changed: callable, new_values_wanted: callable=None):
        """
        Instantiate a new HighDimExplorer.

        Selected parameters :
            space_name : str, the name of the space explored. Stored in a widget
            dataframes_list : list of pd.Dataframes. Stored in ProjectedValues
            labels_list : list of str. 
                Stored in a Select widget
                if len(label_list) = 1, the widget is not created, and the label is ignord
            is_computable_list : list of bool. Indicates wich dataframes are computable from the compute_menu
            init_proj, init_dim : int, int, used to initialize widgets
        """
        if init_dim not in [2, 3]:
            raise ValueError(f"HDE.init: dim must be 2 or 3, not {init_dim}")
        self._current_dim = init_dim

        self.selection_changed = selection_changed
        self.new_values_wanted = new_values_wanted
        
        if not (len(dataframes_list) == len(labels_list) == len(is_computable_list)) :
            raise ValueError(f"HDE.init: values_list, labels_list and is_computable_list must have the same length")

        self.pv_list = []
        for index, values in enumerate(dataframes_list):
            if values is not None:
                self.pv_list.append(ProjectedValues(values))
            else:
                self.pv_list.append(None)
        self._y = y
        
        
        self.current_pv = 0
        for i in range(len(self.pv_list)):
            if self.pv_list[i] is not None :
                self.current_pv = i
                break
        
        self._projection_select = v.Select(
            label="Projection in the " + space_name,
            items=DimReducMethod.dimreduc_methods_as_str_list(),
            style_="width: 150px",
        )
        self._projection_select.on_event("change", self.projection_select_changed)

        # We initiate it in grey, not indeterminate :
        self._progress_circular = v.ProgressCircular(color="grey", width="6", size="35", class_="mx-4 my-3", v_model=100)

        # Since HDE is responsible for storing its current proj, we check init value :
        if init_proj not in DimReducMethod.dimreduc_methods_as_list() :
            raise ValueError(f"HDE.init: {init_proj} is not a valid projection method code")
        self._projection_select.v_model = DimReducMethod.dimreduc_method_as_str(init_proj)


        self._projection_slider_VBoxes = {}
        # We know PaCMAP uses these parameters :
        self._projection_slider_VBoxes[DimReducMethod.PaCMAP] = widgets.VBox([   
                v.Slider(
                    v_model=10, min=5, max=30, step=1, label="Number of neighbours"
                ),
                # v.Html(class_="ml-3", tag="h3", children=["machin"]),
                v.Slider(
                    v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio"
                ),
                # v.Html(class_="ml-3", tag="h3", children=["truc"]),
                v.Slider(
                    v_model=2, min=0.1, max=5, step=0.1, label="FP ratio"
                ),
                # v.Html(class_="ml-3", tag="h3", children=["bidule"]),
                ],
            )
        # TODO : wire slider events and store in below dict
        self._pacmap_params = {
            "previous": {
                "VS": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
                "ES": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
            },
            "current": {
                "VS": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
                "ES": {"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2},
            },
        }

        if len(self.pv_list) == 1:
            self._values_select = None
            self._compute_menu = None
        else:
            # The rest of __init__ is specific to the multi-dataframe case :
            select_items = []
            for i in range(len(self.pv_list)):
                select_items.append({'text': labels_list[i], 'disabled': self.pv_list[i] is None})

            self._values_select = v.Select(
                label="Explanation method",
                items=select_items,
                class_="ma-2 mt-1 ml-6",
                style_="width: 150px",
                disabled=False,
                )
            self._values_select.on_event("change", self._values_select_changed)
            
            computable_labels_list = [labels_list[i] for i in range(len(labels_list)) if is_computable_list[i]]
            tab_list = [v.Tab(children=label) for label in computable_labels_list]
            content_list = [
                v.TabItem(children=[
                    v.Col(
                        class_="d-flex flex-column align-center",
                        children=[
                                v.Html(
                                    tag="h3",
                                    class_="mb-3",
                                    children=["Compute " + label],
                                ),
                                v.ProgressLinear(
                                    style_="width: 80%",
                                    v_model=0,
                                    color="primary",
                                    height="15",
                                    striped=True,
                                ),
                                v.TextField(
                                    class_="w-100",
                                    style_="width: 100%",
                                    v_model = "0.00% [0/?] - 0m0s (estimated time : /min /s)",
                                    readonly=True,
                                ),
                                v.Btn(
                                    children=[v.Icon(class_="mr-2", children=["mdi-calculator-variant"]), "Compute values"],
                                    class_="ma-2 ml-6 pa-3",
                                    elevation="3",
                                    v_model=label,
                                    color="primary",
                                ),
                        ],
                    )
                ])
                for label in computable_labels_list]

            self._compute_menu = v.Menu( 
                    v_slots=[
                    {
                        "name": "activator",
                        "variable": "props",
                        "children": v.Btn(
                            v_on="props.on",
                            icon=True,
                            size="x-large",
                            children=[v.Icon(children=["mdi-timer-sand"], size="large")],
                            class_="ma-2 pa-3",
                            elevation="3",
                        ),
                    }
                    ],
                    children=[
                        v.Card( 
                            class_="pa-4",
                            rounded=True,
                            children=[
                                widgets.VBox([ 
                                    v.Tabs(
                                        v_model=0, 
                                        children=tab_list + content_list
                                        )
                                    ],
                                )
                                ],
                            min_width="500",
                        )
                    ],
                    v_model=False,
                    close_on_content_click=False,
                    offset_y=True,
                )
            

            # SHAP compute button :
            get_widget_at_address(self._compute_menu, "000203").on_event("click", self.compute_btn_clicked)
            # LIME compute button :
            get_widget_at_address(self._compute_menu, "000303").on_event("click", self.compute_btn_clicked)

        #  Now we can init figures 2 and 3D
        self.fig_size = fig_size

        if len(self.pv_list) == 1:
            html_text = "<h3>Values Space<h3>"
            our_marker=dict(
                    color=self._y,
                    colorscale="Viridis",
                    colorbar=dict( # only VS marker has a colorbar
                        title="y",
                        thickness=20,
                        ),
                )
        else:
            html_text = "<h3>Explanations Space<h3>"
            our_marker=dict(
                    color=self._y,
                    colorscale="Viridis"
                )
        if self.pv_list[self.current_pv] is None or self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(),2) is None:
            self._figure_2D = FigureWidget(
                data=Scatter(
                    x=None,
                    y=None,
                    mode="markers", 
                    marker=our_marker,
                    customdata=self._y, 
                    hovertemplate='%{customdata:.3f}'
                )
            )
        else:
            self._figure_2D = FigureWidget(
                data=Scatter(
                    x=self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), 2)[0],
                    y=self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), 2)[1],
                    mode="markers", 
                    marker=our_marker,
                    customdata=self._y, 
                    hovertemplate='%{customdata:.3f}'
                )
            )
        self._figure_2D._config = self._figure_2D._config | {"displaylogo": False}
        self._figure_2D.update_layout(
            margin=dict(
                t=0
                ), 
            width=self.fig_size
            )
        self._figure_2D.update_layout(dragmode="lasso")
        self._figure_2D.data[0].on_selection(self.dots_lasso_selected)

        # We use a WidgetGraph : it will be easier to swap figures
        self._VBox_widget = widgets.VBox(
                            [
                                widgets.HTML(html_text),
                                self._figure_2D, # We init HDE in 2D
                            ],
                            layout=Layout(
                                display="flex", align_items="center", margin="0px 0px 0px 0px"
                            ),
                    )

        if self.pv_list[self.current_pv] is None or self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(),3) is None:
            self._figure_3D = FigureWidget(
                data=Scatter3d(
                    x=None,
                    y=None,
                    z=None,
                    mode="markers", 
                    marker=our_marker,
                    customdata=self._y, 
                    hovertemplate='%{customdata:.3f}'
                )
            )
        else:
            self._figure_3D = FigureWidget(
                data=Scatter3d(
                    x=self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), 3)[0],
                    y=self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), 3)[1],
                    z=self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), 3)[2],
                    mode="markers", 
                    marker= our_marker,
                    customdata=self._y, 
                    hovertemplate='%{customdata:.3f}'
                )
            )
        self._figure_3D._config = self._figure_3D._config | {"displaylogo": False}
        self._figure_3D.update_layout(
            margin=dict(
                t=0
                ), 
            width=self.fig_size
            )
        self._figure_3D.update_layout(dragmode="lasso")
        

    # ---- Methods ------

    def __str__(self)-> str:
        return "HighDimExplorer : values space (VS)" if len(self.pv_list) == 1 else "HighDimExplorer : explanations space (ES)"

    def compute_projected_dots_2D3D_if_needed(self, callback: callable = None):
        """
        If check if our projs (2 and 3D), are computed.
        Note : we only computes the values for _pv_list[self.current_pv]
        If needed, we compute them and store them in the PV
        The callback function may by GUI.update_splash_screen or HDE.update_progress_circular
        depending of the context.
        """
        if self.pv_list[self.current_pv] is None:
            projected_dots_2D = projected_dots_3D = None
        else:
            projected_dots_2D = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), 2)
            projected_dots_3D = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), 3)
        

        if projected_dots_2D is not None and projected_dots_3D is not None:
            # Nothing to compute
            pass
        else:
            if projected_dots_2D is None:
                projected_dots_2D = compute.compute_projection(self.pv_list[self.current_pv].X, self._get_projection_method(), 2, callback)
            
            if projected_dots_3D is None:
                projected_dots_3D = compute.compute_projection(self.pv_list[self.current_pv].X, self._get_projection_method(), 3, callback)

            self.pv_list[self.current_pv].set_proj_values(self._get_projection_method(), 2, projected_dots_2D)
            self.pv_list[self.current_pv].set_proj_values(self._get_projection_method(), 3, projected_dots_3D)
    
    def update_progress_circular(self, caller: DimReducMethod, progress: int, duration:float):
        """
            Each proj computation consists in 2 (2D and 3D) tasks.
            So progress of each task in divided by 2 and summed together
        """
        if self._progress_circular.color == "grey":
            self._progress_circular.color = "blue"
            self._progress_circular.v_model = 0
            self._projection_select.disabled = True

        self._progress_circular.v_model = self._progress_circular.v_model + round(progress / 2)

        if self._progress_circular.v_model == 100:
            self._progress_circular.color = "grey"
            self._projection_select.disabled = False

    def projection_select_changed(self, widget, event, data):
        """" 
            Called when the user changes the projection method
            If needed, we compute the new projection
        """
        # TODO : proj_select gets frozen / unaccessible while computing :
        self.compute_projected_dots_2D3D_if_needed(self.update_progress_circular) # to ensure we got the values
        # TODO : proj_select gets released
        self.redraw()


    def compute_btn_clicked(self, widget, event, data):
            """
                Called  when new explanation computed values are wanted
            """
            # This compute btn is no longer useful / clickable
            widget.disabled = True
            # GUI's new_values_wanted func "speaks" in index of the values_list transmitted :
            index = 1 if data == "SHAP" else 3
            self.new_values_wanted(index, self.update_progress_linear)


    def _values_select_changed(self, widget, event, data):
        """
            Called  when the user choses another dataframe
        """
        # Remember : impossible items ine thee Select are disabled = we have the desired values
        chosen_pv_index = self._label_to_int(data)
        chosen_pv = self.pv_list[chosen_pv_index]
        self.current_pv = chosen_pv_index
        self.redraw()
    

    def update_progress_linear(self, method: int, progress: int, duration:float):

        """
        Called by the computation process (SHAP or LUME) to udpate the progress linear
        """
        progress_linear = None
        if method == ExplanationMethod.SHAP:
            progress_linear = WidgetGraph(self.new_compute_menu).get_widget_at_address("000201")
        else:
            progress_linear = WidgetGraph(self.new_compute_menu).get_widget_at_address("000301")
        
        progress_linear.v_model = progress

        
        if progress == 100:
            tab = None
            if method == ExplanationMethod.SHAP:
                tab = WidgetGraph(self.new_compute_menu).get_widget_at_address("0000")
            else:
                tab = WidgetGraph(self.new_compute_menu).get_widget_at_address("0001")
            tab.disabled = True

    def set_dimension(self, dim : int) :
        # Dimension is stored in the instance variable _current_dim
        """
        At init, dim is 2
        At runtime, GUI calls this function, we swap the figures in our VBox
        """
        self._current_dim = dim
        new_figure = self._figure_2D if dim == 2 else self._figure_3D
        change_widget(self._VBox_widget,"1", new_figure)
        

    def _get_projection_method(self) -> int :
        # proj is stored in the proj Select widget
        """
        Returns the current projection method
        """
        return DimReducMethod.dimreduc_method_as_int(self._projection_select.v_model)

    def redraw(self, opacity_values: pd.Series = None, color: pd.Series = None):
        self._redraw_just_one(self._figure_2D)
        self._redraw_just_one(self._figure_3D)

    def _redraw_just_one(self, fig: FigureWidget, opacity_values:pd.Series = None, color:pd.Series = None):
        if opacity_values is None:
            opacity_values = fig.data[0].marker.opacity
        if color is None:
            color = fig.data[0].marker.color
        
        with fig.batch_update():
                fig.data[0].x = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), self._current_dim)[0]
                fig.data[0].y = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), self._current_dim)[1]
                if isinstance(fig, Scatter3d):
                    fig.data[0].z = self.pv_list[self.current_pv].get_proj_values(self._get_projection_method(), self._current_dim)[2]
                fig.data[0].marker.opacity = opacity_values
                fig.data[0].marker.color = color
                fig.layout.width = self.fig_size
                fig.data[0].customdata = color


    def dots_lasso_selected(self, trace, points, selector, *args):
        """ Called whenever the user selects dots on the scatter plot """
        self.selection_changed(self, Selection(list(trace['selectedpoints']), Selection.LASSO))

    
    def set_selection(self, new_selection:Selection):
        """
        Called by tne UI when a new selection occured on the other HDE
        """
        self._figure_2D.data[0]['selectedpoints']= new_selection.indexes

        # # We set opacity to 10% for the selection
        # new_opacity_serie = pd.Series()
        # for i in range(len(self.X_list[0])):
        #         if i in self.selection.get_indexes():
        #             new_opacity_serie.append(1)
        #         else:
        #             new_opacity_serie.append(0.1)
        # # self._opacity[0 if side == config.VS else 1] = new_opacity_serie

        # self.update_selection_table()
        # # self.redraw_graph(side)


    def get_projection_select(self):
        return self._projection_select
    
    def get_projection_prog_circ(self)-> v.ProgressCircular:
        return self._progress_circular

    
    def get_compute_menu(self):
        return self._compute_menu

    def get_values_select(self):
        return self._values_select
    
    def get_proj_params_menu(self):
        return v.Menu( 
                    v_slots=[
                    {
                        "name": "activator",
                        "variable": "props",
                        "children": v.Btn(
                            v_on="props.on",
                            icon=True,
                            size="x-large",
                            children=[v.Icon(children=["mdi-cogs"], size="large")],
                            class_="ma-2 pa-3",
                            elevation="3",
                        ),
                    }
                    ],
                    children=[
                        v.Card( 
                            class_="pa-4",
                            rounded=True,
                            children=[
                                self._projection_slider_VBoxes[self._get_projection_method()]
                                ],
                            min_width="500",
                        )
                    ],
                    v_model=False,
                    close_on_content_click=False,
                    offset_y=True,
                )
    
    def get_Figure_VBox(self):
        return self._VBox_widget
    
    def _get_space_name(self) -> str:
        """
        For debug purposes only. Not very reliable.
        """
        # TODO : remove
        return "VS" if len(self.pv_list) == 1 else "ES"
    
    def _label_to_int(self, label : str) -> int :
        """
            Returns the index of a PV in the Select items
        """
        return self._values_select.items.index(label) 

    def get_current_X(self)-> pd.DataFrame:
        return self.pv_list[self.current_pv].X



class RuleVariableRefiner :
    """
        A RuleVariableRefiner is a piece of GUI (accordionGrp) that allows the user to refine a rule by selecting a variable and values for this variable.
        It displays the distribution for this variable as well as a beswarm of the explained values.
        The user can use the slider to change the rule.


        _widget : the WidgetGraph of nested widgets that make up the RuleVariableRefiner
        _variable : the variable that is being refined
        _rules : [Rules for VS, Rules for ES]
        refiner_rules_changed : callable of the GUI parent
    """

    def __init__(self, variable : Variable, refiner_rules_changed : callable, rules: Rules = None):
        self.refiner_rules_changed = refiner_rules_changed
        self.selection = None # fix
        self._variable  = variable
        self.root_widget = v.ExpansionPanels( # accordionGrp
            class_="ma-2 mb-1",
            children=[
                v.ExpansionPanel( # 0
                    disabled = False,
                    children=[
                        v.ExpansionPanelHeader( # 00
                            children=
                            ["X1"]
                            ),
                        v.ExpansionPanelContent( # 01
                            children=[
                                widgets.HBox( # accordion # 010 
                                    [ 
                                        widgets.VBox( # histoCtrl # 010 0
                                            [   
                                                v.Layout( # skopeSliderGroup # 010 00 
                                                    children=[
                                                        v.TextField( # 010 000 
                                                            style_="max-width:100px",
                                                            v_model=1, # min value of the slider
                                                            hide_details=True,
                                                            type="number",
                                                            density="compact",
                                                        ),
                                                        v.RangeSlider( # skopeSlider # 010 001
                                                            class_="ma-3",
                                                            v_model=[-1, 1],
                                                            min=-10e10,
                                                            max=10e10,
                                                            step=0.01,
                                                        )                                                                    
                                                        ,
                                                        v.TextField(  # 010 002
                                                            style_="max-width:100px",
                                                            v_model=5, # max value of the slider
                                                            hide_details=True,
                                                            type="number",
                                                            density="compact",
                                                            step="0.1",
                                                        ),
                                                    ],
                                                ),
                                                FigureWidget( # histogram # 010 01
                                                    data=[ 
                                                        Histogram(
                                                            x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')),
                                                            bingroup=1, 
                                                            nbinsx=50, 
                                                            marker_color="grey"
                                                            )
                                                    ]
                                                ),
                                                widgets.HBox( # validateSkopeChangeBtnAndCheck  # 010 02
                                                    [
                                                        v.Btn( # validateSkopeChangeBtn # 010 020
                                                            class_="ma-3",
                                                            children=[
                                                                v.Icon(class_="mr-2", children=["mdi-check"]),
                                                                "Validate the changes",
                                                            ],
                                                        ), 
                                                        v.Checkbox(  # realTimeUpdateCheck # 010 021
                                                                v_model=False, label="Real-time updates on the figures", class_="ma-3"
                                                            )
                                                    ]
                                                )
                                            ] # end VBox # 010 0
                                        ),
                                        widgets.VBox( # beeswarmGrp #010 1
                                            [
                                                v.Row( # bs1ColorChoice # 010 10
                                                    class_="pt-3 mt-0 ml-4",
                                                    children=[
                                                        "Value of Xi",
                                                        v.Switch( # 010 100
                                                            class_="ml-3 mr-2 mt-0 pt-0",
                                                            v_model=False,
                                                            label="",
                                                        ),
                                                        "Current selection",
                                                    ],
                                                ),
                                                FigureWidget( # beeswarm # 010 11
                                                    data=[Scatter(
                                                        x=pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD')), 
                                                        y=[0] * 100, 
                                                        mode="markers")]
                                                )
                                            ],
                                            layout=Layout(margin = "0px 0px 0px 20px")
                                            ), 
                                        v.Col( # rightSide # 010 2
                                            children=[
                                                v.Btn( # deleteSkopeBtn # 010 20
                                                    class_="ma-2 ml-4 pa-1",
                                                    elevation="3",
                                                    icon=True,
                                                    children=[v.Icon(children=["mdi-delete"])],
                                                    disabled=True,
                                                ),
                                                v.Checkbox( # isContinuousChck # 010 21
                                                    v_model=True, 
                                                    label="is continuous?"
                                                    )
                                                ],
                                            class_="d-flex flex-column align-center justify-center",
                                        )
                                    ],
                                    layout=Layout(align_explanationsMenuDict="center"),
                                ) # End HBox 010
                                ]               
                        ), # End ExpansionPanelContent 01
                    ]
                ), # End ExpansionPanel 0 
            ] 
        )
        # We vire the input event on the skopeSlider (010001)
        get_widget_at_address(self.root_widget, "010001").on_event("input", self.skope_rule_changed)
        # We vire the click event on validateSkopeChangeBtn (010020)
        get_widget_at_address(self.root_widget,"010020").on_event("click", self.skope_slider_changed)

    def hide_beeswarm(self, hide : bool):
        # We retrieve the beeswarmGrp (VBox)
        get_widget_at_address(self.root_widget,"0101").disabled = hide
    

    def skope_slider_changed(*change):
        # we just call skope_changed @GUI
        self.skope_changed()

    def redraw_both_graphs(self):
        # We update the refiner's histogram :
        with get_widget_at_address(self.root_widget,"01001").batch_update():
            get_widget_at_address(self.root_widget,"01001").data[0].x = \
                self._ds.get_full_values()[self._selection.get_vs_rules()[self._variable.get_col_index][2]]
        
        # We update the refiner's beeswarm :
        # get_widget_at_address(self.root_widget,"01011").v_model : # TODO Why do we check ?
            with get_widget_at_address(self.root_widget,"01011").batch_update():
                pass
                # TODO to understand
                # TODO : maybe the refiner could handle its colors itself
                # y_color = [0] * self._gui._ds.get_length()
                # if i == rule_index:
                #     indexs = (
                #         self._ds.get_full_values()
                #         .index[
                #             self._ds.getXValues()[
                #                 self._selection.getVSRules()[i][2]
                #             ].between(min, max)
                #         ]
                #         .tolist()
                #     )
                # else:
                #     indexs = (
                #         self._ds.get_full_values().index[
                #             self._ds.getXValues()[
                #                 self._selection.getVSRules()[i][2]
                #             ].between(
                #                 self._selection.getVSRules()[i][0],
                #                 self._selection.getVSRules()[i][4],
                #             )
                #         ].tolist()
                #     )
                # for j in range(
                #     len(self._xds.get_full_values(self._explanationES[0]))
                # ):
                #     if j in total_list:
                #         y_color[j] = "blue"
                #     elif j in indexs:
                #         y_color[j] = "#85afcb"
                #     else:
                #         y_color[j] = "grey"
                # widget_at_address(self._widget, "001011").data[0].marker.color = y_color
        
    def skope_rule_changed(widget, event, data):
        pass
        # # when the value of a slider is modified, the histograms and graphs are modified
        # if widget.__class__.__name__ == "RangeSlider":
        #     # We set the text before the slider (0010000) to the min value of the slider
        #     self._widget.get_widget_at_address("010000").v_model = self._widget.get_widget_at_address("010001").v_model[0]
        #     # We set the text after the slider (0010002) to the min value of the slider
        #     self._widget.get_widget_at_address("010002").v_model = self._widget.get_widget_at_address("010001").v_model[1]
        # else:
        #     if (
        #         self._widget.get_widget_at_address("010000").v_model == "" and self._widget.get_widget_at_address("010002").v_model == ""
        #     ):
        #         # If no value, we return
        #         return
        #     else:
        #         # Inversely, we set the slider to the values after the texts
        #         self._widget.get_widget_at_address("010001").v_model = [
        #             float(self._widget.get_widget_at_address("010000").v_model), # min
        #             float(self._widget.get_widget_at_address("010002").v_model), # max
        #         ]
        
        # new_list = [
        #     g
        #     for g in list(
        #         self._gui.get_dataset().get_full_values()[self._gui.get_selection().getVSRules()[0][2]].values
        #     )
        #     if g >= self._widget.get_widget_at_address("0010001").v_model[0] and g <= self._widget.get_widget_at_address("010001").v_model[1]
        # ]

        # # We updat the histogram (01001)
        # with self._widget.get_widget_at_address("01001").batch_update():
        #     self._widget.get_widget_at_address("01001").data[1].x = new_list
        
        # # TODO : what is _activate_histograms
        # if self._activate_histograms:
        #     self._gui.update_histograms_with_rules(self._widget.get_widget_at_address("010001").v_model[0], self._widget.get_widget_at_address("010001").v_model[1], 0)

        # # If realTimeUpdateCheck (0010021) is checked :
        # self._widget.get_widget_at_address("010021").v_model:
        #     # We update rules with the skopeSlider (0010001) values  
        #     self.selection.getVSRules()[0][0] = float(self._widget.get_widget_at_address("010021").v_model[0]) # min
        #     self.selection.getVSRules()[0][4] = float(self._widget.get_widget_at_address("010021").v_model[1]) # max
        #     get_app_graph.get_widget_at_address("30500101").children = create_rule_card(self.selection.ruleListToStr()) 
        #     # self._gui.update_histograms_with_rules()

    def get_class_selector(self, min : int = 1, max : int = -1, fig_size :int =700) -> v.Layout :
            valuesList = list(set(self._gui.get_dataset().getVariableValue(self._variable)))
            widgetList = []
            for value in valuesList:
                if value <= max and value >= min:
                    inside = True
                else:
                    inside = False
                widget = v.Checkbox(
                    class_="ma-4",
                    v_model=inside,
                    label=str(value).replace("_", " "),
                )
                widgetList.append(widget)
            row = v.Row(class_ = "ml-6 ma-3", children = widgetList)
            text = v.Html(tag="h3", children=["Select the values of the feature " + self._variable.getSymbol()])
            return v.Layout(class_= "d-flex flex-column align-center justify-center", style_="width: "+str(int(fig_size)-70)+"px; height: 303px", children=[v.Spacer(), text, row])

    def real_time_changed(*args):
        """ If changed, we invert the validate button """
        get_widget_at_address(self.root_widget,"0010020").disabled = not get_widget_at_address(self.root_widget,"0010020").disabled
    
        # See realTimeUpdateCheck (0010021)
        get_widget_at_address(self.root_widget,"0010021").on_event("change", real_time_changed)

    def beeswarm_color_changed(*args): 
        """ If changed, we invert the showScake value """
        # See beeswarm :
        show_scale = get_widget_at_address(self.root_widget,"01011").data[0].marker[showscale]
        show_scale = get_widget_at_address(self.root_widget,"01011").update_traces(marker=dict(showscale=not show_scale))
    
        # See bsColorChoice[,v.Switch] (0010101)
        self._widgetGraph.get_widget_at_address("010101").on_event("change", beeswarm_color_changed)


    def continuous_check_changed(widget, event, data): 
        features = [
            self._selection.getVSRules()[i][2]
            for i in range(len(self._selection.getVSRules()))
        ]
        aSet = []
        for i in range(len(features)):
            if features[i] not in aSet:
                aSet.append(features[i])

        index = features.index(aSet[2])
        if widget.v_model :
            # TODO : understand
            # We define accordion (0010) children as histoCtrl (00100) + list (accordion(0010).children[1])
            self._widget.get_widget_at_address("010").children = [self._widget.get_widget_at_address("0100")] + list(self._widget.get_widget_at_address("010").children[1:])
            count = 0
            for i in range(len(self._gui.get_selection().getVSRules())):
                if (
                    self._gui.get_selection().getVSRules()[i - count][2]
                    == self._selection.getVSRules()[index][2]
                    and i - count != index
                ):
                    self._gui.get_selection().getVSRules().pop(i - count)
                    count += 1
            # We set skopeSlider (0010001) values
            self.selection.getVSRules()[index][0] = get_widget_at_address(self.root_widget,"010001").v_model[0]
            self.selection.getVSRules()[index][4] = get_widget_at_address(self.root_widget,"010001").v_model[1]
            
            self._skope_list = create_rule_card(self.selection.ruleListToStr())
        else:
            class_selector = self.get_class_selector()
            get_widget_at_address(self.root_widget,"010").children = [class_selector] + list(
                get_widget_at_address(self.root_widget,"010").children[1:]
            )
            aSet = []
            for i in range(len(self.get_class_selector().children[2].children)):
                if class_selector.children[2].children[i].v_model:
                    aSet.append(
                        int(class_selector.children[2].children[i].label)
                    )
            if len(aSet) == 0:
                widget.v_model = True
                return
            column = deepcopy(self._gui.get_selection().getVSRules()[index][2])
            count = 0
            for i in range(len(self._gui.get_selection().getVSRules())):
                if self._gui.get_selection().getVSRules()[i - count][2] == column:
                    self._gui.get_selection().getVSRules().pop(i - count)
                    count += 1
            ascending = 0  
            for item in aSet:
                self.selection.getVSRules().insert(
                    index + ascending, [item - 0.5, "<=", column, "<=", item + 0.5]
                )
                ascending += 1
            self._skope_list = create_rule_card(self._gui.get_selection().ruleListToStr())

        # We wire the "change" event on the isContinuousChck (001021)
        get_widget_at_address(self.root_widget,"01021").on_event("change", continuous_check_changed)


def update_skr_infocards(selection: Selection, side: int, widget: Widget):
    """ Sets a message + indicates the scores of the sub_models
        Do not set the rules themselves
    """

    if selection.is_empty():
            temp_card_children = [widgets.HTML("Please select points")]
    else :
        if 0 not in selection.getYMaskList() or 1 not in selection.getYMaskList() :
                temp_card_children = [widgets.HTML("You can't choose everything/nothing !")]
        else:
            # If no rule for one of the two, nothing is displayed
            if not selection.has_rules_defined():
                    temp_card_children = [widgets.HTML("No rule found")]
            else:
                if side == config.VS :
                    scores = selection.getVSScore()
                else :
                    scores = selection.getESScore()
                temp_text_children= \
                    "p = " + str(scores[0]) + "% " \
                    + "r = " + str(scores[1]) + "% " \
                    + " ext. of the tree = " + str(scores[2])

    get_widget_at_address(widget, "30500101").children = temp_card_children
    get_widget_at_address(widget, "30500101").children = temp_text_children


splash_widget = v.Layout(
            class_="d-flex flex-column align-center justify-center",
            children=[
                widgets.Image( # 0
                    value=widgets.Image._load_file_value(files("antakia.assets").joinpath("logo_antakia.png")), layout=Layout(width="230px")
                ), 
                v.Row( # 1
                    style_="width:85%;",
                    children=[
                        v.Col( # 10
                            children=[
                                v.Html(
                                    tag="h3",
                                    class_="mt-2 text-right",
                                    children=["Computation of explanation values"],
                                )
                            ]   
                        ),
                        v.Col( # 11
                            class_="mt-3", 
                            children=[
                                v.ProgressLinear(
                                    style_="width: 80%",
                                    class_="py-0 mx-5",
                                    v_model=0,
                                    color="primary",
                                    height="15",
                                    striped=True,
                                )
                            ]
                        ),
                        v.Col( # #12
                            children=[
                                v.TextField( # 120
                                    variant="plain",
                                    v_model="", 
                                    readonly=True,
                                    class_="mt-0 pt-0",
                                    )
                            ]
                        ),
                    ],
                ),
                v.Row( # 2
                    style_="width:85%;",
                    children=[
                        v.Col( # 20
                            children=[
                                v.Html(
                                    tag="h3",
                                    class_="mt-2 text-right",
                                    children=["Computation of dimension reduction values"],
                                )
                            ]
                        ),
                        v.Col( # 21
                            class_="mt-3", 
                               children=[
                                    v.ProgressLinear( # 210
                                        style_="width: 80%",
                                        class_="py-0 mx-5",
                                        v_model=0,
                                        color="primary",
                                        height="15",
                                        striped=True,
                                    )
                        ]
                        ),
                        v.Col( # 22
                            children=[
                                v.TextField( # 220
                                    variant="plain",
                                    v_model="",
                                    readonly=True,
                                    class_="mt-0 pt-0",
                                    )
                            ]
                        ),
                    ],
                ), 
            ]
        )

app_widget = widgets.VBox(
        [
            v.AppBar( # 0
                elevation="4",
                class_="ma-4",
                rounded=True,
                children=[
                    v.Layout(
                        children=[
                            widgets.Image(
                                value=open(files("antakia.assets").joinpath("logo_ai-vidence.png"), "rb").read(), 
                                height=str(864 / 20) + "px", 
                                width=str(3839 / 20) + "px"
                            )
                            ],
                        class_="mt-1",
                    ),
                    v.Html(tag="h2", children=["AntakIA"], class_="ml-3"), # 01
                    v.Spacer(),
                    v.Btn( # backupBtn # 03 
                        icon=True, children=[v.Icon(children=["mdi-content-save"])], elevation=0
                    ),
                    v.Btn( # settingsBtn # 04
                        icon=True, children=[v.Icon(children=["mdi-tune"])], elevation=0
                    ),
                    v.Dialog( # 05
                        children=[
                            v.Card( # 050
                                children=[
                                    v.CardTitle( # 050 0
                                        children=[
                                            v.Icon(class_="mr-5", children=["mdi-cogs"]),
                                            "Settings",
                                        ]
                                    ),
                                    v.CardText( # 050 1
                                        children=[
                                        v.Row( # 050 10
                                            children=[
                                                v.Slider( # figureSizeSlider # 050 100
                                                    style_="width:20%",
                                                    v_model=700,
                                                    min=200,
                                                    max=1200,
                                                    label="With of both scattered plots (in pixels)",
                                                ), 
                                            widgets.IntText(
                                                value="700", disabled=True, layout=Layout(width="40%")
                                            )
                                            ],
                                            ),
                                        ]
                                        ),
                                        ]
                                    ),
                                ]
                    ),
                    v.Btn( # gotoWebBtn # 06
                        icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
                    ),
                ],
            ), 
            widgets.HBox( # 1
                [
                    v.Row( # 10
                        class_="ma-3",
                        children=[
                            v.Icon(children=["mdi-numeric-2-box"]),
                            v.Icon(children=["mdi-alpha-d-box"]),
                            v.Switch( # Dim switch # 102
                                class_="ml-3 mr-2",
                                v_model=False,
                                label="",
                            ),
                            v.Icon(children=["mdi-numeric-3-box"]),
                            v.Icon(children=["mdi-alpha-d-box"]),
                        ],
                    ),
                    v.Layout( # 11
                        class_="pa-2 ma-2",
                        elevation="3",
                            children=[
                                    v.Icon( # 110
                                        children=["mdi-format-color-fill"], class_="mt-n5 mr-4"
                                    ),
                                    v.BtnToggle( # colorChoiceBtnToggle # 111
                                        mandatory=True,
                                        v_model="Y",
                                        children=[
                                            v.Btn( # 1110
                                                icon=True,
                                                children=[v.Icon(children=["mdi-alpha-y-circle-outline"])],
                                                value="y",
                                                v_model=True,
                                            ),
                                            v.Btn( # 1111
                                                icon=True,
                                                children=[v.Icon(children=["mdi-alpha-y-circle"])],
                                                value="y^",
                                                v_model=True,
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[2]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-minus-box-multiple"])],
                                                value="residual",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[3]
                                                icon=True,
                                                children=[v.Icon(children="mdi-lasso")],
                                                value="current selection",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[4]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-ungroup"])],
                                                value="regions",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[5]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-select-off"])],
                                                value="not selected",
                                            ),
                                            v.Btn( # app_graph.children[1].children[1].children[1].children[6]
                                                icon=True,
                                                children=[v.Icon(children=["mdi-star"])],
                                                value="auto",
                                            ),
                                        ],
                                    ),
                                    v.Btn( # opacityBtn # 112
                                        icon=True,
                                        children=[v.Icon(children=["mdi-opacity"])],
                                        class_="ma-2 ml-6 pa-3",
                                        elevation="3",
                                    ),
                                    v.Select( # explanationSelect # 113
                                        label="Explanation method",
                                        items=[
                                            {'text': "SHAP (imported)", 'disabled': True },
                                            {'text': "SHAP (computed)", 'disabled': True },
                                            {'text': "LIME (imported)", 'disabled': True },
                                            {'text': "LIME (computed)", 'disabled': True }
                                            ],
                                        class_="ma-2 mt-1 ml-6",
                                        style_="width: 150px",
                                        disabled = False,
                                    ),
                                    v.Btn( # computeMenuBtnBtn # 114
                                        icon=True,
                                        children=[v.Icon(children=["mdi-opacity"])],
                                        class_="ma-2 ml-6 pa-3",
                                        elevation="3",
                                    )
                                ],
                    ),
                    v.Layout( # 12
                        class_="mt-3",
                        children=[
                            widgets.HBox( # 120
                                [
                                    v.Select( # projSelectVS # 1200
                                        label="Projection in the VS :",
                                        items=DimReducMethod.dimreduc_methods_as_str_list(),
                                        style_="width: 150px",
                                    ),
                                    v.Layout( # 120 1
                                        children=[
                                            v.Menu( # projSettingsMenuVS # 120 10
                                                v_slots=[
                                                {
                                                    "name": "activator",
                                                    "variable": "props",
                                                    "children": v.Btn(
                                                        v_on="props.on",
                                                        icon=True,
                                                        size="x-large",
                                                        children=[v.Icon(children=["mdi-cogs"], size="large")],
                                                        class_="ma-2 pa-3",
                                                        elevation="3",
                                                    ),
                                                }
                                                ],
                                                children=[
                                                    v.Card( # 120 100
                                                        class_="pa-4",
                                                        rounded=True,
                                                        children=[
                                                            widgets.VBox([ # ProjVS sliders # 120 100 0
                                                                v.Slider(
                                                                    v_model=10, min=5, max=30, step=1, label="Number of neighbours"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                v.Slider(
                                                                    v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                v.Slider(
                                                                    v_model=2, min=0.1, max=5, step=0.1, label="FP ratio"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                ],
                                                            )
                                                            ],
                                                        min_width="500",
                                                    )
                                                ],
                                            v_model=False,
                                            close_on_content_click=False,
                                            offset_y=True,
                                            )
                                        ]
                                    ),
                                    widgets.HBox( # 120 2
                                        [ 
                                        v.ProgressCircular(  # 120 20
                                            indeterminate=True, color="blue", width="6", size="35", class_="mx-4 my-3"
                                        )
                                        ]),
                                ]
                            ),
                            widgets.HBox(  # 121
                                [
                                    v.Select( # projSelectES # 121 0
                                        label="Projection in the ES :",
                                        items=DimReducMethod.dimreduc_methods_as_str_list(),
                                        style_="width: 150px",
                                    ),
                                    v.Layout( # 121 1
                                        children=[
                                            v.Menu( # projSettingsMenuES # 121 10
                                                v_slots=[
                                                {
                                                    "name": "activator",
                                                    "variable": "props",
                                                    "children": v.Btn(
                                                        v_on="props.on",
                                                        icon=True,
                                                        size="x-large",
                                                        children=[v.Icon(children=["mdi-cogs"], size="large")],
                                                        class_="ma-2 pa-3",
                                                        elevation="3",
                                                    ),
                                                }
                                                ],
                                                children=[
                                                    v.Card( # 121 100
                                                        class_="pa-4",
                                                        rounded=True,
                                                        children=[
                                                            widgets.VBox([ # ProjES sliders # 121 100 0
                                                                v.Slider(
                                                                    v_model=10, min=5, max=30, step=1, label="Number of neighbours"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                v.Slider(
                                                                    v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                v.Slider(
                                                                    v_model=2, min=0.1, max=5, step=0.1, label="FP ratio"
                                                                ),
                                                                v.Html(class_="ml-3", tag="h3", children=["#"]),
                                                                ],
                                                            )
                                                            ],
                                                        min_width="500",
                                                    )
                                                ],
                                            v_model=False,
                                            close_on_content_click=False,
                                            offset_y=True,
                                            )
                                        ]
                                    ),
                                    widgets.HBox( # 121 2
                                        [
                                        v.ProgressCircular( # ESBusyBox # 121 20
                                            indeterminate=True, color="blue", width="6", size="35", class_="mx-4 my-3"
                                        )
                                        ]),
                                ]
                            ),
                        ],
                    )
                ],
                layout=Layout(
                    width="100%",
                    display="flex",
                    flex_flow="row",
                    justify_content="space-around",
                ),
            ),
            widgets.VBox( # 2
                [
                widgets.HBox( # 20
                    [
                    widgets.VBox( # 200
                            [
                                widgets.HTML("<h3>Values Space<h3>"), # 2000
                                widgets.Text(value='A FigureWidget will be inserted here by the app'), # 2001
                                
                            ],
                            layout=Layout(
                                display="flex", align_items="center", margin="0px 0px 0px 0px"
                            ),
                    ),
                    widgets.VBox( #  #201
                            [
                                widgets.HTML("<h3>Explanations Space<h3>"),  # 201 0
                                widgets.Text(value='A FigureWidget will be inserted here by the app'), # 2011
                            ],
                            layout=Layout(
                                display="flex", align_items="center", margin="0px 0px 0px 0px"
                            )
                    )
                    ],
                    layout=Layout(width="100%")
                    )
            ]    
            ),
            v.Container( # antakiaMethodCard # 3
                fluid = True,
                children=[
                    v.Tabs( # 30
                        v_model=0, # default active tab
                        children=
                        [
                            v.Tab(children=["1. Selection"]),  # 300
                            v.Tab(children=["2. Refinement"]), # 301
                            v.Tab(children=["3. Sub-model"]), # 302
                            v.Tab(children=["4. Regions"]) # 303
                        ] 
                        + 
                        [
                            v.TabItem(  # Tab 1) = tabOneSelectionColumn ? Selection # 304
                                children=[
                                    v.Card( # selectionCard # 304 0
                                        class_="ma-2",
                                        elevation=0,
                                        children=[
                                            v.Layout( # 304 00
                                                children=[
                                                    v.Icon(children=["mdi-lasso"]), # 304 000
                                                    v.Html( # 304 001
                                                        class_="mt-2 ml-4",
                                                        tag="h4",
                                                        children=[ 
                                                            "0 point selected : use the lasso tool on the figures above or use the auto-selection tool below" #304 001 0
                                                        ],
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                    v.ExpansionPanels( # out_accordion # 304 1
                                        class_="ma-2",
                                        children=[
                                            v.ExpansionPanel( # 304 10
                                                children=[
                                                    v.ExpansionPanelHeader( # 304 100
                                                        children=["Data selected"]), #304 100 0
                                                    v.ExpansionPanelContent( # 304 101
                                                        children=[
                                                        v.Alert( # out_selec_all # 304 101 0
                                                            max_height="400px",
                                                            style_="overflow: auto",
                                                            elevation="0",
                                                            children=[
                                                                v.Row( # 304 101 00
                                                                    class_="d-flex flex-row justify-space-between",
                                                                    children=[
                                                                        v.Layout( # out_selec # 304 101 000
                                                                            style_="min-width: 100%; max-width: 94%",
                                                                            children=[
                                                                                v.Html( # out_selec # 304 101 000 0
                                                                                    tag="h4",
                                                                                    children=["Select points on the figure to see their values ​​here"], # 304 101 000 00
                                                                                )
                                                                            ],
                                                                        )
                                                                    ],
                                                                ),
                                                            ],
                                                        ),
                                                        ]),
                                                ]
                                            )
                                        ],
                                    ),
                                    v.Layout( # clusterGrp # 304 2
                                        class_="d-flex flex-row",
                                        children=[
                                            v.Btn( # findClusterBtn # 304 20
                                                class_="ma-1 mt-2 mb-0",
                                                elevation="2",
                                                children=[v.Icon(children=["mdi-magnify"]), "Find clusters"],
                                            ),
                                            v.Checkbox( # clusterCheck # 304 21
                                                v_model=True, label="Optimal number of clusters :", class_="ma-3"
                                            ),
                                            v.Slider( # clustersSlider # 304 22
                                                style_="width : 30%",
                                                class_="ma-3 mb-0",
                                                min=2,
                                                max=20,
                                                step=1,
                                                v_model=3,
                                                disabled=True,
                                            ),
                                            v.Html( # clustersSliderTxt # 304 23
                                                tag="h3",
                                                class_="ma-3 mb-0",
                                                children=["Number of clusters #"],
                                            ),
                                        ],
                                    ),
                                    v.ProgressLinear( # loadingClustersProgLinear # 304 3
                                        indeterminate=True, class_="ma-3", style_="width : 100%"
                                    ),
                                    v.Row( # clusterResults # 304 4
                                        children=[
                                            v.Layout(
                                                class_="flex-grow-0 flex-shrink-0",
                                                children=[
                                                    v.Btn(class_="d-none", elevation=0, disabled=True
                                                    )], # 304 40
                                            ),
                                            v.Layout(  # 304 41
                                                class_="flex-grow-1 flex-shrink-0",
                                                children=[ 
                                                    widgets.Text(value='A v.DataTable will be inserted here by the app'), # A v.DataTable is inserted here by the app. Will be : # cluster_results_table # 304 410
                                                    ],
                                            ),
                                        ],
                                    ),
                                    v.Layout( # magicGUI 304 5
                                        class_="d-flex flex-row justify-center align-center",
                                        children=[
                                            v.Spacer(), # 304 50
                                            v.Btn( # magicBtn # findClusterBtn # 304 51
                                                    class_="ma-3",
                                                    children=[
                                                        v.Icon(children=["mdi-creation"], class_="mr-3"),
                                                        "Magic button",
                                                    ],
                                            ),
                                            v.Checkbox( # # magicCheckBox 304 52
                                                v_model=True, label="Demonstration mode", class_="ma-4"), 
                                            v.TextField( # 304 53
                                                class_="shrink",
                                                type="number",
                                                label="Time between the steps (ds)",
                                                v_model=10,
                                            ),
                                            v.Spacer(), # 304 54
                                        ],
                                    )
                                ]
                            ), 
                            v.TabItem( # Tab 2) = tabTwoSkopeRulesColumn ? Refinement # 305
                                children=[
                                    v.Col( # 305 0
                                        children=[
                                            widgets.VBox( # skopeBtnsGrp # 305 00
                                                [
                                                v.Layout( # skopeBtns # 305 000
                                                    class_="d-flex flex-row",
                                                    children=[
                                                        v.Btn( # validateSkopeBtn # 305 000 0
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(class_="mr-2", children=["mdi-auto-fix"]),
                                                                "Skope-Rules",
                                                            ],
                                                        ),
                                                        v.Btn( # reinitSkopeBtn # 305 000 1
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(class_="mr-2", children=["mdi-skip-backward"]),
                                                                "Come back to the initial rules",
                                                            ],
                                                        ),
                                                        v.Spacer(), # 305 000 2
                                                        v.Checkbox( # beeSwarmCheck # 305 000 3
                                                            v_model=True,
                                                            label="Show Shapley's beeswarm plots",
                                                            class_="ma-1 mr-3",
                                                        )
                                                        ,
                                                    ],
                                                ),
                                                v.Layout( # skopeText # skopeBtns # 305 001
                                                    class_="d-flex flex-row",
                                                    children=[
                                                        v.Card( # ourVSSkopeText # skopeBtns # 305 001 0
                                                            style_="width: 50%;",
                                                            class_="ma-3",
                                                            children=[
                                                                v.Row(  # 30500100
                                                                    class_="ml-4",
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), # 305 010 00
                                                                        v.CardTitle(children=["Rules applied to the Values Space"]), # 305 010 01
                                                                        v.Spacer(), # 305 010 02
                                                                        v.Html( # 305 010 03
                                                                            class_="mr-5 mt-5 font-italic",
                                                                            tag="p",
                                                                            children=["precision = /"],
                                                                        ),
                                                                    ],
                                                                ),
                                                                v.Card( # ourVSSkopeCard # 305 001 01
                                                                    class_="mx-4 mt-0",
                                                                    elevation=0,
                                                                    children=[
                                                                        v.CardText(
                                                                            children=[
                                                                                v.Row(
                                                                                    class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                                                                                    children=[
                                                                                        "Waiting for the skope-rules to be applied...",
                                                                                    ],
                                                                                )
                                                                            ]
                                                                        )
                                                                    ],
                                                                )
                                                            ],
                                                        ),
                                                        v.Card( # ourESSkopeText # 305 001 1
                                                            style_="width: 50%;",
                                                            class_="ma-3",
                                                            children=[
                                                                v.Row( # 305 001 10
                                                                    class_="ml-4",
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), # 305 001 100
                                                                        v.CardTitle(children=["Rules applied on the Explanatory Space"]), 
                                                                        v.Spacer(),
                                                                        v.Html( # 305 001 103
                                                                            class_="mr-5 mt-5 font-italic",
                                                                            tag="p",
                                                                            children=["precision = /"],
                                                                        ),
                                                                    ],
                                                                ),
                                                                v.Card( # ourESSkopeCard # 305 001 11
                                                                    class_="mx-4 mt-0",
                                                                    elevation=0,
                                                                    # style_="width: 100%;",
                                                                    children=[
                                                                        v.CardText(
                                                                            children=[
                                                                                v.Row(
                                                                                    class_="font-weight-black text-h5 mx-10 px-10 d-flex flex-row justify-space-around",
                                                                                    children=[
                                                                                        "Waiting for the Skope-rules to be applied...",
                                                                                    ],
                                                                                )
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                        )
                                                    ]
                                                ) # End v.Layout / skopeText
                                            ]
                                            ), # End VBox / skopeBtnsGrp
                                            widgets.VBox( # skopeAccordion # 305 01
                                                children=[ # RuleVariableRefiner objects are inserted here by the app
                                                    widgets.Text(value='A RuleVariableRefiner will be inserted here by the app'),
                                                    widgets.Text(value='A RuleVariableRefiner will be inserted here by the app'),
                                                    widgets.Text(value='A RuleVariableRefiner will be inserted here by the app'), 
                                                ],
                                                layout=Layout(width="100%", height="auto"),
                                            ), # End of VBox 30501
                                            v.Row( #addButtonsGrp # 305 02
                                                children=[
                                                    v.Btn( # addSkopeBtn # 305 020
                                                        class_="ma-4 pa-2 mb-1",
                                                        children=[v.Icon(children=["mdi-plus"]), "Add a rule"],
                                                    ), 
                                                    v.Select( # addAnotherFeatureWgt # 305 021
                                                        class_="mr-3 mb-0",
                                                        explanationsMenuDict=["/"],
                                                        v_model="/",
                                                        style_="max-width : 15%",
                                                    ), 
                                                    v.Spacer(), # 305 022
                                                    v.Btn( # addMapBtn # 305 023
                                                        class_="ma-4 pa-2 mb-1",
                                                        children=[v.Icon(class_="mr-4", children=["mdi-map"]), "Display the map"],
                                                        color="white",
                                                        disabled=True,
                                                    ),
                                                    ]
                                            ),
                                            ]
                                    )
                                ]
                            ), 
                            v.TabItem( # Tab 3) = tabThreeSubstitutionVBox ? # 306
                                children=[
                                        widgets.VBox( # 306 0
                                            [
                                                v.ProgressLinear( # loadingModelsProgLinear # 306 00
                                                    indeterminate=True,
                                                    class_="my-0 mx-15",
                                                    style_="width: 100%;",
                                                    color="primary",
                                                    height="5",
                                                ), 
                                                v.SlideGroup( # subModelslides # 306 01
                                                    v_model=None,
                                                    class_="ma-3 pa-3",
                                                    elevation=4,
                                                    center_active=True,
                                                    show_arrows=True,
                                                    children=
                                                    [
                                                        v.SlideItem( # 306 010 # dummy SlideItem. Will be replaced by the app
                                                            # style_="width: 30%",
                                                            children=[
                                                                v.Card(
                                                                    class_="grow ma-2",
                                                                    children=[
                                                                        v.Row(
                                                                            class_="ml-5 mr-4",
                                                                            children=[
                                                                                v.Icon(children=["a name"]),
                                                                                v.CardTitle(
                                                                                    children=["model foo"]
                                                                                ),
                                                                            ],
                                                                        ),
                                                                        v.CardText(
                                                                            class_="mt-0 pt-0",
                                                                            children=["Model's score"],
                                                                        ),
                                                                    ],
                                                                )
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                ]
                                        )
                                    ]
                            ),
                            v.TabItem( # Tab 4) = tabFourRegionListVBox # 307
                                children=[
                                    v.Col( # 307 0
                                    children=[
                                        widgets.VBox( # 307 00
                                            [
                                                v.Btn( # 307 000
                                                        class_="ma-4 pa-2 mb-1",
                                                        children=[v.Icon(class_="mr-4", children=["mdi-map"]), "Validate the region"],
                                                        color="white",
                                                        disabled=True,
                                                )
                                        ]
                                        ),         
                                    ]
                                    )
                                ]
                            )
                        ]
                    )
                ],
                class_="mt-0",
                outlined=True
            )
        ]
    )

