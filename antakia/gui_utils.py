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
from antakia.rules import Rule
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
    get_widget_at_address(root_widget, '0') returns root_widgetn first child
    TODO : allow childhood rank > 9
    """
    try:
        int(address)
    except ValueError:
        raise ValueError(address, "must be a string composed of digits")

    if len(address) > 1:
        try:
            return get_widget_at_address(root_widget.children[int(address[0])], address[1:])
        except IndexError:
            raise IndexError(f"Nothing found @{address} in this {root_widget.__class__.__name__}")
    else:
        return root_widget.children[int(address[0])]


def _get_parent(root_widget: Widget, address: str) -> Widget:
    return get_widget_at_address(root_widget, address[:-1])


def is_ipyvuetify(widget: Widget) -> bool:
    return widget.__module__.split('.')[0] == 'ipyvuetify'


def change_widget(root_widget: Widget, address: str, sub_widget: Widget):
    """
    Substitutes a sub_widget in a root_widget.
    Address is a sequence of childhood ranks as a string, root_widget first child address is  '0'
    The root_widget is altered but the object remains the same
    """

    if len(address) > 1:
        parent_widget = _get_parent(root_widget, address)

        # Because ipywidgets store their children in a tuple (vs list):
        if is_ipyvuetify(parent_widget):
            parent_children_clone = copy(parent_widget.children)
        else:
            parent_children_clone = copy(list(parent_widget.children))


        try:
            parent_children_clone[int(address[-1])] = sub_widget
        except IndexError:
            raise IndexError(f"Nothing found @{address} in this {type(root_widget)}")

        if is_ipyvuetify(parent_widget):
            try:        
                parent_widget.children = parent_children_clone
            except TraitError:
                raise TraitError(
                    f"{type(parent_widget)} cannot have for children {parent_children_clone}"
                )
        else:
            parent_widget.children = tuple(parent_children_clone)
        
        change_widget(root_widget, address[:-1], parent_widget)
    else:
        parent_children_clone = copy(list(root_widget.children))
        parent_children_clone[int(address)] = sub_widget
        if is_ipyvuetify(root_widget):
            root_widget.children = tuple(parent_children_clone)
        else:
            root_widget.children = parent_children_clone


def create_rule_card(object) -> list:
    return None


def datatable_from_Selection(sel: list, length: int) -> v.Row:
    """Returns a DataTable from a list of Selections"""
    new_df = []

    for i in range(len(sel)):
        new_df.append(
            [
                i + 1,
                sel[i].size(),
                str(
                    round(
                        sel[i].size() / length * 100,
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
    columns = [{"text": c, "sortable": False, "value": c} for c in new_df.columns]
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
        color = sns.color_palette("viridis", size * coeff).as_hex()[round(i)]
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
            v.Layout(class_="flex-grow-0 flex-shrink-0", children=[radio_group]),
            v.Layout(class_="flex-grow-0 flex-shrink-0", children=[chips_col]),
            v.Layout(
                class_="flex-grow-1 flex-shrink-0",
                children=[datatable],
            ),
        ],
    )


class HighDimExplorer:
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
    figure_2D and figure_3D : FigureWidget
        Plotly scatter plot
    _selection_disabled : bool
    container : a thin v.Container wrapper around the current Figure. Allows us to swap between 2D and 3D figures alone (without GUI)
    _projection_select : v.Select, with mutliple dimreduc methods
    _progress_circular : v.ProgressCircular, used to display the progress of the computation
    _projection_slider_VBoxes : dict of VBox,  parameters for dimreduc methods
    fig_size

    """

    def __init__(
        self,
        space_name: str,
        dataframes_list: list,
        labels_list: list,
        is_computable_list: list,
        y: pd.Series,
        init_proj: int,
        init_dim: int,
        fig_size: int,
        border_size: int,
        selection_changed: callable,
        new_values_wanted: callable = None,
    ):
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

        if not (len(dataframes_list) == len(labels_list) == len(is_computable_list)):
            raise ValueError(
                f"HDE.init: values_list, labels_list and is_computable_list must have the same length"
            )

        self.pv_list = []
        for index, values in enumerate(dataframes_list):
            if values is not None:
                self.pv_list.append(ProjectedValues(values))
            else:
                self.pv_list.append(None)
        self._y = y

        self.current_pv = 0
        for i in range(len(self.pv_list)):
            if self.pv_list[i] is not None:
                self.current_pv = i
                break

        self._projection_select = v.Select(
            label="Projection in the " + space_name,
            items=DimReducMethod.dimreduc_methods_as_str_list(),
            style_="width: 150px",
        )
        self._projection_select.on_event("change", self.projection_select_changed)

        # We initiate it in grey, not indeterminate :
        self._progress_circular = v.ProgressCircular(
            color="grey", width="6", size="35", class_="mx-4 my-3", v_model=100
        )

        # Since HDE is responsible for storing its current proj, we check init value :
        if init_proj not in DimReducMethod.dimreduc_methods_as_list():
            raise ValueError(
                f"HDE.init: {init_proj} is not a valid projection method code"
            )
        self._projection_select.v_model = DimReducMethod.dimreduc_method_as_str(
            init_proj
        )

        self._projection_slider_VBoxes = {}
        # We know PaCMAP uses these parameters :
        self._projection_slider_VBoxes[DimReducMethod.PaCMAP] = widgets.VBox(
            [
                v.Slider(
                    v_model=10, min=5, max=30, step=1, label="Number of neighbours"
                ),
                # v.Html(class_="ml-3", tag="h3", children=["machin"]),
                v.Slider(v_model=0.5, min=0.1, max=0.9, step=0.1, label="MN ratio"),
                # v.Html(class_="ml-3", tag="h3", children=["truc"]),
                v.Slider(v_model=2, min=0.1, max=5, step=0.1, label="FP ratio"),
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
                select_items.append(
                    {"text": labels_list[i], "disabled": self.pv_list[i] is None}
                )

            self._values_select = v.Select(
                label="Explanation method",
                items=select_items,
                class_="ma-2 mt-1 ml-6",
                style_="width: 150px",
                disabled=False,
            )
            self._values_select.on_event("change", self._values_select_changed)

            computable_labels_list = [
                labels_list[i] for i in range(len(labels_list)) if is_computable_list[i]
            ]
            tab_list = [v.Tab(children=label) for label in computable_labels_list]
            content_list = [
                v.TabItem(
                    children=[
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
                                    v_model="0.00% [0/?] - 0m0s (estimated time : /min /s)",
                                    readonly=True,
                                ),
                                v.Btn(
                                    children=[
                                        v.Icon(
                                            class_="mr-2",
                                            children=["mdi-calculator-variant"],
                                        ),
                                        "Compute values",
                                    ],
                                    class_="ma-2 ml-6 pa-3",
                                    elevation="3",
                                    v_model=label,
                                    color="primary",
                                ),
                            ],
                        )
                    ]
                )
                for label in computable_labels_list
            ]

            self._compute_menu = v.Menu(
                v_slots=[
                    {
                        "name": "activator",
                        "variable": "props",
                        "children": v.Btn(
                            v_on="props.on",
                            icon=True,
                            size="x-large",
                            children=[
                                v.Icon(children=["mdi-timer-sand"], size="large")
                            ],
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
                            widgets.VBox(
                                [v.Tabs(v_model=0, children=tab_list + content_list)],
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
            get_widget_at_address(self._compute_menu, "000203").on_event(
                "click", self.compute_btn_clicked
            )
            # LIME compute button :
            get_widget_at_address(self._compute_menu, "000303").on_event(
                "click", self.compute_btn_clicked
            )

        #  Now we can init figures 2 and 3D
        self.fig_size = fig_size
        self._selection_disabled = False

        self.container = v.Container()

        self.create_figure(2)
        self.create_figure(3)

        

    # ---- Methods ------

    def __str__(self) -> str:
        return (
            "HighDimExplorer : values space (VS)"
            if len(self.pv_list) == 1
            else "HighDimExplorer : explanations space (ES)"
        )

    def display_rules(self, df_ids_list: list):
        # We add a second trace (Scatter) to the figure to display the rules
        if df_ids_list is None or len(df_ids_list) == 0:
            # We remove the 'rule trace' if exists
            if len(self.figure_2D.data) > 1:
                if self.figure_2D.data[1] is not None:
                    # It seems impossible to remove a trace from a figure once created
                    # So we hide or update&show this 'rule_trace'
                    self.figure_2D.data[1].visible = False
        else:
            # Let's create a color scale with rule postives in blue and the rest in grey
            colors = ["grey"] * self.pv_list[self.current_pv].get_length()
            # IMPORTANT : we convert df_ids to row_ids (dataframe may not be indexed by row !!)
            row_ids_list = Rule.indexes_to_rows(
                self.pv_list[self.current_pv].X, df_ids_list
            )
            for row_id in row_ids_list:
                colors[row_id] = "blue"

            if len(self.figure_2D.data) == 1:
                # We need to add a 'rule_trace'
                self.figure_2D.add_trace(
                    Scatter(
                        x=self.pv_list[self.current_pv].get_proj_values(
                            self._get_projection_method(), 2
                        )[0],
                        y=self.pv_list[self.current_pv].get_proj_values(
                            self._get_projection_method(), 2
                        )[1],
                        mode="markers",
                        marker=dict(
                            color=colors,
                        ),
                    )
                )

            elif len(self.figure_2D.data) == 2:
                # We replace the existing 'rule_trace'

                with self.figure_2D.batch_update():
                    self.figure_2D.data[1].x = self.pv_list[
                        self.current_pv
                    ].get_proj_values(self._get_projection_method(), 2)[0]
                    self.figure_2D.data[1].y = self.pv_list[
                        self.current_pv
                    ].get_proj_values(self._get_projection_method(), 2)[1]
                    self.figure_2D.layout.width = self.fig_size
                    self.figure_2D.data[1].marker.color = colors

                self.figure_2D.data[1].visible = True  # in case it was hidden
            else:
                raise ValueError(
                    f"HDE.{'VS' if len(self.pv_list) == 1 else 'ES'}.display_rules : to be debugged : the figure has {len(self.figure_2D.data)} traces, not 1 or 2"
                )

    def check(self) -> str:
        """
        Debug info
        """
        text = (
            self._get_space_name()
            + ", in ("
            + DimReducMethod.dimreduc_method_as_str(self._get_projection_method())
            + ", "
            + str(self._current_dim)
            + ") projection\n"
        )

        text += (
            str(len(self.pv_list)) + " PVs. PV[0]'s is " + str(self.pv_list[0]) + "\n"
        )
        text += (
            "_figure_2D has its len(x) = "
            + str(len(self.figure_2D.data[0].x))
            + ", len(y) = "
            + str(len(self.figure_2D.data[0].y))
            + "\n"
        )
        text += (
            "_figure_3D has its len(x) = "
            + str(len(self.figure_3D.data[0].x))
            + ", len(y) = "
            + str(len(self.figure_3D.data[0].y))
            + ", len (z) = "
            + str(len(self.figure_3D.data[0].z))
            + "\n"
        )
        return text

    def create_figure(self, dim: int) -> FigureWidget:
        """
        Called by __init__ and by set_selection
        Builds and  returns the FigureWidget for the given dimension
        """

        x = y = z = [0, 1, 2]  # dummy data

        if self.pv_list[self.current_pv] is not None:
            proj_values = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), dim
            )
            if proj_values is not None:
                x = self.pv_list[self.current_pv].get_proj_values(
                    self._get_projection_method(), dim
                )[0]
                y = self.pv_list[self.current_pv].get_proj_values(
                    self._get_projection_method(), dim
                )[1]
                if dim == 3:
                    z = self.pv_list[self.current_pv].get_proj_values(
                        self._get_projection_method(), dim
                    )[2]

        if len(self.pv_list) == 1:
            hde_marker = dict(
                color=self._y,
                colorscale="Viridis",
                colorbar=dict(  # only VS marker has a colorbar
                    title="y",
                    thickness=20,
                ),
            )
        else:
            hde_marker = dict(color=self._y, colorscale="Viridis")
        if dim == 2:
            fig = FigureWidget(
                data=Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=hde_marker,
                    customdata=self._y,
                    hovertemplate="%{customdata:.3f}",
                )
            )
            fig.data[0].on_selection(self._selection_event)
            fig.data[0].on_deselect(self._deselection_event)
            fig.update_layout(dragmode=False if self._selection_disabled else "lasso")
            fig.update_traces(
                selected={"marker": {"opacity": 1.0}},
                unselected={"marker": {"opacity": 0.1}},
                selector=dict(type="scatter"),
            )
        else:
            fig = FigureWidget(
                data=Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=hde_marker,
                    customdata=self._y,
                    hovertemplate="%{customdata:.3f}",
                )
            )

        fig.update_layout(margin=dict(t=0), width=self.fig_size)
        fig._config = fig._config | {"displaylogo": False}

        if dim == 2:
            self.figure_2D = fig
        else:
            self.figure_3D = fig
        
        self.container.children = [self.figure_2D if self._current_dim == 2 else self.figure_3D]
        

    def set_selection_disabled(self, mode: bool):
        self._selection_disabled = mode
        self.figure_2D.update_layout(
            dragmode=False if self._selection_disabled else "lasso"
        )

    def compute_projs(self, callback: callable = None):
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
            projected_dots_2D = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), 2
            )
            projected_dots_3D = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), 3
            )

        if projected_dots_2D is None:
            self.pv_list[self.current_pv].set_proj_values(
                self._get_projection_method(),
                2,
                compute.compute_projection(
                    self.pv_list[self.current_pv].X,
                    self._get_projection_method(),
                    2,
                    callback,
                ),
            )

            self._redraw_figure(self.figure_2D)

        if projected_dots_3D is None:
            self.pv_list[self.current_pv].set_proj_values(
                self._get_projection_method(),
                3,
                compute.compute_projection(
                    self.pv_list[self.current_pv].X,
                    self._get_projection_method(),
                    3,
                    callback,
                ),
            )
            self._redraw_figure(self.figure_3D)

    def update_progress_circular(
        self, caller: DimReducMethod, progress: int, duration: float
    ):
        """
        Each proj computation consists in 2 (2D and 3D) tasks.
        So progress of each task in divided by 2 and summed together
        """
        if self._progress_circular.color == "grey":
            self._progress_circular.color = "blue"
            # Since we don't have fine-grained progress, we set it to 'indeterminate'
            self._progress_circular.indeterminate = True
            # But i still need to store total progress in v_model :
            self._progress_circular.v_model = 0
            # We lock it during computation :
            self._projection_select.disabled = True

        # Strange sicen we're in 'indeterminate' mode, but i need it, cf supra
        self._progress_circular.v_model = self._progress_circular.v_model + round(
            progress / 2
        )

        if self._progress_circular.v_model == 100:
            self._progress_circular.indeterminate = False
            self._progress_circular.color = "grey"
            self._projection_select.disabled = False

    def projection_select_changed(self, widget, event, data):
        """ "
        Called when the user changes the projection method
        If needed, we compute the new projection
        """
        self._projection_select.disabled = True
        self.compute_projs(self.update_progress_circular)  # to ensure we got the values
        self._projection_select.disabled = False
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

    def update_progress_linear(self, method: int, progress: int, duration: float):
        """
        Called by the computation process (SHAP or LUME) to udpate the progress linear
        """
        progress_linear = None
        if method == ExplanationMethod.SHAP:
            progress_linear = get_widget_at_address(self._compute_menu, "000201")
        else:
            progress_linear = get_widget_at_address(self._compute_menu, "000301")

        progress_linear.v_model = progress

        if progress == 100:
            tab = None
            if method == ExplanationMethod.SHAP:
                tab = get_widget_at_address(self._compute_menu, "0000")
            else:
                tab = get_widget_at_address(self._compute_menu, "0001")
            tab.disabled = True

    def set_dimension(self, dim: int):
        # Dimension is stored in the instance variable _current_dim
        """
        At runtime, GUI calls this function and swap our 2 and 3D figures
        """
        self._current_dim = dim
        self.container.children = [self.figure_2D] if dim == 2 else [self.figure_3D]

    def _get_projection_method(self) -> int:
        # proj is stored in the proj Select widget
        """
        Returns the current projection method
        """
        return DimReducMethod.dimreduc_method_as_int(self._projection_select.v_model)

    def redraw(self, color: pd.Series = None, opacity_values: pd.Series = None):
        """
        Redraws the 2D and 3D figures. FigureWidgets are not recreated.
        """
        self._redraw_figure(self.figure_2D, color)
        self._redraw_figure(self.figure_3D, color)

    def _redraw_figure(
        self,
        fig: FigureWidget,
        color: pd.Series = None,
        opacity_values: pd.Series = None,
    ):
        dim = (
            2 if fig == self.figure_2D else 3
        )  # dont' use self._current_dim: it may be 3D while we want to redraw figure_2D

        with fig.batch_update():
            fig.data[0].x = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), dim
            )[0]
            fig.data[0].y = self.pv_list[self.current_pv].get_proj_values(
                self._get_projection_method(), dim
            )[1]
            if fig == self.figure_3D:
                fig.data[0].z = self.pv_list[self.current_pv].get_proj_values(
                    self._get_projection_method(), dim
                )[2]
            fig.layout.width = self.fig_size
            if color is not None:
                fig.data[0].marker.color = color
            if opacity_values is not None:
                fig.data[0].marker.opacity = opacity_values
            fig.data[0].customdata = color

    def _selection_event(self, trace, points, selector, *args):
        """Called whenever the user selects dots on the scatter plot"""
        # We don't call GUI.selection_changed if 'selectedpoints' length is 0 : it's handled by -deselection_event
        # We convert Plotly 'selectedpoints' in our Dataframe indexes
        if (
            self.figure_2D.data[0]["selectedpoints"] is not None
            and len(self.figure_2D.data[0]["selectedpoints"]) > 0
        ):
            # Note that if a selection is done on the ES HDE, it will have the explained values as X
            self.selection_changed(
                self,
                Rule.rows_to_indexes(
                    self.get_current_X(), list(trace["selectedpoints"])
                ),
            )

    def _deselection_event(self, trace, points, append: bool = False):
        """Called on deselection"""
        self.selection_changed(self, [])

    def _scatter_clicked(self, trace, points, selector, *args):
        logger.debug(f"HDE._scatter_clicked : {self._get_space_name()} entering ... ")

    def set_selection(self, new_selection_indexes: list):
        """
        Called by tne UI when a new selection occured on the other HDE
        """
        # If We alread have selected point, we create another figure # TODO : improve
        if (
            self.figure_2D.data[0]["selectedpoints"] is not None
            and len(self.figure_2D.data[0]["selectedpoints"]) > 0
        ):
            self.figure_2D = self.create_figure(2)
            # And I add the newly created figure to our container :
            if self._current_dim == 2:
                self.container.children = [self.figure_2D]

        if new_selection_indexes is not None and len(new_selection_indexes) > 0:
            # We set the new selection
            # Our app "speaks" in Dataframe indexes. So we convert them in Plotly row numbers
            self.figure_2D.update_traces(
                selectedpoints=Rule.indexes_to_rows(
                    self.get_current_X(), new_selection_indexes
                )
            )
        else:
            # No selection
            self.figure_2D.update_traces(selectedpoints=None)

    def get_projection_select(self):
        return self._projection_select

    def get_projection_prog_circ(self) -> v.ProgressCircular:
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

    def _get_space_name(self) -> str:
        """
        For debug purposes only. Not very reliable.
        """
        # TODO : remove
        return "VS" if len(self.pv_list) == 1 else "ES"

    def is_value_space(self) -> bool:
        return len(self.pv_list) == 1

    def _label_to_int(self, label: str) -> int:
        """
        Returns the index of a PV in the Select items
        """
        return [item["text"] for item in self._values_select.items].index(label)

    def get_current_X(self) -> pd.DataFrame:
        return self.pv_list[self.current_pv].X


class RuleWidget:
    """
    A piece of UI handled by a RulesExplorer
    It allows the user to modify one rule

    Attributes :
    rule : Rule
    values_space : bool
    expansion_panel_widget : its widget representation
    """

    def __init__(self, rule: Rule, X: pd.DataFrame, values_space: bool):
        self.rule = rule
        self.X = X
        self.values_space = values_space
        
        self.expansion_panel_widget = get_widget_at_address(app_widget, "3050101" if self.values_space else "3050111")
        
        if self.rule.is_categorical_rule():
            change_widget(self.expansion_panel_widget, "00", f"{self.rule.variable.symbol} possible values :")
            # We use a multiple select widget
            widhet = v.Select(
                label=self.rule.variable.symbol,
                items=self.rule.cat_values,
                style_="width: 150px",
                multiple=True,
                )
        # Rules on continuous variables :
        if not self.rule.is_inner_interval_rule():
            if self.rule.min is None: # var < max
                change_widget(self.expansion_panel_widget, "00", f"{self.rule.variable.symbol} lesser than {'or equal to ' if self.rule.operator_max == 0 else ''}:")
                widget = v.Slider(
                    class_="ma-3",
                    v_model=[self.rule.max],
                    min=-5, # TODO : easy to set : min(var)
                    max=self.rule.max*1.5, # TODO : easy to set : max(var)
                    step=0.1, # TODO we could divide the spread by 50 ?
                    thumb_label="always"
                    )
            else: # var > min
                change_widget(self.expansion_panel_widget, "00", f"{self.rule.variable.symbol} greater than {'or equal to ' if self.rule.operator_min == 4 else ''}:")
                widget = v.Slider(
                    class_="ma-3",
                    v_model=[self.rule.min],
                    min=-self.rule.min*1.5, # TODO set according to the variable distribution
                    max=2,
                    step=0.1, # TODO set according to the variable distribution
                    thumb_label="always"
                    )
        else: # We represent an intervel rule
            if self.rule.is_inner_interval_rule():
                widget = v.RangeSlider(
                    class_="ma-3",
                    v_model=[
                        self.rule.min,
                        self.rule.max,
                    ],
                    # min=-5,
                    # max=5,
                    step=0.1,
                    thumb_label="always"
                    )
            else: # We have a outter interval rule
                widget = v.RangeSlider(
                    class_="ma-3",
                    v_model=[
                        self.rule.min,
                        self.rule.max,
                    ],
                    # min=-5,
                    # max=5,
                    step=0.1,
                    thumb_label="always"
                )

        widget.on_event("change", self.widget_value_changed)
        change_widget(self.expansion_panel_widget, "0101", widget)

        # TODO : populate our figure !!

    def widget_value_changed(self, widget, event, data):
        if isinstance(widget, v.Select):
            logger.debug(f"RW.widget_value_changed : data = {data}")
        elif isinstance(widget, v.Slider):
            # Simple rule
            logger.debug(f"RW.widget_value_changed : data = {data}")
        else: # Interval rule
            logger.debug(f"RW.widget_value_changed : data = {data}")


class RulesWidget:
    """
    A RulesWidget is a piece of GUI that allows the user to refine a set of rules.
    The user can use the slider to change the rules.
    There are 2 RW : VS and ES slides

    rules_db : a dict of list :
        [[rules_list, scores]]], so that at iteration i we have :
        [i][0] : the list of rules
        [i][1] : the list of scores
    current_index : int refers to rules_db

    X : pd.DataFrame, values or explanations Dataframe depending on the context
    variables : list of Variable
    is_value_space : bool

    rules_updated : callable of the GUI parent
    vbox_wiget : its widget representation
    """

    def __init__(
        self,
        X: pd.DataFrame,
        variables: list,
        values_space: bool,
        rules_list: list,
        score_list: list,
        rules_updated: callable,
    ):
        """ 
        rules : initial lost of Rule
        rules_updated : callback for the GUI
        """

        self.X = X
        self.variables = variables
        self.is_value_space = values_space
        self.rules_updated = rules_updated
        self.vbox_widget = get_widget_at_address(app_widget, "305010" if values_space else "305011")

        self.rules_db = {}
        self.rules_db[0] = [rules_list, score_list]
        self.current_index = 0

        if rules_list is not None:  # We're not an empty RsW
            self._update_card()
            self._set_rule_widgets(rules_list)

            self.rules_updated(
                self,
                Rule.rules_to_indexes(self.get_current_rule_list(), self.X, self.variables),
            )

    def dump_db(self) -> str:
        """
        For debug purposes
        """
        txt = ""
        for i in range(len(self.rules_db)):
            rules_list = self.rules_db[i][0]
            scores_list = self.rules_db[i][1]
            txt = txt + f"({i}) : {len(rules_list)} rules:\n"
            for rule in rules_list:
                txt = txt + f"    {rule}\n"
            txt = txt + f"   scores = {scores_list}\n"

        return txt

    def get_current_rule_list(self):
        return self.rules_db[self.current_index][0]

    def get_current_score_list(self):
        return self.rules_db[self.current_index][1]

    def _set_rule_widgets(self, rules_list):
        """
        Ensure our vbox_wigdet has one RuleWidget per rule below the v.Card
        """
        rule_widget_list = []
        for rule in rules_list:
            rule_widget_list.append(RuleWidget(rule, self.X, self.is_value_space))

        # get_widget_at_address(self.vbox_widget, "1") = v.ExpansionPanels where we plug RuleWidgets.expansion_panel_widget
        get_widget_at_address(self.vbox_widget, "1").children = [
            rw.expansion_panel_widget for rw in rule_widget_list
        ]

    def _update_card(self):
        # We set the scores
        if (
            self.get_current_score_list() is None
            or len(self.get_current_score_list()) == 0
        ):
            scores_txt = f"Precision : -1, recall : -1, f1_score : -1"
        else:
            scores_txt = f"Precision : {self.get_current_score_list()['precision']}, recall : {self.get_current_score_list()['recall']}, f1_score : {self.get_current_score_list()['f1']}"
        change_widget(self.vbox_widget, "0100", scores_txt)

        # We set the rules expressions in the DataTable
        get_widget_at_address(self.vbox_widget, "011").items = Rule.rules_to_records(self.get_current_rule_list())

    def hide_beeswarm(self, hide: bool):
        # We retrieve the beeswarmGrp (VBox)
        get_widget_at_address(self.root_widget, "0101").disabled = hide

    def skope_slider_changed(*change):
        # we just call skope_changed @GUI
        self.skope_changed()

    def redraw_both_graphs(self):
        # We update the refiner's histogram :
        with get_widget_at_address(self.root_widget, "01001").batch_update():
            get_widget_at_address(self.vbox_widget, "01001").data[
                0
            ].x = self._ds.get_full_values()[
                self._selection.get_vs_rules()[self._variable.get_col_index][2]
            ]

            # We update the refiner's beeswarm :
            # get_widget_at_address(self.root_widget,"01011").v_model : # TODO Why do we check ?
            with get_widget_at_address(self.vbox_widget, "01011").batch_update():
                pass
                

    def skope_rule_changed(widget, event, data):
        pass
        

    def get_class_selector(
        self, min: int = 1, max: int = -1, fig_size: int = 700
    ) -> v.Layout:
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
        row = v.Row(class_="ml-6 ma-3", children=widgetList)
        text = v.Html(
            tag="h3",
            children=["Select the values of the feature " + self._variable.getSymbol()],
        )
        return v.Layout(
            class_="d-flex flex-column align-center justify-center",
            style_="width: " + str(int(fig_size) - 70) + "px; height: 303px",
            children=[v.Spacer(), text, row],
        )

    def real_time_changed(*args):
        """If changed, we invert the validate button"""
        get_widget_at_address(
            self.root_widget, "0010020"
        ).disabled = not get_widget_at_address(self.root_widget, "0010020").disabled

        # See realTimeUpdateCheck (0010021)
        get_widget_at_address(self.root_widget, "0010021").on_event(
            "change", real_time_changed
        )

    def beeswarm_color_changed(*args):
        """If changed, we invert the showScake value"""
        # See beeswarm :
        show_scale = (
            get_widget_at_address(self.root_widget, "01011").data[0].marker[showscale]
        )
        show_scale = get_widget_at_address(self.root_widget, "01011").update_traces(
            marker=dict(showscale=not show_scale)
        )

        # See bsColorChoice[,v.Switch] (0010101)
        self._widgetGraph.get_widget_at_address("010101").on_event(
            "change", beeswarm_color_changed
        )

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
        if widget.v_model:
            # TODO : understand
            # We define accordion (0010) children as histoCtrl (00100) + list (accordion(0010).children[1])
            self._widget.get_widget_at_address("010").children = [
                self._widget.get_widget_at_address("0100")
            ] + list(self._widget.get_widget_at_address("010").children[1:])
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
            self.selection.getVSRules()[index][0] = get_widget_at_address(
                self.root_widget, "010001"
            ).v_model[0]
            self.selection.getVSRules()[index][4] = get_widget_at_address(
                self.root_widget, "010001"
            ).v_model[1]

            self._skope_list = create_rule_card(self.selection.ruleListToStr())
        else:
            class_selector = self.get_class_selector()
            get_widget_at_address(self.root_widget, "010").children = [
                class_selector
            ] + list(get_widget_at_address(self.root_widget, "010").children[1:])
            aSet = []
            for i in range(len(self.get_class_selector().children[2].children)):
                if class_selector.children[2].children[i].v_model:
                    aSet.append(int(class_selector.children[2].children[i].label))
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
            self._skope_list = create_rule_card(
                self._gui.get_selection().ruleListToStr()
            )

        # We wire the "change" event on the isContinuousChck (001021)
        get_widget_at_address(self.root_widget, "01021").on_event(
            "change", continuous_check_changed
        )


def update_skr_infocards(selection_indexes: list, side: int, widget: Widget):
    """Sets a message + indicates the scores of the sub_models
    Do not set the rules themselves
    """

    if selection_indexes.is_empty():
        temp_card_children = [widgets.HTML("Please select points")]
    else:
        if (
            0 not in selection_indexes.getYMaskList()
            or 1 not in selection_indexes.getYMaskList()
        ):
            temp_card_children = [widgets.HTML("You can't choose everything/nothing !")]
        else:
            # If no rule for one of the two, nothing is displayed
            if not selection_indexes.has_rules_defined():
                temp_card_children = [widgets.HTML("No rule found")]
            else:
                if side == config.VS:
                    scores = selection_indexes.getVSScore()
                else:
                    scores = selection_indexes.getESScore()
                temp_text_children = (
                    "p = "
                    + str(scores[0])
                    + "% "
                    + "r = "
                    + str(scores[1])
                    + "% "
                    + " ext. of the tree = "
                    + str(scores[2])
                )

    get_widget_at_address(widget, "30500101").children = temp_card_children
    get_widget_at_address(widget, "30500101").children = temp_text_children

def create_empty_ruleswidget(values_space:bool)-> RulesWidget:
    empty_rsw = RulesWidget(None, None, None, None, None, None)
    empty_rsw.vbox_widget = get_widget_at_address(app_widget, "305010" if values_space else "305011")
    change_widget(empty_rsw.vbox_widget, "0100", "Rule scores will be displayed here")
    change_widget(empty_rsw.vbox_widget, "011", v.Html( # 305 010 010
            class_="ml-3 light-grey", 
            tag="h3", 
            children=[
                "Rule will be described here"
                ]
            ))
    change_widget(empty_rsw.vbox_widget, "1", v.ExpansionPanels(
        children=[
                v.ExpansionPanel(
                    children=[
                        v.ExpansionPanelHeader(
                            class_="font-weight-bold blue lighten-4",
                            children=[
                                "Variable"
                            ]
                        ),
                        v.ExpansionPanelContent(
                            children=[
                                v.Col(
                                    class_="ma-3 pa-3",
                                    children=[
                                        v.Spacer(), 
                                        v.RangeSlider(
                                            class_="ma-3",
                                            v_model=[
                                                -1,
                                                1,
                                            ],
                                            min=-5,
                                            max=5,
                                            step=1,
                                            thumb_label="always",
                                        ),
                                    ],
                                ),
                                v.Html(
                                    class_="ml-3 light-grey", 
                                    tag="h3", 
                                    children=[
                                        
                                        "Histogram will be displayed here" if values_space else "Beeswarm will be displayed here"
                                        ]
                                    ),
                            ]
                        ),
                    ]
                )
            ]
        )
    )
    return empty_rsw


# Static UI elements

dummy_df = pd.DataFrame(
    {
        "Variable": ["Population", "MedInc", "Latitude", "Longitude"],
        "Unit": ["people", "k€", "° N", "° W"],
        "Desc": ["People living in the block", "Median income", "-", "-"],
        "Critical ?": [False, True, False, False],
        "Rule": [
            "Population ≤ 2 309",
            "MedInc ∈ [3.172, 5.031⟧",
            "Latitude ≥ 37.935",
            "Longitude > 0.559",
        ],
    }
)

splash_widget = v.Layout(
    class_="d-flex flex-column align-center justify-center",
    children=[
        widgets.Image(  # 0
            value=widgets.Image._load_file_value(
                files("antakia.assets").joinpath("logo_antakia.png")
            ),
            layout=Layout(width="230px"),
        ),
        v.Row(  # 1
            style_="width:85%;",
            children=[
                v.Col(  # 10
                    children=[
                        v.Html(
                            tag="h3",
                            class_="mt-2 text-right",
                            children=["Computation of explanation values"],
                        )
                    ]
                ),
                v.Col(  # 11
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
                    ],
                ),
                v.Col(  # #12
                    children=[
                        v.TextField(  # 120
                            variant="plain",
                            v_model="",
                            readonly=True,
                            class_="mt-0 pt-0",
                        )
                    ]
                ),
            ],
        ),
        v.Row(  # 2
            style_="width:85%;",
            children=[
                v.Col(  # 20
                    children=[
                        v.Html(
                            tag="h3",
                            class_="mt-2 text-right",
                            children=["Computation of dimension reduction values"],
                        )
                    ]
                ),
                v.Col(  # 21
                    class_="mt-3",
                    children=[
                        v.ProgressLinear(  # 210
                            style_="width: 80%",
                            class_="py-0 mx-5",
                            v_model=0,
                            color="primary",
                            height="15",
                            striped=True,
                        )
                    ],
                ),
                v.Col(  # 22
                    children=[
                        v.TextField(  # 220
                            variant="plain",
                            v_model="",
                            readonly=True,
                            class_="mt-0 pt-0",
                        )
                    ]
                ),
            ],
        ),
    ],
)



app_widget = widgets.VBox(
    [
        v.AppBar(  # 0
            elevation="4",
            class_="ma-4",
            rounded=True,
            children=[
                v.Layout(
                    children=[
                        widgets.Image(  # 010
                            value=open(
                                files("antakia.assets").joinpath("logo_ai-vidence.png"),
                                "rb",
                            ).read(),
                            height=str(864 / 20) + "px",
                            width=str(3839 / 20) + "px",
                        )
                    ],
                    class_="mt-1",
                ),
                v.Html( # 01
                    tag="h2", 
                    children=["AntakIA"], # 010
                    class_="ml-3"),  
                v.Spacer(),  # 02
                v.Btn(  # backupBtn # 03
                    icon=True,
                    children=[v.Icon(children=["mdi-content-save"])],
                    elevation=0,
                    disabled=True,
                ),
                v.Menu(  # 04
                    v_slots=[
                        {
                            "name": "activator",
                            "variable": "props",
                            "children": v.Btn(
                                v_on="props.on",
                                icon=True,
                                size="x-large",
                                children=[v.Icon(children=["mdi-tune"])],
                                class_="ma-2 pa-3",
                                elevation="0",
                            ),
                        }
                    ],
                    children=[
                        v.Card(  # 040
                            class_="pa-4",
                            rounded=True,
                            children=[
                                widgets.VBox(  # 040 0
                                    [
                                        v.Slider(  # 040 0O
                                            v_model=400,
                                            min=100,
                                            max=3000,
                                            step=10,
                                            label="Figure width",
                                        )
                                    ]
                                )
                            ],
                            min_width="500",
                        )
                    ],
                    v_model=False,
                    close_on_content_click=False,
                    offset_y=True,
                ),
                v.Btn(  # gotoWebBtn # 06
                    icon=True, children=[v.Icon(children=["mdi-web"])], elevation=0
                ),
            ],
        ),
        widgets.HBox(  # 1
            [
                v.Row(  # 10
                    class_="ma-3",
                    children=[
                        v.Icon(children=["mdi-numeric-2-box"]),
                        v.Icon(children=["mdi-alpha-d-box"]),
                        v.Switch(  # Dim switch # 102
                            class_="ml-3 mr-2",
                            v_model=False,
                            label="",
                        ),
                        v.Icon(children=["mdi-numeric-3-box"]),
                        v.Icon(children=["mdi-alpha-d-box"]),
                    ],
                ),
                v.Layout(  # 11
                    class_="pa-2 ma-2",
                    elevation="3",
                    children=[
                        v.BtnToggle(  # colorChoiceBtnToggle # 110
                            mandatory=True,
                            v_model="Y",
                            children=[
                                v.Btn(  # 1100
                                    icon=True,
                                    children=[
                                        v.Icon(children=["mdi-alpha-y-circle-outline"])
                                    ],
                                    value="y",
                                    v_model=True,
                                ),
                                v.Btn(  # 1101
                                    icon=True,
                                    children=[v.Icon(children=["mdi-alpha-y-circle"])],
                                    value="y^",
                                    v_model=True,
                                ),
                                v.Btn(  # 1102
                                    icon=True,
                                    children=[v.Icon(children=["mdi-delta"])],
                                    value="residual",
                                ),
                            ],
                        ),
                        v.Select(  # explanationSelect # 111
                            label="Explanation method",
                            items=[
                                {"text": "SHAP (imported)", "disabled": True},
                                {"text": "SHAP (computed)", "disabled": True},
                                {"text": "LIME (imported)", "disabled": True},
                                {"text": "LIME (computed)", "disabled": True},
                            ],
                            class_="ma-2 mt-1 ml-6",
                            style_="width: 150px",
                            disabled=False,
                        ),
                        v.Btn(  # computeMenuBtnBtn # 112
                            icon=True,
                            children=[
                                v.Icon(  # 112 0
                                    children=[
                                        "mdi-opacity"  # 112 00
                                        ])
                                ],
                            class_="ma-2 ml-6 pa-3",
                            elevation="3",
                        ),
                    ],
                ),
                v.Layout(  # 12
                    class_="mt-3",
                    children=[
                        widgets.HBox(  # 120
                            [
                                v.Select(  # projSelectVS # 1200
                                    label="Projection in the VS :",
                                    items=DimReducMethod.dimreduc_methods_as_str_list(),
                                    style_="width: 150px",
                                ),
                                v.Layout(  # 120 1
                                    children=[
                                        v.Menu(  # projSettingsMenuVS # 120 10
                                            v_slots=[
                                                {
                                                    "name": "activator",
                                                    "variable": "props",
                                                    "children": v.Btn(
                                                        v_on="props.on",
                                                        icon=True,
                                                        size="x-large",
                                                        children=[
                                                            v.Icon(
                                                                children=["mdi-cogs"],
                                                                size="large",
                                                            )
                                                        ],
                                                        class_="ma-2 pa-3",
                                                        elevation="3",
                                                    ),
                                                }
                                            ],
                                            children=[
                                                v.Card(  # 120 100
                                                    class_="pa-4",
                                                    rounded=True,
                                                    children=[
                                                        widgets.VBox(
                                                            [  # ProjVS sliders # 120 100 0
                                                                v.Slider(
                                                                    v_model=10,
                                                                    min=5,
                                                                    max=30,
                                                                    step=1,
                                                                    label="Number of neighbours",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=0.5,
                                                                    min=0.1,
                                                                    max=0.9,
                                                                    step=0.1,
                                                                    label="MN ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=2,
                                                                    min=0.1,
                                                                    max=5,
                                                                    step=0.1,
                                                                    label="FP ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
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
                                widgets.HBox(  # 120 2
                                    [
                                        v.ProgressCircular(  # 120 20
                                            indeterminate=True,
                                            color="blue",
                                            width="6",
                                            size="35",
                                            class_="mx-4 my-3",
                                        )
                                    ]
                                ),
                            ]
                        ),
                        widgets.HBox(  # 121
                            [
                                v.Select(  # projSelectES # 121 0
                                    label="Projection in the ES :",
                                    items=DimReducMethod.dimreduc_methods_as_str_list(),
                                    style_="width: 150px",
                                ),
                                v.Layout(  # 121 1
                                    children=[
                                        v.Menu(  # projSettingsMenuES # 121 10
                                            v_slots=[
                                                {
                                                    "name": "activator",
                                                    "variable": "props",
                                                    "children": v.Btn(
                                                        v_on="props.on",
                                                        icon=True,
                                                        size="x-large",
                                                        children=[
                                                            v.Icon(
                                                                children=["mdi-cogs"],
                                                                size="large",
                                                            )
                                                        ],
                                                        class_="ma-2 pa-3",
                                                        elevation="3",
                                                    ),
                                                }
                                            ],
                                            children=[
                                                v.Card(  # 121 100
                                                    class_="pa-4",
                                                    rounded=True,
                                                    children=[
                                                        widgets.VBox(
                                                            [  # ProjES sliders # 121 100 0
                                                                v.Slider(
                                                                    v_model=10,
                                                                    min=5,
                                                                    max=30,
                                                                    step=1,
                                                                    label="Number of neighbours",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=0.5,
                                                                    min=0.1,
                                                                    max=0.9,
                                                                    step=0.1,
                                                                    label="MN ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
                                                                v.Slider(
                                                                    v_model=2,
                                                                    min=0.1,
                                                                    max=5,
                                                                    step=0.1,
                                                                    label="FP ratio",
                                                                ),
                                                                v.Html(
                                                                    class_="ml-3",
                                                                    tag="h3",
                                                                    children=["#"],
                                                                ),
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
                                widgets.HBox(  # 121 2
                                    [
                                        v.ProgressCircular(  # ESBusyBox # 121 20
                                            indeterminate=True,
                                            color="blue",
                                            width="6",
                                            size="35",
                                            class_="mx-4 my-3",
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                ),
            ],
            layout=Layout(
                width="100%",
                display="flex",
                flex_flow="row",
                justify_content="space-around",
            ),
        ),
        widgets.VBox(  # 2
            [
                widgets.HBox(  # 20
                    [
                        widgets.VBox(  # 200
                            [
                                widgets.HTML("<h3>Values Space<h3>"),  # 2000
                                v.Container(  # 2001 # placeholder for vs_hde.figure_widget
                                    children=[
                                        FigureWidget(
                                            [Scatter(mode="markers")]
                                        ),  # 20010
                                    ]
                                ),
                            ],
                            layout=Layout(
                                display="flex",
                                align_items="center",
                                margin="0px 0px 0px 0px",
                            ),
                        ),
                        widgets.VBox(  #  #201
                            [
                                widgets.HTML("<h3>Explanations Space<h3>"),  # 201 0
                                v.Container(  # 201 1
                                    children=[
                                        FigureWidget(
                                            [Scatter(mode="markers")]
                                        ),  # 201 10
                                    ]
                                ),
                            ],
                            layout=Layout(
                                display="flex",
                                align_items="center",
                                margin="0px 0px 0px 0px",
                            ),
                        ),
                    ],
                    layout=Layout(width="100%"),
                )
            ]
        ),
        v.Container(  # antakiaMethodCard # 3
            fluid=True,
            children=[
                v.Tabs(  # 30
                    v_model=0,  # default active tab
                    children=[
                        v.Tab(children=["1. Selection"]),  # 300
                        v.Tab(children=["2. Refinement"]),  # 301
                        v.Tab(children=["3. Sub-model"]),  # 302
                        v.Tab(children=["4. Regions"]),  # 303
                    ]
                    + [
                        v.TabItem(  # Tab 1) = tabOneSelectionColumn ? Selection # 304
                            children=[
                                v.Card(  # selectionCard # 304 0
                                    class_="ma-2",
                                    elevation=0,
                                    children=[
                                        v.Layout(  # 304 00
                                            children=[
                                                v.Icon( # 304 000
                                                    children=["mdi-lasso"]
                                                ),  # 304 000
                                                v.Html(  # 304 001
                                                    class_="mt-2 ml-4",
                                                    tag="h4",
                                                    children=[
                                                        "0 point selected : use the lasso tool on the figures above or use the auto-selection tool below"  # 304 001 0
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                                v.ExpansionPanels(  # out_accordion # 304 1
                                    class_="ma-2",
                                    children=[
                                        v.ExpansionPanel(  # 304 10
                                            children=[
                                                v.ExpansionPanelHeader(  # 304 100
                                                    children=["Data selected"]
                                                ),  # 304 100 0
                                                v.ExpansionPanelContent(  # 304 101
                                                    children=[
                                                        v.Alert(  # out_selec_all # 304 101 0
                                                            max_height="400px",
                                                            style_="overflow: auto",
                                                            elevation="0",
                                                            children=[
                                                                v.Row(  # 304 101 00
                                                                    class_="d-flex flex-row justify-space-between",
                                                                    children=[
                                                                        v.Layout(  # out_selec # 304 101 000
                                                                            style_="min-width: 100%; max-width: 94%",
                                                                            children=[
                                                                                v.DataTable(  # 304 410
                                                                                    v_model=[],
                                                                                    show_select=False,
                                                                                    headers=[
                                                                                        {
                                                                                            "text": column,
                                                                                            "sortable": True,
                                                                                            "value": column,
                                                                                        }
                                                                                        for column in dummy_df.columns
                                                                                    ],
                                                                                    items=dummy_df.to_dict(
                                                                                        "records"
                                                                                    ),
                                                                                    hide_default_footer=False,
                                                                                    disable_sort=False,
                                                                                )
                                                                            ],
                                                                        )
                                                                    ],
                                                                ),
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        )
                                    ],
                                ),
                                v.Layout(  # clusterGrp # 304 2
                                    class_="d-flex flex-row",
                                    children=[
                                        v.Btn(  # findClusterBtn # 304 20
                                            class_="ma-1 mt-2 mb-0",
                                            elevation="2",
                                            children=[
                                                v.Icon(children=["mdi-magnify"]),
                                                "Find clusters",
                                            ],
                                        ),
                                        v.Checkbox(  # clusterCheck # 304 21
                                            v_model=True,
                                            label="Optimal number of clusters :",
                                            class_="ma-3",
                                        ),
                                        v.Slider(  # clustersSlider # 304 22
                                            style_="width : 30%",
                                            class_="ma-3 mb-0",
                                            min=2,
                                            max=20,
                                            step=1,
                                            v_model=3,
                                            disabled=True,
                                        ),
                                        v.Html(  # clustersSliderTxt # 304 23
                                            tag="h3",
                                            class_="ma-3 mb-0",
                                            children=["Number of clusters #"],
                                        ),
                                    ],
                                ),
                                v.ProgressLinear(  # loadingClustersProgLinear # 304 3
                                    indeterminate=True,
                                    class_="ma-3",
                                    style_="width : 100%",
                                ),
                                v.Row(  # clusterResults # 304 4
                                    children=[
                                        v.Layout(
                                            class_="flex-grow-0 flex-shrink-0",
                                            children=[
                                                v.Btn(
                                                    class_="d-none",
                                                    elevation=0,
                                                    disabled=True,
                                                )
                                            ],  # 304 40
                                        ),
                                        v.Layout(  # 304 41
                                            class_="flex-grow-1 flex-shrink-0",
                                            children=[
                                                v.DataTable(  # 304 410
                                                    v_model=[],
                                                    show_select=False,
                                                    headers=[
                                                        {
                                                            "text": column,
                                                            "sortable": True,
                                                            "value": column,
                                                        }
                                                        for column in dummy_df.columns
                                                    ],
                                                    items=dummy_df.to_dict("records"),
                                                    hide_default_footer=False,
                                                    disable_sort=False,
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                                v.Layout(  # magicGUI 304 5
                                    class_="d-flex flex-row justify-center align-center",
                                    children=[
                                        v.Spacer(),  # 304 50
                                        v.Btn(  # magicBtn # findClusterBtn # 304 51
                                            class_="ma-3",
                                            children=[
                                                v.Icon(
                                                    children=["mdi-creation"],
                                                    class_="mr-3",
                                                ),
                                                "Magic button",
                                            ],
                                        ),
                                        v.Checkbox(  # # magicCheckBox 304 52
                                            v_model=True,
                                            label="Demonstration mode",
                                            class_="ma-4",
                                        ),
                                        v.TextField(  # 304 53
                                            class_="shrink", 
                                            type="number",
                                            label="Time between the steps (ds)",
                                            v_model=10,
                                        ),
                                        v.Spacer(),  # 304 54
                                    ],
                                ),
                            ]
                        ),
                        v.TabItem(  # Tab 2) = tabTwoSkopeRulesColumn ? Refinement # 305
                            children=[
                                v.Col(  # 305 0
                                    children=[
                                        widgets.VBox(  # skopeBtnsGrp # 305 00
                                            [
                                                v.Layout(  # skopeBtns # 305 000
                                                    class_="d-flex flex-row",
                                                    children=[
                                                        v.Btn(  # validateSkopeBtn # 305 000 0
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(
                                                                    class_="mr-2",
                                                                    children=[
                                                                        "mdi-auto-fix"
                                                                    ],
                                                                ),
                                                                "Skope-Rules",
                                                            ],
                                                        ),
                                                        v.Btn(  # reinitSkopeBtn # 305 000 1
                                                            class_="ma-1",
                                                            children=[
                                                                v.Icon(
                                                                    class_="mr-2",
                                                                    children=[
                                                                        "mdi-skip-backward"
                                                                    ],
                                                                ),
                                                                "Come back to the initial rules",
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        widgets.HBox(  # 305 01
                                            [
                                                widgets.VBox(  # placeholder for the VS RulesWidget (RsW) # 305 010
                                                    [
                                                        v.Col( # placeholder for the VS RulesWidget card # 305 010 0
                                                            children=[
                                                                v.Row( # 305 010 00
                                                                    class_="ml-4",
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), # 305 010 000
                                                                        v.Html(class_="ml-3", tag="h2", children=["Rule(s) applied on the value space"]), # 305 010 001
                                                                    ]
                                                                    ),
                                                                v.Col( # 305 010 01
                                                                    elevation=10,
                                                                    children=[ 
                                                                        v.Html( # 305 010 010
                                                                            class_="ml-3", 
                                                                            tag="p", 
                                                                            children=[
                                                                                "Precision = 0.3, Recall = 0.8, F1 = 22" # 305 010 010 0
                                                                                ]
                                                                            ), 
                                                                            v.DataTable( # 305 010 011
                                                                                    v_model=[],
                                                                                    show_select=False,
                                                                                    headers=[{"text": column, "sortable": False, "value": column } for column in dummy_df.columns],
                                                                                    items=dummy_df.to_dict("records"),
                                                                                    hide_default_footer=True,
                                                                                    disable_sort=True,
                                                                                ),
                                                                        ]
                                                                    )
                                                                ]
                                                        ),
                                                        v.ExpansionPanels( # placeholder for the VS RuleWidgets # 305 010 1
                                                            children=[
                                                                v.ExpansionPanel( # 305 010 10
                                                                    children=[
                                                                        v.ExpansionPanelHeader( # 305 010 100
                                                                            class_="font-weight-bold blue lighten-4",
                                                                            children=[
                                                                                "MedInc"
                                                                            ]
                                                                        ),
                                                                        v.ExpansionPanelContent( # 305 010 101
                                                                            children=[
                                                                                v.Col( # 305 010 101 0
                                                                                    class_="ma-3 pa-3",
                                                                                    children=[
                                                                                        v.Spacer(), # 305 010 101 00
                                                                                        v.RangeSlider(  # 305 010 101 01
                                                                                            class_="ma-3",
                                                                                            v_model=[
                                                                                                -1,
                                                                                                1,
                                                                                            ],
                                                                                            min=-5,
                                                                                            max=5,
                                                                                            step=0.1,
                                                                                            thumb_label="always",
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                                FigureWidget( # 305 010 101 1
                                                                                    data=[
                                                                                        Histogram(
                                                                                            x=pd.DataFrame(
                                                                                                np.random.randint(
                                                                                                    0,
                                                                                                    100,
                                                                                                    size=(
                                                                                                        100,
                                                                                                        4,
                                                                                                    ),
                                                                                                ),
                                                                                                columns=list(
                                                                                                    "ABCD"
                                                                                                ),
                                                                                            ),
                                                                                            bingroup=1,
                                                                                            nbinsx=50,
                                                                                            marker_color="grey",
                                                                                        )
                                                                                    ]
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                widgets.VBox(  # placeholder for the ES RulesWidget (RsW) # 305 011
                                                    [
                                                        v.Col( # placeholder for the ES RulesWidget card
                                                            children=[
                                                                v.Row(
                                                                    class_="ml-4",
                                                                    children=[
                                                                        v.Icon(children=["mdi-target"]), 
                                                                        v.Html(class_="ml-3", tag="h2", children=["Rule(s) applied on the explanations space"]),
                                                                    ]
                                                                    ),
                                                                v.Col( 
                                                                    elevation=10,
                                                                    children=[ 
                                                                        v.Html(class_="ml-3", tag="p", children=["Precision = 0.3, Recall = 0.8, F1 = 22"]),
                                                                        v.DataTable( 
                                                                                    v_model=[],
                                                                                    show_select=False,
                                                                                    headers=[{"text": column, "sortable": False, "value": column } for column in dummy_df.columns],
                                                                                    items=dummy_df.to_dict("records"),
                                                                                    hide_default_footer=True,
                                                                                    disable_sort=True,
                                                                                ),
                                                                        ]
                                                                    )
                                                                ]
                                                        ),
                                                        v.ExpansionPanels( # placeholder for the ES RuleWidgets # 305 011 1
                                                            elevation=4,
                                                            children=[
                                                                v.ExpansionPanel(
                                                                    children=[
                                                                        v.ExpansionPanelHeader(
                                                                            class_="font-weight-bold blue lighten-4",
                                                                            variant="outlined",
                                                                            children=[
                                                                                "MedInc"
                                                                            ]
                                                                        ),
                                                                        v.ExpansionPanelContent(
                                                                            children=[
                                                                                v.Col(
                                                                                    class_="ma-3 pa-3",
                                                                                    children=[
                                                                                        v.Spacer(),
                                                                                        v.RangeSlider(  # skopeSlider
                                                                                            class_="ma-3",
                                                                                            v_model=[
                                                                                                -1,
                                                                                                1,
                                                                                            ],
                                                                                            min=-5,
                                                                                            max=5,
                                                                                            step=0.1,
                                                                                            thumb_label="always",
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                                FigureWidget(
                                                                                    data=[
                                                                                        Histogram(
                                                                                            x=pd.DataFrame(
                                                                                                np.random.randint(
                                                                                                    0,
                                                                                                    100,
                                                                                                    size=(
                                                                                                        100,
                                                                                                        4,
                                                                                                    ),
                                                                                                ),
                                                                                                columns=list(
                                                                                                    "ABCD"
                                                                                                ),
                                                                                            ),
                                                                                            bingroup=1,
                                                                                            nbinsx=50,
                                                                                            marker_color="grey",
                                                                                        )
                                                                                    ]
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ]
                                                                )
                                                            ]
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                        v.Row(  # addButtonsGrp # 305 02
                                            children=[
                                                v.Btn(  # addSkopeBtn # 305 020
                                                    class_="ma-4 pa-2 mb-1",
                                                    children=[
                                                        v.Icon(children=["mdi-plus"]),
                                                        "Add a rule",
                                                    ],
                                                ),
                                                v.Select(  # addAnotherFeatureWgt # 305 021
                                                    class_="mr-3 mb-0",
                                                    explanationsMenuDict=["/"],
                                                    v_model="/",
                                                    style_="max-width : 15%",
                                                ),
                                                v.Spacer(),  # 305 022
                                                v.Btn(  # addMapBtn # 305 023
                                                    class_="ma-4 pa-2 mb-1",
                                                    children=[
                                                        v.Icon(
                                                            class_="mr-4",
                                                            children=["mdi-map"],
                                                        ),
                                                        "Display the map",
                                                    ],
                                                    color="white",
                                                    disabled=True,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),
                        v.TabItem(  # Tab 3) = tabThreeSubstitutionVBox ? # 306
                            children=[
                                widgets.VBox(  # 306 0
                                    [
                                        v.ProgressLinear(  # loadingModelsProgLinear # 306 00
                                            indeterminate=True,
                                            class_="my-0 mx-15",
                                            style_="width: 100%;",
                                            color="primary",
                                            height="5",
                                        ),
                                        v.SlideGroup(  # subModelslides # 306 01
                                            v_model=None,
                                            class_="ma-3 pa-3",
                                            elevation=4,
                                            center_active=True,
                                            show_arrows=True,
                                            children=[
                                                v.SlideItem(  # 306 010 # dummy SlideItem. Will be replaced by the app
                                                    # style_="width: 30%",
                                                    children=[
                                                        v.Card(
                                                            class_="grow ma-2",
                                                            children=[
                                                                v.Row(
                                                                    class_="ml-5 mr-4",
                                                                    children=[
                                                                        v.Icon(
                                                                            children=[
                                                                                "a name"
                                                                            ]
                                                                        ),
                                                                        v.CardTitle(
                                                                            children=[
                                                                                "model foo"
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                                v.CardText(
                                                                    class_="mt-0 pt-0",
                                                                    children=[
                                                                        "Model's score"
                                                                    ],
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
                        v.TabItem(  # Tab 4) = tabFourRegionListVBox # 307
                            children=[
                                v.Col(  # 307 0
                                    children=[
                                        widgets.VBox(  # 307 00
                                            [
                                                v.Btn(  # 307 000
                                                    class_="ma-4 pa-2 mb-1",
                                                    children=[
                                                        v.Icon(
                                                            class_="mr-4",
                                                            children=["mdi-map"],
                                                        ),
                                                        "Validate the region",
                                                    ],
                                                    color="white",
                                                    disabled=True,
                                                )
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ],
                )
            ],
            class_="mt-0",
            outlined=True,
        ),
    ]
)
