import pandas as pd

from plotly.graph_objects import FigureWidget, Scatter, Scatter3d
import ipyvuetify as v
from ipywidgets.widgets import Widget
from ipywidgets import Layout, widgets

from antakia.data import ProjectedValues, DimReducMethod, ExplanationMethod
import antakia.compute as compute
from antakia.rules import Rule
from antakia.gui.widgets import get_widget

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
            get_widget(self._compute_menu, "000203").on_event(
                "click", self.compute_btn_clicked
            )
            # LIME compute button :
            get_widget(self._compute_menu, "000303").on_event(
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
        self._display_rules_one_side(df_ids_list, 2)
        self._display_rules_one_side(df_ids_list, 3)

    def _display_rules_one_side(self, df_ids_list: list, dim: int):

        fig = self.figure_2D if dim == 2 else self.figure_3D

        # We add a second trace (Scatter) to the figure to display the rules
        if df_ids_list is None or len(df_ids_list) == 0:
            # We remove the 'rule trace' if exists
            if len(fig.data) > 1:
                if fig.data[1] is not None:
                    # It seems impossible to remove a trace from a figure once created
                    # So we hide or update&show this 'rule_trace'
                    fig.data[1].visible = False
        else:
            # Let's create a color scale with rule postives in blue and the rest in grey
            colors = ["grey"] * self.pv_list[self.current_pv].get_length()
            # IMPORTANT : we convert df_ids to row_ids (dataframe may not be indexed by row !!)
            row_ids_list = Rule.indexes_to_rows(
                self.pv_list[self.current_pv].X, df_ids_list
            )
            for row_id in row_ids_list:
                colors[row_id] = "blue"

            if len(fig.data) == 1:
                # We need to add a 'rule_trace'
                if dim == 2:
                    fig.add_trace(
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
                else:
                    fig.add_trace(
                        Scatter3d(
                            x=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 3
                            )[0],
                            y=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 3
                            )[1],
                            z=self.pv_list[self.current_pv].get_proj_values(
                                self._get_projection_method(), 3
                            )[2],
                            mode="markers",
                            marker=dict(
                                color=colors,
                            ),
                        )
                    )
            elif len(fig.data) == 2:
                # We replace the existing 'rule_trace'
                with fig.batch_update():
                    fig.data[1].x = self.pv_list[
                        self.current_pv
                    ].get_proj_values(self._get_projection_method(), dim)[0]
                    fig.data[1].y = self.pv_list[
                        self.current_pv
                    ].get_proj_values(self._get_projection_method(), dim)[1]
                    if dim == 3:
                        fig.data[1].z = self.pv_list[
                            self.current_pv
                        ].get_proj_values(self._get_projection_method(), dim)[2]
                    fig.layout.width = self.fig_size
                    fig.data[1].marker.color = colors

                fig.data[1].visible = True  # in case it was hidden
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
            self.figure_2D.data[0].on_selection(self._selection_event)
            self.figure_2D.data[0].on_deselect(self._deselection_event)
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
            progress_linear = get_widget(self._compute_menu, "000201")
        else:
            progress_linear = get_widget(self._compute_menu, "000301")

        progress_linear.v_model = progress

        if progress == 100:
            tab = None
            if method == ExplanationMethod.SHAP:
                tab = get_widget(self._compute_menu, "0000")
            else:
                tab = get_widget(self._compute_menu, "0001")
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
