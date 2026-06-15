"""
Tab de visualisation et d'exploration des Tesselles.

Ce tab permet de :
1. Visualiser l'arbre hiérarchique des Tesselles
2. Explorer les caractéristiques de chaque Tesselle
3. Générer et exporter des rapports
4. Interagir avec les modèles locaux

Intégration avec ipyvuetify et plotly.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import ipyvuetify as v
    import ipywidgets as widgets
    from IPython.display import HTML, display

    HAS_IPYVUETIFY = True
except ImportError:
    HAS_IPYVUETIFY = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class TessellationTabConfig:
    """Configuration du tab Tessellation."""

    show_tree: bool = True
    show_metrics: bool = True
    show_local_model: bool = True
    show_rules: bool = True
    show_counterfactuals: bool = True
    export_formats: List[str] = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["json", "markdown", "html"]


class TessellationTab:
    """
    Tab de visualisation des Tesselles dans l'UI AntakIA.

    Attributes
    ----------
    tessellation_result : TessellationResult
        Résultat de la tessellation à visualiser
    config : TessellationTabConfig
        Configuration du tab
    selected_tesselle_id : str
        ID de la Tesselle actuellement sélectionnée

    Examples
    --------
    >>> tab = TessellationTab(result, feature_names)
    >>> tab.display()
    """

    def __init__(
        self,
        tessellation_result: Any = None,
        feature_names: Optional[List[str]] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        shap_values: Optional[np.ndarray] = None,
        model: Optional[Callable] = None,
        config: Optional[TessellationTabConfig] = None,
        on_selection_change: Optional[Callable] = None,
    ):
        """
        Initialise le tab Tessellation.

        Parameters
        ----------
        tessellation_result : TessellationResult
            Résultat de la tessellation
        feature_names : List[str], optional
            Noms des features
        X : np.ndarray, optional
            Données originales
        y : np.ndarray, optional
            Cible (prédictions)
        shap_values : np.ndarray, optional
            Valeurs SHAP
        model : Callable, optional
            Fonction de prédiction du modèle
        config : TessellationTabConfig, optional
            Configuration
        on_selection_change : Callable, optional
            Callback quand la sélection change
        """
        self.result = tessellation_result
        self.feature_names = feature_names
        self.X = X
        self.y = y
        self.shap_values = shap_values
        self.model = model
        self.config = config or TessellationTabConfig()
        self.on_selection_change = on_selection_change

        self.selected_tesselle_id = None

        if HAS_IPYVUETIFY:
            self._build_widget()

    def _build_widget(self):
        """Construit le widget principal."""
        # Header
        self.header = v.Row(
            class_="ma-2",
            children=[
                v.Html(tag="h2", children=["Tessellation Explorer"]),
                v.Spacer(),
                self._build_export_menu(),
            ],
        )

        # Summary card
        self.summary_card = self._build_summary_card()

        # Tree view
        self.tree_view = self._build_tree_view()

        # Detail panel
        self.detail_panel = self._build_detail_panel()

        # Main layout
        self.widget = v.Container(
            fluid=True,
            children=[
                self.header,
                v.Divider(class_="my-2"),
                self.summary_card,
                v.Row(
                    children=[
                        v.Col(
                            cols=4,
                            children=[
                                v.Card(
                                    outlined=True,
                                    children=[
                                        v.CardTitle(children=["Hiérarchie des Tesselles"]),
                                        self.tree_view,
                                    ],
                                )
                            ],
                        ),
                        v.Col(cols=8, children=[self.detail_panel]),
                    ]
                ),
            ],
        )

    def _build_summary_card(self) -> v.Card:
        """Construit la carte de résumé."""
        if self.result is None:
            return v.Card(
                outlined=True,
                class_="ma-2 pa-3",
                children=[v.Html(tag="p", children=["Aucune tessellation disponible"])],
            )

        final = self.result.tesselles.final_tesselles

        return v.Card(
            outlined=True,
            class_="ma-2",
            children=[
                v.CardTitle(children=[f"Résumé: {len(final)} Tesselles"]),
                v.CardText(
                    children=[
                        v.Row(
                            children=[
                                v.Col(
                                    children=[
                                        v.Html(tag="strong", children=["Couverture"]),
                                        v.Html(tag="p", children=[f"{self.result.coverage:.1%}"]),
                                    ]
                                ),
                                v.Col(
                                    children=[
                                        v.Html(tag="strong", children=["Pureté moyenne"]),
                                        v.Html(
                                            tag="p", children=[f"{self.result.mean_purity:.2f}"]
                                        ),
                                    ]
                                ),
                                v.Col(
                                    children=[
                                        v.Html(tag="strong", children=["R² moyen"]),
                                        v.Html(tag="p", children=[f"{self.result.mean_r2:.2f}"]),
                                    ]
                                ),
                                v.Col(
                                    children=[
                                        v.Html(tag="strong", children=["Itérations"]),
                                        v.Html(tag="p", children=[f"{self.result.iterations}"]),
                                    ]
                                ),
                            ]
                        )
                    ]
                ),
            ],
        )

    def _build_tree_view(self) -> v.Treeview:
        """Construit la vue arbre des Tesselles."""
        if self.result is None:
            return v.Treeview(items=[])

        # Construire l'arbre
        items = self._build_tree_items()

        tree = v.Treeview(
            items=items,
            activatable=True,
            open_on_click=True,
            dense=True,
            item_key="id",
            item_text="name",
        )

        tree.on_event("update:active", self._on_tesselle_selected)

        return tree

    def _build_tree_items(self) -> List[Dict]:
        """Construit les items de l'arbre."""
        items = []

        if self.result is None:
            return items

        # Trouver les racines (pas de parent)
        roots = [t for t in self.result.tesselles if t.parent_id is None]

        for root in sorted(roots, key=lambda x: -x.size):
            item = self._tesselle_to_tree_item(root)
            items.append(item)

        return items

    def _tesselle_to_tree_item(self, tesselle) -> Dict:
        """Convertit une Tesselle en item d'arbre."""
        # Icône selon le statut
        icon_map = {
            "pure": "mdi-check-circle",
            "final": "mdi-check-circle",
            "too_small": "mdi-alert-circle",
            "impure": "mdi-arrow-down-bold",
            "pending": "mdi-clock-outline",
        }
        icon = icon_map.get(tesselle.status.value, "mdi-help-circle")

        # Couleur selon la pureté
        if tesselle.purity_score > 0.8:
            color = "green"
        elif tesselle.purity_score > 0.5:
            color = "orange"
        else:
            color = "red"

        item = {
            "id": tesselle.id,
            "name": f"{tesselle.id} ({tesselle.size} pts, R²={tesselle.local_r2:.2f})",
            "icon": icon,
            "color": color,
            "children": [],
        }

        # Ajouter les enfants
        children = self.result.tesselles.get_children(tesselle.id)
        for child in sorted(children, key=lambda x: -x.size):
            item["children"].append(self._tesselle_to_tree_item(child))

        return item

    def _build_detail_panel(self) -> v.Card:
        """Construit le panneau de détail."""
        self.detail_title = v.CardTitle(children=["Sélectionnez une Tesselle"])
        self.detail_content = v.CardText(children=[])

        self.detail_tabs = v.Tabs(
            v_model=0,
            children=[
                v.Tab(children=["Métriques"]),
                v.Tab(children=["Modèle Local"]),
                v.Tab(children=["Règles"]),
                v.Tab(children=["Contrefactuel"]),
            ],
        )

        self.detail_tab_items = v.TabsItems(
            v_model=0,
            children=[
                v.TabItem(children=[self._build_metrics_tab()]),
                v.TabItem(children=[self._build_model_tab()]),
                v.TabItem(children=[self._build_rules_tab()]),
                v.TabItem(children=[self._build_counterfactual_tab()]),
            ],
        )

        # Lier les tabs
        widgets.jslink((self.detail_tabs, "v_model"), (self.detail_tab_items, "v_model"))

        return v.Card(
            outlined=True, children=[self.detail_title, self.detail_tabs, self.detail_tab_items]
        )

    def _build_metrics_tab(self) -> v.Container:
        """Construit l'onglet des métriques."""
        self.metrics_content = v.Container(
            children=[
                v.Html(tag="p", children=["Sélectionnez une Tesselle pour voir ses métriques"])
            ]
        )
        return self.metrics_content

    def _build_model_tab(self) -> v.Container:
        """Construit l'onglet du modèle local."""
        self.model_content = v.Container(
            children=[
                v.Html(tag="p", children=["Sélectionnez une Tesselle pour voir son modèle local"])
            ]
        )
        return self.model_content

    def _build_rules_tab(self) -> v.Container:
        """Construit l'onglet des règles."""
        self.rules_content = v.Container(
            children=[v.Html(tag="p", children=["Sélectionnez une Tesselle pour voir ses règles"])]
        )
        return self.rules_content

    def _build_counterfactual_tab(self) -> v.Container:
        """Construit l'onglet des contrefactuels."""
        self.cf_content = v.Container(
            children=[
                v.Html(tag="p", children=["Sélectionnez une Tesselle pour voir ses contrefactuels"])
            ]
        )
        return self.cf_content

    def _build_export_menu(self) -> v.Menu:
        """Construit le menu d'export."""
        export_items = []

        for fmt in self.config.export_formats:
            item = v.ListItem(children=[v.ListItemTitle(children=[fmt.upper()])], value=fmt)
            export_items.append(item)

        menu = v.Menu(
            offset_y=True,
            v_slots=[
                {
                    "name": "activator",
                    "variable": "menuData",
                    "children": v.Btn(
                        v_on="menuData.on",
                        color="primary",
                        class_="ma-2",
                        children=[v.Icon(left=True, children=["mdi-download"]), "Exporter"],
                    ),
                }
            ],
            children=[v.List(children=export_items)],
        )

        return menu

    def _on_tesselle_selected(self, widget, event, data):
        """Callback quand une Tesselle est sélectionnée."""
        if data and len(data) > 0:
            tesselle_id = data[0]
            self.selected_tesselle_id = tesselle_id
            self._update_detail_panel(tesselle_id)

            if self.on_selection_change:
                tesselle = self.result.tesselles.get(tesselle_id)
                if tesselle:
                    self.on_selection_change(tesselle)

    def _update_detail_panel(self, tesselle_id: str):
        """Met à jour le panneau de détail."""
        tesselle = self.result.tesselles.get(tesselle_id)
        if tesselle is None:
            return

        # Titre
        self.detail_title.children = [f"Tesselle {tesselle_id}"]

        # Métriques
        self._update_metrics_tab(tesselle)

        # Modèle local
        self._update_model_tab(tesselle)

        # Règles
        self._update_rules_tab(tesselle)

        # Contrefactuel
        self._update_counterfactual_tab(tesselle)

    def _update_metrics_tab(self, tesselle):
        """Met à jour l'onglet des métriques."""
        metrics = [
            ("Taille", f"{tesselle.size} points"),
            ("Pureté", f"{tesselle.purity_score:.2%}"),
            ("Score d'interaction", f"{tesselle.interaction_score:.3f}"),
            ("Cohésion VS", f"{tesselle.cohesion_vs:.3f}"),
            ("Cohésion ES", f"{tesselle.cohesion_es:.3f}"),
            ("R² local", f"{tesselle.local_r2:.3f}"),
        ]

        rows = []
        for name, value in metrics:
            rows.append(
                v.Row(
                    children=[
                        v.Col(cols=6, children=[v.Html(tag="strong", children=[name])]),
                        v.Col(cols=6, children=[v.Html(tag="span", children=[value])]),
                    ]
                )
            )

        self.metrics_content.children = rows

    def _update_model_tab(self, tesselle):
        """Met à jour l'onglet du modèle local."""
        content = [
            v.Row(
                children=[
                    v.Col(
                        children=[
                            v.Html(tag="strong", children=["Type de modèle:"]),
                            v.Html(tag="span", class_="ml-2", children=[tesselle.local_model_type]),
                        ]
                    )
                ]
            ),
            v.Row(
                children=[
                    v.Col(
                        children=[
                            v.Html(tag="strong", children=["R²:"]),
                            v.Html(
                                tag="span", class_="ml-2", children=[f"{tesselle.local_r2:.3f}"]
                            ),
                        ]
                    )
                ]
            ),
        ]

        if tesselle.local_coefficients:
            content.append(v.Divider(class_="my-2"))
            content.append(v.Html(tag="strong", children=["Coefficients:"]))

            sorted_coef = sorted(
                tesselle.local_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
            )

            for name, coef in sorted_coef[:10]:
                color = "green" if coef > 0 else "red"
                content.append(
                    v.Row(
                        children=[
                            v.Col(cols=6, children=[v.Html(tag="span", children=[name])]),
                            v.Col(
                                cols=6,
                                children=[
                                    v.Html(
                                        tag="span",
                                        style_=f"color: {color}",
                                        children=[f"{coef:+.3f}"],
                                    )
                                ],
                            ),
                        ]
                    )
                )

        self.model_content.children = content

    def _update_rules_tab(self, tesselle):
        """Met à jour l'onglet des règles."""
        if not tesselle.rules:
            self.rules_content.children = [
                v.Html(tag="p", children=["Aucune règle extraite pour cette Tesselle"])
            ]
            return

        rules_list = []
        for i, rule in enumerate(tesselle.rules):
            rules_list.append(
                v.ListItem(
                    children=[
                        v.ListItemIcon(children=[v.Icon(children=["mdi-arrow-right"])]),
                        v.ListItemContent(children=[v.ListItemTitle(children=[rule])]),
                    ]
                )
            )

        self.rules_content.children = [v.List(dense=True, children=rules_list)]

    def _update_counterfactual_tab(self, tesselle):
        """Met à jour l'onglet des contrefactuels."""
        if tesselle.archetype_idx is None:
            self.cf_content.children = [
                v.Html(tag="p", children=["Pas d'archétype défini pour cette Tesselle"])
            ]
            return

        content = [
            v.Row(
                children=[
                    v.Col(
                        children=[
                            v.Html(tag="strong", children=["Archétype:"]),
                            v.Html(
                                tag="span",
                                class_="ml-2",
                                children=[f"Point {tesselle.archetype_idx}"],
                            ),
                        ]
                    )
                ]
            )
        ]

        if tesselle.counterfactual_idx:
            content.append(
                v.Row(
                    children=[
                        v.Col(
                            children=[
                                v.Html(tag="strong", children=["Contrefactuel:"]),
                                v.Html(
                                    tag="span",
                                    class_="ml-2",
                                    children=[f"Point {tesselle.counterfactual_idx}"],
                                ),
                            ]
                        )
                    ]
                )
            )

        self.cf_content.children = content

    def display(self):
        """Affiche le widget."""
        if HAS_IPYVUETIFY:
            display(self.widget)
        else:
            print("ipyvuetify n'est pas installé. Utilisez `pip install ipyvuetify`.")

    def export(self, format: str = "json") -> str:
        """
        Exporte la tessellation dans le format spécifié.

        Parameters
        ----------
        format : str
            Format d'export : 'json', 'markdown', 'html'

        Returns
        -------
        str
            Contenu exporté
        """
        if self.result is None:
            return ""

        from antakia.tessellation.description import DescriptionFormat, describe_tessellation

        format_map = {
            "json": DescriptionFormat.JSON,
            "markdown": DescriptionFormat.MARKDOWN,
            "html": DescriptionFormat.HTML,
            "text": DescriptionFormat.TEXT,
        }

        desc = describe_tessellation(
            self.result.tesselles,
            self.feature_names,
            format=format_map.get(format, DescriptionFormat.TEXT),
        )

        return desc.content


def create_tessellation_figure(
    result: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    x_feature: int = 0,
    y_feature: int = 1,
    title: str = "Visualisation des Tesselles",
) -> go.Figure:
    """
    Crée une figure Plotly pour visualiser les Tesselles.

    Parameters
    ----------
    result : TessellationResult
        Résultat de la tessellation
    X : np.ndarray
        Données
    y : np.ndarray
        Cible
    feature_names : List[str], optional
        Noms des features
    x_feature : int
        Index de la feature pour l'axe X
    y_feature : int
        Index de la feature pour l'axe Y
    title : str
        Titre du graphique

    Returns
    -------
    go.Figure
        Figure Plotly
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly est requis. Installez avec: pip install plotly")

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]

    fig = go.Figure()

    # Couleurs par Tesselle
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    final_tesselles = result.tesselles.final_tesselles

    for i, tesselle in enumerate(final_tesselles):
        color = colors[i % len(colors)]

        indices = tesselle.indices

        fig.add_trace(
            go.Scatter(
                x=X[indices, x_feature],
                y=X[indices, y_feature],
                mode="markers",
                name=f"{tesselle.id} ({tesselle.size} pts)",
                marker=dict(color=color, size=8, opacity=0.7),
                text=[f"Point {idx}<br>Prédiction: {y[idx]:.2f}" for idx in indices],
                hoverinfo="text+name",
            )
        )

        # Ajouter l'archétype si disponible
        if tesselle.archetype_idx is not None:
            arch_idx = tesselle.archetype_idx
            fig.add_trace(
                go.Scatter(
                    x=[X[arch_idx, x_feature]],
                    y=[X[arch_idx, y_feature]],
                    mode="markers",
                    name=f"Archétype {tesselle.id}",
                    marker=dict(
                        color=color, size=15, symbol="star", line=dict(width=2, color="black")
                    ),
                    showlegend=False,
                    hoverinfo="text",
                    text=[f"Archétype de {tesselle.id}"],
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title=feature_names[x_feature],
        yaxis_title=feature_names[y_feature],
        legend_title="Tesselles",
        hovermode="closest",
    )

    return fig


def create_hierarchy_figure(result: Any, title: str = "Hiérarchie des Tesselles") -> go.Figure:
    """
    Crée une figure Plotly pour visualiser la hiérarchie des Tesselles (sunburst).

    Parameters
    ----------
    result : TessellationResult
        Résultat de la tessellation
    title : str
        Titre du graphique

    Returns
    -------
    go.Figure
        Figure Plotly (Sunburst)
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly est requis")

    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    for tesselle in result.tesselles:
        ids.append(tesselle.id)
        labels.append(f"{tesselle.id}\n{tesselle.size} pts")
        parents.append(tesselle.parent_id or "")
        values.append(tesselle.size)

        # Couleur selon la pureté
        purity = tesselle.purity_score
        colors.append(purity)

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors, colorscale="RdYlGn", showscale=True, colorbar=dict(title="Pureté")
            ),
            branchvalues="total",
        )
    )

    fig.update_layout(title=title, margin=dict(t=50, l=0, r=0, b=0))

    return fig
