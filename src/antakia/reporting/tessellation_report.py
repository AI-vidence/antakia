"""
Rapport de tessellation détaillé.

Génère un rapport complet du modèle ensembliste avec toutes les tesselles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence

if TYPE_CHECKING:
    from antakia_core.data_handler import Region

    from antakia.antakia import AntakIA

logger = logging.getLogger(__name__)


@dataclass
class TesselleReport:
    """Rapport pour une tesselle individuelle."""

    # Identifiant
    region_num: int
    color: str

    # Statistiques
    n_points: int
    coverage_pct: float
    mean_y: float
    std_y: float

    # Règles
    rules_str: str
    rules_list: List[Dict[str, Any]]

    # Modèle surrogate
    surrogate_model_name: Optional[str] = None
    surrogate_score: Optional[float] = None

    # Explications globales (SHAP)
    shap_values: Optional[pd.DataFrame] = None
    shap_importance: Optional[pd.Series] = None

    # Top features
    top_features: List[str] = field(default_factory=list)

    # PDP data (feature -> (grid, pdp_initial, pdp_initial_filtered, pdp_surrogate))
    pdp_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]] = field(
        default_factory=dict
    )


@dataclass
class TessellationReportResult:
    """Résultat complet du rapport de tessellation."""

    # Métadonnées
    model_name: str
    n_samples: int
    n_features: int
    feature_names: List[str]

    # Statistiques globales
    total_regions: int
    total_coverage_pct: float
    initial_model_score: Optional[float] = None
    ensemble_score: Optional[float] = None

    # Rapports par tesselle
    tesselle_reports: List[TesselleReport] = field(default_factory=list)

    # Données pour visualisations globales
    global_shap_importance: Optional[pd.Series] = None

    def display(self):
        """Affiche le rapport dans un notebook Jupyter."""
        from IPython.display import Markdown, display

        from antakia.reporting.visualizations import (
            plot_pdp_comparison,
            plot_shap_summary,
        )

        # En-tête
        display(Markdown("# Rapport de Tessellation"))
        display(Markdown(f"**Modèle**: {self.model_name}"))
        display(Markdown(f"**Dataset**: {self.n_samples} points, {self.n_features} features"))
        display(
            Markdown(
                f"**Tesselles**: {self.total_regions} régions ({self.total_coverage_pct:.1f}% couverture)"
            )
        )

        if self.initial_model_score is not None:
            display(Markdown(f"**Score modèle initial**: {self.initial_model_score:.4f}"))
        if self.ensemble_score is not None:
            display(Markdown(f"**Score modèle ensembliste**: {self.ensemble_score:.4f}"))

        display(Markdown("---"))

        # Rapport par tesselle
        for tr in self.tesselle_reports:
            display(Markdown(f"## Tesselle {tr.region_num} ({tr.color})"))
            display(Markdown(f"- **Points**: {tr.n_points} ({tr.coverage_pct:.1f}%)"))
            display(Markdown(f"- **Règles**: {tr.rules_str}"))
            display(Markdown(f"- **y moyen**: {tr.mean_y:.3f} (±{tr.std_y:.3f})"))

            if tr.surrogate_model_name:
                display(
                    Markdown(
                        f"- **Surrogate**: {tr.surrogate_model_name} (score={tr.surrogate_score:.4f})"
                    )
                )

            if tr.top_features:
                display(Markdown(f"- **Top features**: {', '.join(tr.top_features[:5])}"))

            # Visualisations
            if tr.shap_values is not None and len(tr.shap_values) > 0:
                display(Markdown("### SHAP Summary"))
                fig = plot_shap_summary(tr.shap_values, max_features=10)
                display(fig)

            if tr.pdp_data:
                display(Markdown("### PDP Comparatif"))
                for feature, (grid, pdp_init, pdp_filt, pdp_surr) in list(tr.pdp_data.items())[:3]:
                    fig = plot_pdp_comparison(
                        feature, grid, pdp_init, pdp_filt, pdp_surr, region_num=tr.region_num
                    )
                    display(fig)

            display(Markdown("---"))


class TessellationReport:
    """
    Générateur de rapports de tessellation.

    Analyse un modèle AntakIA avec ses régions/tesselles et génère
    un rapport détaillé avec explications globales et comparaisons.

    Examples
    --------
    >>> from antakia.reporting import TessellationReport
    >>>
    >>> # Après avoir créé des régions dans le GUI
    >>> report = TessellationReport(atk)
    >>> result = report.generate_report()
    >>>
    >>> # Affichage inline
    >>> result.display()
    >>>
    >>> # Export
    >>> report.export_html("rapport.html")
    """

    def __init__(self, atk: "AntakIA"):
        """
        Initialise le générateur de rapport.

        Parameters
        ----------
        atk : AntakIA
            Instance AntakIA avec des régions définies
        """
        self.atk = atk
        self.data_store = atk.data_store
        self.X = self.data_store.X
        self.y = self.data_store.y
        self.model = self.data_store.model
        self.region_set = self.data_store.region_set

        # Cache pour les calculs
        self._shap_cache: Dict[int, pd.DataFrame] = {}

    def generate_report(
        self,
        compute_shap: bool = True,
        compute_pdp: bool = True,
        top_n_features: int = 5,
        pdp_grid_resolution: int = 50,
    ) -> TessellationReportResult:
        """
        Génère le rapport complet.

        Parameters
        ----------
        compute_shap : bool
            Calculer les SHAP values par tesselle
        compute_pdp : bool
            Calculer les PDP comparatifs
        top_n_features : int
            Nombre de top features à analyser
        pdp_grid_resolution : int
            Résolution de la grille PDP

        Returns
        -------
        TessellationReportResult
            Rapport complet
        """
        logger.info("Génération du rapport de tessellation...")

        # Métadonnées
        result = TessellationReportResult(
            model_name=type(self.model).__name__,
            n_samples=len(self.X),
            n_features=len(self.X.columns),
            feature_names=list(self.X.columns),
            total_regions=len(self.region_set),
            total_coverage_pct=self.region_set.stats()["coverage"],
        )

        # Score modèle initial
        if self.data_store.X_test is not None:
            try:
                result.initial_model_score = self.model.score(
                    self.data_store.X_test, self.data_store.y_test
                )
            except Exception as e:
                logger.warning(f"Impossible de calculer le score initial: {e}")

        # Rapport par tesselle
        for region in self.region_set:
            if region.num == -1:  # Skip "left outs"
                continue

            tr = self._generate_tesselle_report(
                region,
                compute_shap=compute_shap,
                compute_pdp=compute_pdp,
                top_n_features=top_n_features,
                pdp_grid_resolution=pdp_grid_resolution,
            )
            result.tesselle_reports.append(tr)

        # Score ensembliste (si tous les surrogates sont définis)
        # TODO: Calculer le score du modèle ensembliste

        logger.info(f"Rapport généré: {len(result.tesselle_reports)} tesselles")
        return result

    def _generate_tesselle_report(
        self,
        region: "Region",
        compute_shap: bool,
        compute_pdp: bool,
        top_n_features: int,
        pdp_grid_resolution: int,
    ) -> TesselleReport:
        """Génère le rapport pour une tesselle."""
        mask = region.mask
        X_region = self.X[mask]
        y_region = self.y[mask]

        # Statistiques de base
        tr = TesselleReport(
            region_num=region.num,
            color=region.color,
            n_points=mask.sum(),
            coverage_pct=100 * mask.mean(),
            mean_y=y_region.mean(),
            std_y=y_region.std(),
            rules_str=region.rules_to_str()
            if hasattr(region, "rules_to_str")
            else str(region.rules),
            rules_list=region.rules.to_dict_list() if hasattr(region.rules, "to_dict_list") else [],
        )

        # Modèle surrogate
        if region.sub_model is not None:
            tr.surrogate_model_name = type(region.sub_model).__name__
            try:
                tr.surrogate_score = region.sub_model.score(X_region, y_region)
            except:
                pass

        # SHAP values pour cette tesselle
        if compute_shap and len(X_region) > 10:
            tr.shap_values = self._compute_shap_for_region(X_region)
            if tr.shap_values is not None:
                tr.shap_importance = tr.shap_values.abs().mean().sort_values(ascending=False)
                tr.top_features = list(tr.shap_importance.head(top_n_features).index)

        # PDP comparatif pour les top features
        if compute_pdp and tr.top_features:
            for feature in tr.top_features[:3]:  # Top 3 features
                pdp_data = self._compute_pdp_comparison(
                    feature, mask, region.sub_model, pdp_grid_resolution
                )
                if pdp_data is not None:
                    tr.pdp_data[feature] = pdp_data

        return tr

    def _compute_shap_for_region(self, X_region: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calcule les SHAP values pour les points d'une région."""
        try:
            # Utiliser TreeExplainer si possible
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_region)

            if isinstance(shap_values, list):
                shap_values = shap_values[-1]  # Classe positive

            return pd.DataFrame(shap_values, columns=X_region.columns, index=X_region.index)
        except Exception as e:
            logger.warning(f"Impossible de calculer SHAP: {e}")
            return None

    def _compute_pdp_comparison(
        self,
        feature: str,
        region_mask: pd.Series,
        surrogate_model: Optional[Any],
        grid_resolution: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        Calcule le PDP comparatif pour une feature.

        Returns
        -------
        tuple
            (grid, pdp_initial_all, pdp_initial_filtered, pdp_surrogate)
        """
        try:
            feature_idx = list(self.X.columns).index(feature)

            # Grille commune
            feature_values = self.X[feature]
            grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

            # PDP modèle initial (tout le dataset)
            pdp_init_all = partial_dependence(
                self.model, self.X, [feature_idx], grid_resolution=grid_resolution, kind="average"
            )

            # PDP modèle initial (filtré sur la région)
            X_region = self.X[region_mask]
            if len(X_region) > 10:
                pdp_init_filtered = partial_dependence(
                    self.model,
                    X_region,
                    [feature_idx],
                    grid_resolution=grid_resolution,
                    kind="average",
                )
            else:
                pdp_init_filtered = pdp_init_all

            # PDP modèle surrogate (si disponible)
            pdp_surrogate = None
            if surrogate_model is not None and len(X_region) > 10:
                try:
                    pdp_surrogate = partial_dependence(
                        surrogate_model,
                        X_region,
                        [feature_idx],
                        grid_resolution=grid_resolution,
                        kind="average",
                    )
                except:
                    pass

            return (
                pdp_init_all["grid_values"][0],
                pdp_init_all["average"][0],
                pdp_init_filtered["average"][0],
                pdp_surrogate["average"][0] if pdp_surrogate else None,
            )

        except Exception as e:
            logger.warning(f"Impossible de calculer PDP pour {feature}: {e}")
            return None

    def export_html(
        self,
        path: str,
        include_visualizations: bool = True,
        use_template: bool = True,
    ) -> None:
        """
        Exporte le rapport en HTML interactif.

        Parameters
        ----------
        path : str
            Chemin du fichier HTML
        include_visualizations : bool
            Inclure les visualisations (SHAP, PDP) en base64
        use_template : bool
            Utiliser le template Jinja2 (True) ou le HTML basique (False)
        """
        result = self.generate_report()

        if use_template:
            html_content = self._render_html_jinja(result, include_visualizations)
        else:
            html_content = self._render_html_basic(result)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Rapport exporté: {path}")

    def _render_html_jinja(
        self, result: TessellationReportResult, include_visualizations: bool
    ) -> str:
        """Génère le HTML du rapport avec Jinja2."""
        import base64
        import io
        from datetime import datetime
        from pathlib import Path

        try:
            from jinja2 import Environment, FileSystemLoader
        except ImportError:
            logger.warning("jinja2 non installé, utilisation du template basique")
            return self._render_html_basic(result)

        from antakia.reporting.visualizations import (
            plot_pdp_comparison,
            plot_shap_summary,
            plot_tessellation_overview,
        )

        # Charger le template
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))

        # Filtre personnalisé pour formater les nombres
        def format_number(value):
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:,.2f}"
            return f"{value:,}"

        env.filters["format_number"] = format_number

        template = env.get_template("report.html.j2")

        # Préparer les données pour le template
        context = {
            "model_name": result.model_name,
            "n_samples": result.n_samples,
            "n_features": result.n_features,
            "feature_names": result.feature_names,
            "total_regions": result.total_regions,
            "total_coverage_pct": result.total_coverage_pct,
            "initial_model_score": result.initial_model_score,
            "ensemble_score": result.ensemble_score,
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tesselle_reports": [],
            "overview_image": None,
        }

        # Générer l'image overview si visualisations demandées
        if include_visualizations and result.tesselle_reports:
            try:
                region_stats = [
                    {
                        "num": tr.region_num,
                        "color": tr.color,
                        "coverage": tr.coverage_pct,
                        "score": tr.surrogate_score or 0,
                    }
                    for tr in result.tesselle_reports
                ]
                fig = plot_tessellation_overview(region_stats)
                context["overview_image"] = self._fig_to_base64(fig)
            except Exception as e:
                logger.warning(f"Impossible de générer l'overview: {e}")

        # Préparer les rapports de tesselles avec images
        for tr in result.tesselle_reports:
            tesselle_data = {
                "region_num": tr.region_num,
                "color": tr.color,
                "n_points": tr.n_points,
                "coverage_pct": tr.coverage_pct,
                "mean_y": tr.mean_y,
                "std_y": tr.std_y,
                "rules_str": tr.rules_str,
                "surrogate_model_name": tr.surrogate_model_name,
                "surrogate_score": tr.surrogate_score,
                "top_features": tr.top_features,
                "shap_image": None,
                "pdp_images": {},
            }

            if include_visualizations:
                # SHAP image
                if tr.shap_values is not None and len(tr.shap_values) > 0:
                    try:
                        fig = plot_shap_summary(tr.shap_values, max_features=10)
                        tesselle_data["shap_image"] = self._fig_to_base64(fig)
                    except Exception as e:
                        logger.warning(f"Impossible de générer SHAP plot: {e}")

                # PDP images
                for feature, (grid, pdp_init, pdp_filt, pdp_surr) in tr.pdp_data.items():
                    try:
                        fig = plot_pdp_comparison(
                            feature, grid, pdp_init, pdp_filt, pdp_surr, tr.region_num
                        )
                        tesselle_data["pdp_images"][feature] = self._fig_to_base64(fig)
                    except Exception as e:
                        logger.warning(f"Impossible de générer PDP plot pour {feature}: {e}")

            context["tesselle_reports"].append(tesselle_data)

        return template.render(**context)

    def _fig_to_base64(self, fig) -> str:
        """Convertit une figure matplotlib en base64."""
        import base64
        import io

        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64

    def _render_html_basic(self, result: TessellationReportResult) -> str:
        """Génère le HTML du rapport (version basique sans Jinja2)."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rapport de Tessellation - {result.model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .tesselle {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .tesselle h3 {{ margin-top: 0; }}
    </style>
</head>
<body>
    <h1>Rapport de Tessellation</h1>

    <div class="stats">
        <p><strong>Modèle:</strong> {result.model_name}</p>
        <p><strong>Dataset:</strong> {result.n_samples} points, {result.n_features} features</p>
        <p><strong>Tesselles:</strong> {result.total_regions} régions ({result.total_coverage_pct:.1f}% couverture)</p>
    </div>
"""

        for tr in result.tesselle_reports:
            html += f"""
    <div class="tesselle">
        <h3>Tesselle {tr.region_num} <span style="color:{tr.color}">●</span></h3>
        <p><strong>Points:</strong> {tr.n_points} ({tr.coverage_pct:.1f}%)</p>
        <p><strong>Règles:</strong> {tr.rules_str}</p>
        <p><strong>y moyen:</strong> {tr.mean_y:.3f} (±{tr.std_y:.3f})</p>
        {f'<p><strong>Surrogate:</strong> {tr.surrogate_model_name} (score={tr.surrogate_score:.4f})</p>' if tr.surrogate_model_name else ''}
        {f'<p><strong>Top features:</strong> {", ".join(tr.top_features[:5])}</p>' if tr.top_features else ''}
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def export_pdf(self, path: str, include_visualizations: bool = True) -> None:
        """
        Exporte le rapport en PDF.

        Nécessite weasyprint.

        Parameters
        ----------
        path : str
            Chemin du fichier PDF
        include_visualizations : bool
            Inclure les visualisations (SHAP, PDP)
        """
        try:
            import weasyprint

            result = self.generate_report()
            html_content = self._render_html_jinja(result, include_visualizations)
            weasyprint.HTML(string=html_content).write_pdf(path)
            logger.info(f"Rapport PDF exporté: {path}")
        except ImportError:
            logger.error(
                "weasyprint non installé. Installez avec: pip install weasyprint"
            )
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'export PDF: {e}")
            raise
