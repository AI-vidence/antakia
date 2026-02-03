"""
Module de reporting pour AntakIA.

Génère des rapports détaillés sur les tessellations, incluant :
- Vue d'ensemble du modèle ensembliste
- Explications globales par tesselle (SHAP, PDP)
- Comparaison modèle initial vs modèle surrogate
"""

from antakia.reporting.tessellation_report import (
    TessellationReport,
    TessellationReportResult,
    TesselleReport,
)
from antakia.reporting.visualizations import (
    plot_feature_importance_comparison,
    plot_pdp_comparison,
    plot_rules_venn,
    plot_shap_summary,
    plot_tessellation_overview,
)

__all__ = [
    "TessellationReport",
    "TessellationReportResult",
    "TesselleReport",
    "plot_shap_summary",
    "plot_pdp_comparison",
    "plot_feature_importance_comparison",
    "plot_tessellation_overview",
    "plot_rules_venn",
]
