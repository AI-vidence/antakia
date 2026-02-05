"""
Tab 4: Report Generation

Allows generating and exporting tessellation reports.
"""

import ipyvuetify as v
from IPython.display import display, HTML

from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import Log
from antakia.utils.stats import log_errors


class Tab4:
    """Report generation tab."""

    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self._report_result = None
        self._build_widget()

    def _build_widget(self):
        """Build the report tab UI."""
        
        # Header
        self.header = v.Html(
            tag="h3",
            class_="mb-4",
            children=["Tessellation Report"],
        )
        
        # Description
        self.description = v.Html(
            tag="p",
            class_="text-body-2 grey--text mb-4",
            children=[
                "Generate a detailed report of your tessellation with SHAP analysis, "
                "PDP comparisons, and surrogate model performance."
            ],
        )
        
        # Options
        self.include_shap = v.Checkbox(
            v_model=True,
            label="Include SHAP analysis",
            class_="mt-2",
        )
        
        self.include_pdp = v.Checkbox(
            v_model=True,
            label="Include PDP comparisons",
            class_="mt-0",
        )
        
        self.top_features = v.Slider(
            v_model=5,
            min=3,
            max=10,
            step=1,
            label="Top features to analyze",
            thumb_label="always",
            class_="mt-4",
            style_="max-width: 300px",
        )
        
        # Generate button
        self.generate_btn = v.Btn(
            color="primary",
            class_="mt-4",
            children=[
                v.Icon(class_="mr-2", children=["mdi-file-document-outline"]),
                "Generate Report",
            ],
        )
        self.generate_btn.on_event("click", self._on_generate_click)
        
        # Export buttons
        self.export_html_btn = v.Btn(
            color="success",
            class_="mt-2 mr-2",
            disabled=True,
            children=[
                v.Icon(class_="mr-2", children=["mdi-language-html5"]),
                "Export HTML",
            ],
        )
        self.export_html_btn.on_event("click", self._on_export_html)
        
        self.export_pdf_btn = v.Btn(
            color="error",
            class_="mt-2",
            disabled=True,
            children=[
                v.Icon(class_="mr-2", children=["mdi-file-pdf-box"]),
                "Export PDF",
            ],
        )
        self.export_pdf_btn.on_event("click", self._on_export_pdf)
        
        # Progress indicator
        self.progress = v.ProgressLinear(
            indeterminate=True,
            color="primary",
            class_="mt-4",
            style_="display: none;",
        )
        
        # Status message
        self.status = v.Html(
            tag="p",
            class_="text-body-2 mt-2",
            children=[""],
        )
        
        # Report output area (Container pour afficher correctement les Cards et widgets)
        self.report_output = v.Container(
            fluid=True,
            class_="mt-4 pa-4",
            style_="border: 1px solid #ddd; border-radius: 8px; min-height: 200px; background: #fafafa;",
            children=[
                v.Html(
                    tag="p",
                    class_="grey--text text-center",
                    children=["Report will appear here after generation"],
                )
            ],
        )
        
        # Assemble widget
        self.widget = [
            v.Container(
                class_="pa-4",
                children=[
                    self.header,
                    self.description,
                    v.Divider(class_="mb-4"),
                    
                    # Options section
                    v.Html(tag="h4", class_="mb-2", children=["Options"]),
                    self.include_shap,
                    self.include_pdp,
                    self.top_features,
                    
                    v.Divider(class_="my-4"),
                    
                    # Actions
                    v.Row(
                        children=[
                            self.generate_btn,
                        ]
                    ),
                    self.progress,
                    self.status,
                    
                    v.Row(
                        class_="mt-4",
                        children=[
                            self.export_html_btn,
                            self.export_pdf_btn,
                        ]
                    ),
                    
                    v.Divider(class_="my-4"),
                    
                    # Report preview
                    v.Html(tag="h4", class_="mb-2", children=["Report Preview"]),
                    self.report_output,
                ]
            )
        ]

    @log_errors
    def _on_generate_click(self, widget, event, data):
        """Generate the tessellation report."""
        with Log("generate_report", 1):
            # Check if there are regions
            if len(self.data_store.region_set) == 0:
                self.status.children = ["⚠️ No regions defined. Create regions first."]
                return
            
            self._show_progress(True)
            self.status.children = ["Generating report..."]
            
            try:
                from antakia.reporting import TessellationReport
                
                # Create a mock AntakIA-like object for the report
                class ReportContext:
                    def __init__(self, data_store):
                        self.data_store = data_store
                
                # Generate report
                report = TessellationReport.__new__(TessellationReport)
                report.data_store = self.data_store
                report.X = self.data_store.X
                report.y = self.data_store.y
                report.model = self.data_store.model
                report.region_set = self.data_store.region_set
                report._shap_cache = {}
                
                self._report_result = report.generate_report(
                    compute_shap=self.include_shap.v_model,
                    compute_pdp=self.include_pdp.v_model,
                    top_n_features=int(self.top_features.v_model),
                )
                
                # Display summary (récap + cartes par tesselle)
                self._display_report_summary()
                
                # Afficher aussi le rapport HTML complet avec graphiques (iframe)
                self._display_report_html_preview()
                
                # Enable export buttons
                self.export_html_btn.disabled = False
                self.export_pdf_btn.disabled = False
                
                self.status.children = [
                    f"✅ Report generated: {len(self._report_result.tesselle_reports)} tesselles"
                ]
                
            except Exception as e:
                self.status.children = [f"❌ Error: {str(e)}"]
                import traceback
                traceback.print_exc()
            finally:
                self._show_progress(False)

    def _display_report_summary(self):
        """Display a summary of the generated report."""
        if self._report_result is None:
            return
        
        r = self._report_result
        
        # Build summary HTML
        summary_items = [
            v.Html(tag="h4", children=[f"Model: {r.model_name}"]),
            v.Html(tag="p", children=[
                f"Dataset: {r.n_samples:,} samples × {r.n_features} features"
            ]),
            v.Html(tag="p", children=[
                f"Tesselles: {r.total_regions} ({r.total_coverage_pct:.1f}% coverage)"
            ]),
        ]
        
        if r.initial_model_score is not None:
            summary_items.append(
                v.Html(tag="p", children=[f"Initial model score: {r.initial_model_score:.4f}"])
            )
        
        # Add tesselle cards
        summary_items.append(v.Divider(class_="my-3"))
        summary_items.append(v.Html(tag="h4", children=["Tesselles"]))
        
        for tr in r.tesselle_reports:
            card = v.Card(
                class_="ma-2 pa-3",
                outlined=True,
                children=[
                    v.CardTitle(
                        class_="py-1",
                        children=[
                            v.Html(
                                tag="span",
                                style_=f"color: {tr.color}; font-size: 1.5em; margin-right: 8px;",
                                children=["●"],
                            ),
                            f"Tesselle {tr.region_num}",
                        ]
                    ),
                    v.CardText(
                        children=[
                            f"Points: {tr.n_points} ({tr.coverage_pct:.1f}%)",
                            v.Html(tag="br"),
                            f"y mean: {tr.mean_y:.3f} (±{tr.std_y:.3f})",
                            v.Html(tag="br"),
                            f"Rules: {tr.rules_str[:50]}..." if len(tr.rules_str) > 50 else f"Rules: {tr.rules_str}",
                        ]
                    ),
                ]
            )
            summary_items.append(card)
        
        self.report_output.children = summary_items

    def _display_report_html_preview(self):
        """Affiche le rapport HTML complet (avec graphiques SHAP/PDP) dans le notebook.

        Utilise IPython.display.HTML car l'iframe avec data URL est souvent bloquée
        par la politique de sécurité (CSP) de Jupyter, ce qui rendait les graphiques vides.
        """
        if self._report_result is None:
            return
        try:
            from antakia.reporting import TessellationReport

            report = TessellationReport.__new__(TessellationReport)
            report.data_store = self.data_store
            report.X = self.data_store.X
            report.y = self.data_store.y
            report.model = self.data_store.model
            report.region_set = self.data_store.region_set
            report._shap_cache = {}

            html_content = report._render_html_jinja(
                self._report_result, include_visualizations=True
            )
            # Affichage direct dans le notebook (évite le blocage CSP des iframes)
            display(HTML(html_content))
            self.report_output.children = list(self.report_output.children) + [
                v.Divider(class_="my-4"),
                v.Html(
                    tag="p",
                    class_="grey--text",
                    children=[
                        "Rapport complet affiché ci-dessus. Utilisez « Export HTML » pour sauvegarder."
                    ],
                ),
            ]
        except Exception as e:
            self.report_output.children = list(self.report_output.children) + [
                v.Html(
                    tag="p",
                    class_="orange--text",
                    children=[f"Rapport HTML non affiché: {e}. Utilisez Export HTML."],
                )
            ]

    def _show_progress(self, show: bool):
        """Show/hide progress indicator."""
        self.progress.style_ = "" if show else "display: none;"
        self.generate_btn.disabled = show

    @log_errors
    def _on_export_html(self, widget, event, data):
        """Export report to HTML."""
        if self._report_result is None:
            return
        
        self.status.children = ["Exporting to HTML..."]
        
        try:
            from antakia.reporting import TessellationReport
            import tempfile
            import os
            
            # Create report object
            report = TessellationReport.__new__(TessellationReport)
            report.data_store = self.data_store
            report.X = self.data_store.X
            report.y = self.data_store.y
            report.model = self.data_store.model
            report.region_set = self.data_store.region_set
            report._shap_cache = {}
            
            # Export to temp file
            output_path = "tessellation_report.html"
            report.export_html(output_path, include_visualizations=True)
            
            self.status.children = [f"✅ HTML exported: {output_path}"]
            
        except Exception as e:
            self.status.children = [f"❌ Export failed: {str(e)}"]

    @log_errors
    def _on_export_pdf(self, widget, event, data):
        """Export report to PDF."""
        if self._report_result is None:
            return
        
        self.status.children = ["Exporting to PDF (requires weasyprint)..."]
        
        try:
            from antakia.reporting import TessellationReport
            
            # Create report object
            report = TessellationReport.__new__(TessellationReport)
            report.data_store = self.data_store
            report.X = self.data_store.X
            report.y = self.data_store.y
            report.model = self.data_store.model
            report.region_set = self.data_store.region_set
            report._shap_cache = {}
            
            output_path = "tessellation_report.pdf"
            report.export_pdf(output_path, include_visualizations=True)
            
            self.status.children = [f"✅ PDF exported: {output_path}"]
            
        except ImportError as e:
            self.status.children = [
                "❌ Aucun moteur PDF. Installez: pip install xhtml2pdf"
            ]
        except Exception as e:
            self.status.children = [f"❌ Export failed: {str(e)}"]
