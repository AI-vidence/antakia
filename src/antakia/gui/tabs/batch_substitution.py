"""
Batch substitution module for multi-region asynchronous model training.

Allows selecting multiple regions and training substitution models
in parallel with progressive UI updates.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import ipyvuetify as v
from antakia_core.data_handler import ModelRegion

from antakia.config import AppConfig
from antakia.gui.helpers.data import DataStore
from antakia.utils.logging_utils import Log, conf_logger
from antakia.utils.stats import log_errors, stats_logger

logger = logging.getLogger(__name__)
conf_logger(logger)


class SubstitutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class RegionSubstitution:
    """Tracks the substitution state for a single region."""

    region: ModelRegion
    status: SubstitutionStatus = SubstitutionStatus.PENDING
    error: Optional[str] = None
    best_model: Optional[str] = None
    best_delta: Optional[float] = None


class BatchSubstitutionManager:
    """
    Manages batch substitution of multiple regions with async execution.

    Features:
    - Multi-region selection
    - Parallel model training with ThreadPoolExecutor
    - Progressive UI updates as each region completes
    - Summary view of all substitutions
    """

    MAX_WORKERS = 3  # Limit parallel training to avoid memory issues

    def __init__(
        self,
        data_store: DataStore,
        on_region_complete: Callable[[RegionSubstitution], None],
        on_all_complete: Callable[[], None],
    ):
        self.data_store = data_store
        self.on_region_complete = on_region_complete
        self.on_all_complete = on_all_complete

        self.substitutions: List[RegionSubstitution] = []
        self.executor: Optional[ThreadPoolExecutor] = None
        self.is_running = False

    def start_batch(self, regions: List[ModelRegion]):
        """Start batch substitution for multiple regions."""
        if self.is_running:
            logger.warning("Batch substitution already running")
            return

        self.substitutions = [
            RegionSubstitution(region=r)
            for r in regions
            if r.num_points() >= AppConfig.ATK_MIN_POINTS_NUMBER
        ]

        if not self.substitutions:
            logger.warning("No valid regions for substitution")
            return

        self.is_running = True
        logger.info(f"Starting batch substitution for {len(self.substitutions)} regions")

        # Start async execution
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        futures = {}

        for sub in self.substitutions:
            sub.status = SubstitutionStatus.RUNNING
            future = self.executor.submit(self._train_region, sub)
            futures[future] = sub

        # Process completions in a separate thread to not block UI
        import threading

        def process_results():
            for future in as_completed(futures):
                sub = futures[future]
                try:
                    future.result()  # Get result or raise exception
                    sub.status = SubstitutionStatus.COMPLETED
                except Exception as e:
                    sub.status = SubstitutionStatus.ERROR
                    sub.error = str(e)
                    logger.error(f"Error training region {sub.region.num}: {e}")

                # Callback for progressive update (must be thread-safe for UI)
                self.on_region_complete(sub)

            self.is_running = False
            self.on_all_complete()
            self.executor.shutdown(wait=False)

        threading.Thread(target=process_results, daemon=True).start()

    def _train_region(self, sub: RegionSubstitution):
        """Train substitution models for a single region."""
        logger.info(f"Training substitution models for region {sub.region.num}")

        sub.region.train_substitution_models(task_type=self.data_store.problem_category)

        # Extract best model info
        if len(sub.region.perfs) > 0:
            perfs = sub.region.perfs
            best_idx = perfs["delta"].idxmin()
            sub.best_model = best_idx
            sub.best_delta = perfs.loc[best_idx, "delta"]

            # Auto-select best model
            sub.region.select_model(best_idx)

        logger.info(
            f"Region {sub.region.num} complete: best={sub.best_model}, delta={sub.best_delta:.3f}"
        )

    def cancel(self):
        """Cancel ongoing batch substitution."""
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
        self.is_running = False


class BatchSubstitutionWidget:
    """
    Widget for displaying batch substitution progress and results.
    """

    def __init__(
        self,
        data_store: DataStore,
        validate_callback: Callable,
    ):
        self.data_store = data_store
        self.validate_callback = validate_callback

        self.manager = BatchSubstitutionManager(
            data_store,
            on_region_complete=self._on_region_complete,
            on_all_complete=self._on_all_complete,
        )

        self._build_widget()

    def _build_widget(self):
        """Build the batch substitution widget."""
        self.title = v.Html(tag="h3", class_="mb-3", children=["Batch Substitution"])

        self.progress_text = v.Html(
            tag="div",
            class_="mb-2 grey--text",
            children=["Select regions and click 'Substitute All'"],
        )

        self.results_table = v.DataTable(
            headers=[
                {"text": "Region", "value": "region", "sortable": False},
                {"text": "Status", "value": "status", "sortable": False},
                {"text": "Best Model", "value": "best_model", "sortable": False},
                {"text": "Delta", "value": "delta", "sortable": False},
            ],
            items=[],
            hide_default_footer=True,
            dense=True,
            class_="elevation-1",
        )

        self.validate_all_btn = v.Btn(
            class_="ma-1 green white--text",
            disabled=True,
            children=[
                v.Icon(class_="mr-2", children=["mdi-check-all"]),
                "Validate All",
            ],
        )
        self.validate_all_btn.on_event("click", self._validate_all_clicked)

        self.widget = [
            v.Col(
                children=[
                    self.title,
                    self.progress_text,
                    self.results_table,
                    v.Row(
                        class_="mt-3",
                        children=[self.validate_all_btn],
                    ),
                ]
            )
        ]

    def start_batch(self, regions: List[ModelRegion]):
        """Start batch substitution."""
        # Initialize table with pending status
        items = []
        for region in regions:
            if region.num_points() >= AppConfig.ATK_MIN_POINTS_NUMBER:
                items.append(
                    {
                        "region": f"Region {region.num}",
                        "region_num": region.num,
                        "status": "⏳ Pending",
                        "best_model": "-",
                        "delta": "-",
                    }
                )

        self.results_table.items = items
        self.progress_text.children = [f"Training {len(items)} regions..."]
        self.validate_all_btn.disabled = True

        self.manager.start_batch(regions)

    def _on_region_complete(self, sub: RegionSubstitution):
        """Called when a region completes (from worker thread)."""
        # Update table item for this region
        items = list(self.results_table.items)
        for item in items:
            if item["region_num"] == sub.region.num:
                if sub.status == SubstitutionStatus.COMPLETED:
                    item["status"] = "✅ Complete"
                    item["best_model"] = sub.best_model or "-"
                    item["delta"] = f"{sub.best_delta:.3f}" if sub.best_delta else "-"
                elif sub.status == SubstitutionStatus.ERROR:
                    item["status"] = "❌ Error"
                    item["best_model"] = sub.error or "Unknown error"
                break

        self.results_table.items = items

        # Update progress text
        completed = sum(
            1
            for s in self.manager.substitutions
            if s.status in [SubstitutionStatus.COMPLETED, SubstitutionStatus.ERROR]
        )
        total = len(self.manager.substitutions)
        self.progress_text.children = [f"Progress: {completed}/{total} regions"]

    def _on_all_complete(self):
        """Called when all regions complete."""
        completed = sum(
            1 for s in self.manager.substitutions if s.status == SubstitutionStatus.COMPLETED
        )
        total = len(self.manager.substitutions)

        self.progress_text.children = [f"Complete: {completed}/{total} regions succeeded"]
        self.validate_all_btn.disabled = completed == 0

    @log_errors
    def _validate_all_clicked(self, *args):
        """Validate all successfully substituted regions."""
        with Log("validate_all_substitutions", 2):
            validated = 0
            for sub in self.manager.substitutions:
                if sub.status == SubstitutionStatus.COMPLETED and sub.best_model:
                    sub.region.validate()
                    validated += 1

            stats_logger.log("validate_all_substitutions", {"count": validated})
            self.progress_text.children = [f"Validated {validated} regions"]
            self.validate_all_btn.disabled = True
            self.validate_callback()
