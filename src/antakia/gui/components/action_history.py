"""
Action history panel for AntakIA GUI.

Tracks user actions and provides undo/redo functionality (future).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, Optional

import ipyvuetify as v

ActionType = Literal[
    "region_created",
    "region_deleted",
    "region_validated",
    "shap_computed",
    "substitution_trained",
    "outliers_detected",
    "projection_changed",
    "session_loaded",
    "session_saved",
]


@dataclass
class ActionRecord:
    """Record of a user action."""

    timestamp: datetime
    action_type: ActionType
    description: str
    details: Optional[str] = None
    icon: str = "mdi-circle-small"
    color: str = "grey"


class ActionHistory:
    """
    Manages and displays action history.

    Usage
    -----
    >>> from antakia.gui.components.action_history import action_history
    >>> action_history.add("region_created", "Region 1 created", "MedInc > 5.2")
    >>> action_history.add("shap_computed", "SHAP computed", "TreeExplainer, 1234 pts")
    """

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.records: List[ActionRecord] = []
        self._build_widget()

    def _build_widget(self):
        """Build the history panel widget."""
        self.history_list = v.List(
            dense=True,
            class_="pa-0",
            children=[],
        )

        self.clear_btn = v.Btn(
            small=True,
            text=True,
            children=["Clear"],
        )
        self.clear_btn.on_event("click", self._on_clear)

        self.widget = v.Card(
            class_="elevation-1",
            children=[
                v.CardTitle(
                    class_="py-2",
                    children=[
                        v.Icon(small=True, class_="mr-2", children=["mdi-history"]),
                        "History",
                        v.Spacer(),
                        self.clear_btn,
                    ],
                ),
                v.Divider(),
                v.Sheet(
                    style_="max-height: 300px; overflow-y: auto;",
                    children=[self.history_list],
                ),
            ],
        )

    def _get_icon_and_color(self, action_type: ActionType) -> tuple:
        """Get icon and color for action type."""
        mapping = {
            "region_created": ("mdi-shape-plus", "success"),
            "region_deleted": ("mdi-shape-minus", "error"),
            "region_validated": ("mdi-check-circle", "success"),
            "shap_computed": ("mdi-chart-scatter-plot", "info"),
            "substitution_trained": ("mdi-swap-horizontal", "primary"),
            "outliers_detected": ("mdi-alert-circle-outline", "warning"),
            "projection_changed": ("mdi-axis-arrow", "secondary"),
            "session_loaded": ("mdi-folder-open", "info"),
            "session_saved": ("mdi-content-save", "success"),
        }
        return mapping.get(action_type, ("mdi-circle-small", "grey"))

    def add(
        self,
        action_type: ActionType,
        description: str,
        details: Optional[str] = None,
    ) -> None:
        """
        Add an action to history.

        Parameters
        ----------
        action_type : ActionType
            Type of action performed
        description : str
            Short description of the action
        details : str, optional
            Additional details
        """
        icon, color = self._get_icon_and_color(action_type)

        record = ActionRecord(
            timestamp=datetime.now(),
            action_type=action_type,
            description=description,
            details=details,
            icon=icon,
            color=color,
        )

        self.records.insert(0, record)

        # Trim old records
        if len(self.records) > self.max_history:
            self.records = self.records[: self.max_history]

        self._update_display()

    def _update_display(self):
        """Update the history list display."""
        items = []

        for record in self.records[:20]:  # Show last 20
            time_str = record.timestamp.strftime("%H:%M:%S")

            item = v.ListItem(
                dense=True,
                children=[
                    v.ListItemIcon(
                        class_="mr-2",
                        children=[
                            v.Icon(
                                small=True,
                                color=record.color,
                                children=[record.icon],
                            )
                        ],
                    ),
                    v.ListItemContent(
                        children=[
                            v.ListItemTitle(
                                class_="text-body-2",
                                children=[record.description],
                            ),
                            v.ListItemSubtitle(
                                class_="text-caption",
                                children=[
                                    f"{time_str}"
                                    + (f" - {record.details}" if record.details else "")
                                ],
                            ),
                        ]
                    ),
                ],
            )
            items.append(item)

        if not items:
            items = [
                v.ListItem(
                    children=[
                        v.ListItemContent(
                            children=[
                                v.ListItemTitle(
                                    class_="text-caption grey--text",
                                    children=["No actions yet"],
                                )
                            ]
                        )
                    ]
                )
            ]

        self.history_list.children = items

    def _on_clear(self, widget, event, data):
        """Clear history."""
        self.records = []
        self._update_display()

    def clear(self):
        """Clear all history."""
        self.records = []
        self._update_display()


# Global action history instance
action_history = ActionHistory()
