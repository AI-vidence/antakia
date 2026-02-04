"""
Toggle pour l'affichage des règles de sélection.

Icône activée quand des règles existent, grisée sinon.
"""

from typing import Callable

import ipyvuetify as v


class RulesDisplayToggle:
    """
    Toggle button for showing/hiding selection rules display.

    The icon is active when rules exist, grayed out otherwise.
    """

    def __init__(
        self,
        has_rules_getter: Callable[[], bool],
        show_rules_getter: Callable[[], bool],
        toggle_callback: Callable[[], None],
    ):
        """
        Initialize the rules display toggle.

        Parameters
        ----------
        has_rules_getter : Callable
            Function that returns True if rules exist
        show_rules_getter : Callable
            Function that returns True if rules should be shown
        toggle_callback : Callable
            Function to call when toggle is clicked
        """
        self.has_rules_getter = has_rules_getter
        self.show_rules_getter = show_rules_getter
        self.toggle_callback = toggle_callback

        self._build_widget()

    def _build_widget(self):
        """Build the toggle widget."""
        self.icon = v.Icon(
            children=["mdi-filter-variant"],
            color="primary",
        )

        self.widget = v.Tooltip(
            bottom=True,
            v_slots=[
                {
                    "name": "activator",
                    "variable": "tooltip",
                    "children": v.Btn(
                        v_bind="tooltip.attrs",
                        v_on="tooltip.on",
                        icon=True,
                        small=True,
                        children=[self.icon],
                    ),
                }
            ],
            children=["Afficher/masquer les règles"],
        )

        # Wire click event (activator slot is in v_slots, not in children)
        btn = self.widget.v_slots[0]["children"]
        btn.on_event("click", lambda *_: self._on_click())

        self.refresh()

    def refresh(self):
        """Update the toggle appearance based on current state."""
        has_rules = self.has_rules_getter()
        show_rules = self.show_rules_getter()

        if not has_rules:
            # No rules - gray and disabled
            self.icon.color = "grey lighten-1"
            self.icon.children = ["mdi-filter-variant-remove"]
        elif show_rules:
            # Rules shown - active
            self.icon.color = "primary"
            self.icon.children = ["mdi-filter-variant"]
        else:
            # Rules hidden - outlined
            self.icon.color = "grey"
            self.icon.children = ["mdi-filter-variant"]

    def _on_click(self):
        """Handle click event."""
        if self.has_rules_getter():
            self.toggle_callback()
            self.refresh()
