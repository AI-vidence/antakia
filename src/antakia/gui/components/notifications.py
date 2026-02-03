"""
Notification system for AntakIA GUI.

Provides toast-style notifications for user feedback.
"""

from typing import Literal, Optional

import ipyvuetify as v

NotificationType = Literal["success", "info", "warning", "error"]


class NotificationManager:
    """
    Manages toast notifications in the GUI.

    Usage
    -----
    >>> from antakia.gui.components.notifications import notifications
    >>> notifications.success("Region created successfully")
    >>> notifications.error("Failed to compute SHAP values")
    >>> notifications.info("Computing...", timeout=0)  # persistent
    >>> notifications.clear()  # hide current notification
    """

    def __init__(self):
        self._build_widget()

    def _build_widget(self):
        """Build the snackbar widget."""
        self.icon = v.Icon(class_="mr-2", children=["mdi-check-circle"])
        self.message = v.Html(tag="span", children=[""])
        self.close_btn = v.Btn(
            icon=True,
            dark=True,
            children=[v.Icon(children=["mdi-close"])],
        )
        self.close_btn.on_event("click", self._on_close)

        self.snackbar = v.Snackbar(
            v_model=False,
            timeout=4000,
            top=True,
            right=True,
            children=[
                self.icon,
                self.message,
                v.Spacer(),
                self.close_btn,
            ],
        )

    @property
    def widget(self):
        """Get the snackbar widget."""
        return self.snackbar

    def _on_close(self, widget, event, data):
        """Handle close button click."""
        self.snackbar.v_model = False

    def _get_icon(self, type_: NotificationType) -> str:
        """Get icon for notification type."""
        icons = {
            "success": "mdi-check-circle",
            "info": "mdi-information",
            "warning": "mdi-alert",
            "error": "mdi-alert-circle",
        }
        return icons.get(type_, "mdi-information")

    def _get_color(self, type_: NotificationType) -> str:
        """Get color for notification type."""
        colors = {
            "success": "success",
            "info": "info",
            "warning": "warning",
            "error": "error",
        }
        return colors.get(type_, "info")

    def show(
        self,
        message: str,
        type_: NotificationType = "info",
        timeout: int = 4000,
    ) -> None:
        """
        Show a notification.

        Parameters
        ----------
        message : str
            The message to display
        type_ : NotificationType
            Type of notification (success, info, warning, error)
        timeout : int
            Milliseconds before auto-hide. 0 for persistent.
        """
        self.icon.children = [self._get_icon(type_)]
        self.message.children = [message]
        self.snackbar.color = self._get_color(type_)
        self.snackbar.timeout = timeout
        self.snackbar.v_model = True

    def success(self, message: str, timeout: int = 4000) -> None:
        """Show a success notification."""
        self.show(message, "success", timeout)

    def info(self, message: str, timeout: int = 4000) -> None:
        """Show an info notification."""
        self.show(message, "info", timeout)

    def warning(self, message: str, timeout: int = 5000) -> None:
        """Show a warning notification."""
        self.show(message, "warning", timeout)

    def error(self, message: str, timeout: int = 6000) -> None:
        """Show an error notification."""
        self.show(message, "error", timeout)

    def clear(self) -> None:
        """Hide the current notification."""
        self.snackbar.v_model = False


# Global notification manager
notifications = NotificationManager()
