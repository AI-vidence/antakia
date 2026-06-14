"""
AntakIA Theme Configuration.

Centralizes colors, spacing, and styling for consistent UI.
Supports light and dark modes.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ThemeColors:
    """Color palette for the theme."""

    # Primary colors
    primary: str = "#2c3e50"
    secondary: str = "#3498db"
    accent: str = "#9b59b6"

    # Status colors
    success: str = "#27ae60"
    warning: str = "#f39c12"
    error: str = "#e74c3c"
    info: str = "#17a2b8"

    # Background colors
    background: str = "#ffffff"
    surface: str = "#f8f9fa"
    card: str = "#ffffff"

    # Text colors
    text_primary: str = "#212529"
    text_secondary: str = "#6c757d"
    text_disabled: str = "#adb5bd"

    # Border
    border: str = "#dee2e6"

    # Region colors (for tessellation)
    region_palette: List[str] = field(
        default_factory=lambda: [
            "#e74c3c",  # Red
            "#3498db",  # Blue
            "#27ae60",  # Green
            "#9b59b6",  # Purple
            "#f39c12",  # Orange
            "#1abc9c",  # Teal
            "#e91e63",  # Pink
            "#00bcd4",  # Cyan
            "#ff5722",  # Deep Orange
            "#607d8b",  # Blue Grey
        ]
    )


@dataclass
class ThemeColorsDark:
    """Dark mode color palette."""

    # Primary colors
    primary: str = "#3498db"
    secondary: str = "#5dade2"
    accent: str = "#bb8fce"

    # Status colors
    success: str = "#2ecc71"
    warning: str = "#f1c40f"
    error: str = "#e74c3c"
    info: str = "#5bc0de"

    # Background colors
    background: str = "#1a1a2e"
    surface: str = "#16213e"
    card: str = "#0f3460"

    # Text colors
    text_primary: str = "#e8e8e8"
    text_secondary: str = "#b0b0b0"
    text_disabled: str = "#666666"

    # Border
    border: str = "#333333"

    # Region colors (brighter for dark mode)
    region_palette: List[str] = field(
        default_factory=lambda: [
            "#ff6b6b",  # Red
            "#4ecdc4",  # Teal
            "#45b7d1",  # Blue
            "#96ceb4",  # Green
            "#ffeaa7",  # Yellow
            "#dfe6e9",  # Light Grey
            "#fd79a8",  # Pink
            "#a29bfe",  # Purple
            "#fab1a0",  # Peach
            "#81ecec",  # Cyan
        ]
    )


@dataclass
class ThemeSpacing:
    """Spacing values."""

    xs: str = "4px"
    sm: str = "8px"
    md: str = "16px"
    lg: str = "24px"
    xl: str = "32px"
    xxl: str = "48px"


@dataclass
class ThemeTypography:
    """Typography settings."""

    font_family: str = "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    font_size_xs: str = "0.75rem"
    font_size_sm: str = "0.875rem"
    font_size_md: str = "1rem"
    font_size_lg: str = "1.25rem"
    font_size_xl: str = "1.5rem"
    font_size_xxl: str = "2rem"


class Theme:
    """
    Central theme manager for AntakIA GUI.

    Usage
    -----
    >>> from antakia.gui.theme import theme
    >>> theme.colors.primary  # '#2c3e50'
    >>> theme.toggle_dark_mode()
    >>> theme.colors.primary  # '#3498db' (dark mode)
    """

    def __init__(self):
        self._dark_mode = False
        self._light_colors = ThemeColors()
        self._dark_colors = ThemeColorsDark()
        self.spacing = ThemeSpacing()
        self.typography = ThemeTypography()
        self._observers = []

    @property
    def dark_mode(self) -> bool:
        """Check if dark mode is enabled."""
        return self._dark_mode

    @property
    def colors(self) -> ThemeColors:
        """Get current color palette."""
        return self._dark_colors if self._dark_mode else self._light_colors

    def toggle_dark_mode(self) -> bool:
        """Toggle dark mode and notify observers."""
        self._dark_mode = not self._dark_mode
        self._notify_observers()
        return self._dark_mode

    def set_dark_mode(self, enabled: bool) -> None:
        """Set dark mode explicitly."""
        if self._dark_mode != enabled:
            self._dark_mode = enabled
            self._notify_observers()

    def add_observer(self, callback) -> None:
        """Add observer for theme changes."""
        self._observers.append(callback)

    def remove_observer(self, callback) -> None:
        """Remove observer."""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self) -> None:
        """Notify all observers of theme change."""
        for callback in self._observers:
            try:
                callback(self)
            except Exception:
                pass

    def get_region_color(self, index: int) -> str:
        """Get color for a region by index."""
        palette = self.colors.region_palette
        return palette[index % len(palette)]

    def get_vuetify_theme(self) -> Dict:
        """Get Vuetify theme configuration."""
        c = self.colors
        return {
            "dark": self._dark_mode,
            "themes": {
                "light": {
                    "primary": c.primary,
                    "secondary": c.secondary,
                    "accent": c.accent,
                    "success": c.success,
                    "warning": c.warning,
                    "error": c.error,
                    "info": c.info,
                },
                "dark": {
                    "primary": self._dark_colors.primary,
                    "secondary": self._dark_colors.secondary,
                    "accent": self._dark_colors.accent,
                    "success": self._dark_colors.success,
                    "warning": self._dark_colors.warning,
                    "error": self._dark_colors.error,
                    "info": self._dark_colors.info,
                },
            },
        }

    def css_vars(self) -> str:
        """Generate CSS custom properties for the theme."""
        c = self.colors
        s = self.spacing
        return f"""
        :root {{
            --atk-primary: {c.primary};
            --atk-secondary: {c.secondary};
            --atk-accent: {c.accent};
            --atk-success: {c.success};
            --atk-warning: {c.warning};
            --atk-error: {c.error};
            --atk-info: {c.info};
            --atk-background: {c.background};
            --atk-surface: {c.surface};
            --atk-card: {c.card};
            --atk-text-primary: {c.text_primary};
            --atk-text-secondary: {c.text_secondary};
            --atk-border: {c.border};
            --atk-spacing-xs: {s.xs};
            --atk-spacing-sm: {s.sm};
            --atk-spacing-md: {s.md};
            --atk-spacing-lg: {s.lg};
            --atk-spacing-xl: {s.xl};
        }}
        """


# Global theme instance
theme = Theme()
