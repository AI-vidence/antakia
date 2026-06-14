"""
Advanced color management for AntakIA regions.

Features:
- Multiple color palettes (modern, pastel, vibrant, etc.)
- Automatic shade generation for subdivided regions
- Stable color assignment
- UI controls for palette switching
"""

import colorsys
from typing import Dict, List, Tuple

# ==================== Color Palettes ==================== #

# Modern palette - visually distinct, professional
PALETTE_MODERN = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#f39c12",  # Orange
    "#9b59b6",  # Purple
    "#1abc9c",  # Teal
    "#e91e63",  # Pink
    "#00bcd4",  # Cyan
    "#ff9800",  # Amber
    "#795548",  # Brown
]

# Pastel palette - soft, easy on the eyes
PALETTE_PASTEL = [
    "#a8d5e2",  # Light blue
    "#f9a8a8",  # Light red/coral
    "#a8e6cf",  # Light green
    "#ffd3a8",  # Light orange
    "#d5a8e6",  # Light purple
    "#a8e6e6",  # Light teal
    "#f9c8d8",  # Light pink
    "#c8e6f9",  # Light cyan
    "#e6d5a8",  # Light amber
    "#d5c8c8",  # Light brown
]

# Vibrant palette - high contrast, bold
PALETTE_VIBRANT = [
    "#ff6b6b",  # Coral red
    "#4ecdc4",  # Turquoise
    "#45b7d1",  # Sky blue
    "#96ceb4",  # Sage green
    "#ffeaa7",  # Pale yellow
    "#dfe6e9",  # Light gray
    "#fd79a8",  # Pink
    "#a29bfe",  # Lavender
    "#00b894",  # Mint
    "#fdcb6e",  # Mustard
]

# Categorical palette - for clear distinction (ColorBrewer-inspired)
PALETTE_CATEGORICAL = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# Earth tones - natural, warm
PALETTE_EARTH = [
    "#8d6e63",  # Brown
    "#a1887f",  # Taupe
    "#bcaaa4",  # Beige
    "#d7ccc8",  # Light beige
    "#795548",  # Dark brown
    "#6d4c41",  # Chocolate
    "#5d4037",  # Coffee
    "#4e342e",  # Espresso
    "#3e2723",  # Dark espresso
    "#efebe9",  # Cream
]

ALL_PALETTES = {
    "modern": PALETTE_MODERN,
    "pastel": PALETTE_PASTEL,
    "vibrant": PALETTE_VIBRANT,
    "categorical": PALETTE_CATEGORICAL,
    "earth": PALETTE_EARTH,
}

DEFAULT_PALETTE = "modern"


# ==================== Color Utilities ==================== #

# CSS named colors to hex mapping
CSS_COLORS = {
    "red": "#ff0000",
    "blue": "#0000ff",
    "green": "#008000",
    "yellow": "#ffff00",
    "orange": "#ffa500",
    "pink": "#ffc0cb",
    "brown": "#a52a2a",
    "cyan": "#00ffff",
    "black": "#000000",
    "white": "#ffffff",
    "grey": "#808080",
    "gray": "#808080",
    "purple": "#800080",
    "teal": "#008080",
    "navy": "#000080",
    "maroon": "#800000",
    "olive": "#808000",
    "lime": "#00ff00",
    "aqua": "#00ffff",
    "fuchsia": "#ff00ff",
    "silver": "#c0c0c0",
}


def normalize_color(color: str) -> str:
    """Convert any color format to hex."""
    if color is None:
        return "#808080"  # Default grey

    color = color.strip().lower()

    # Already hex
    if color.startswith("#"):
        return color

    # CSS named color
    if color in CSS_COLORS:
        return CSS_COLORS[color]

    # Unknown - return grey
    return "#808080"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = normalize_color(hex_color)
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"


def adjust_lightness(color: str, factor: float) -> str:
    """
    Adjust the lightness of a color.

    factor > 1.0 = lighter
    factor < 1.0 = darker

    Accepts hex colors (#ff0000) or CSS names (red).
    """
    hex_color = normalize_color(color)
    r, g, b = hex_to_rgb(hex_color)
    # Convert to HLS
    h, lightness, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    # Adjust lightness
    lightness = max(0, min(1, lightness * factor))
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return rgb_to_hex(int(r * 255), int(g * 255), int(b * 255))


def generate_shades(base_color: str, n_shades: int) -> List[str]:
    """
    Generate n shades of a base color.

    Returns colors from lighter to darker variations of the base.
    Accepts hex colors (#ff0000) or CSS names (red).
    """
    # Normalize the base color first
    normalized = normalize_color(base_color)

    if n_shades == 1:
        return [normalized]

    if n_shades <= 0:
        return []

    shades = []
    # Generate shades from light to dark
    factors = [1.3 - (i * 0.5 / (n_shades - 1)) for i in range(n_shades)]

    for factor in factors:
        shades.append(adjust_lightness(normalized, factor))

    return shades


def get_contrasting_text_color(hex_color: str) -> str:
    """Return black or white text color based on background brightness."""
    r, g, b = hex_to_rgb(hex_color)
    # Calculate luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.5 else "#ffffff"


# ==================== Color Manager Class ==================== #


class ColorManager:
    """
    Manages colors for regions with support for:
    - Multiple palettes
    - Automatic shade generation for subdivisions
    - Stable color assignment
    """

    def __init__(self, palette_name: str = DEFAULT_PALETTE):
        self.palette_name = palette_name
        self.palette = ALL_PALETTES.get(palette_name, PALETTE_MODERN)
        self._color_index = 0
        self._region_colors: Dict[int, str] = {}
        self._parent_colors: Dict[int, str] = {}  # Track parent colors for shading

    def reset(self):
        """Reset color assignments."""
        self._color_index = 0
        self._region_colors.clear()
        self._parent_colors.clear()

    def set_palette(self, palette_name: str):
        """Change the color palette."""
        if palette_name in ALL_PALETTES:
            self.palette_name = palette_name
            self.palette = ALL_PALETTES[palette_name]
            # Reassign colors with new palette
            self._reassign_colors()

    def _reassign_colors(self):
        """Reassign all colors with current palette."""
        old_assignments = list(self._region_colors.keys())
        self._region_colors.clear()
        self._color_index = 0
        for region_num in old_assignments:
            self.get_color(region_num)

    def get_color(self, region_num: int, parent_region: int = None) -> str:
        """
        Get color for a region.

        If parent_region is specified, generates a shade of the parent's color.
        """
        if region_num in self._region_colors:
            return self._region_colors[region_num]

        if parent_region is not None and parent_region in self._region_colors:
            # Generate a shade of the parent color
            parent_color = self._region_colors[parent_region]
            # Determine shade factor based on how many children exist
            shade_factor = 0.85 + (hash(region_num) % 30) / 100  # Slight variation
            color = adjust_lightness(parent_color, shade_factor)
        else:
            # Assign next color from palette
            color = self.palette[self._color_index % len(self.palette)]
            self._color_index += 1

        self._region_colors[region_num] = color
        return color

    def assign_subdivision_colors(self, parent_num: int, child_nums: List[int]):
        """
        Assign shaded colors to subdivided regions.

        All children will be shades of the parent's color.
        """
        if parent_num not in self._region_colors:
            # Parent doesn't have a color, assign one
            self.get_color(parent_num)

        parent_color = self._region_colors[parent_num]
        shades = generate_shades(parent_color, len(child_nums))

        for child_num, shade in zip(child_nums, shades):
            self._region_colors[child_num] = shade
            self._parent_colors[child_num] = parent_num

    def get_all_colors(self) -> Dict[int, str]:
        """Get all assigned colors."""
        return self._region_colors.copy()

    def remove_color(self, region_num: int):
        """Remove color assignment for a region."""
        if region_num in self._region_colors:
            del self._region_colors[region_num]
        if region_num in self._parent_colors:
            del self._parent_colors[region_num]

    @staticmethod
    def get_available_palettes() -> List[str]:
        """Get list of available palette names."""
        return list(ALL_PALETTES.keys())

    @staticmethod
    def get_palette_preview(palette_name: str, n_colors: int = 5) -> List[str]:
        """Get first n colors of a palette for preview."""
        palette = ALL_PALETTES.get(palette_name, PALETTE_MODERN)
        return palette[:n_colors]


# ==================== Global Instance ==================== #

# Global color manager instance
color_manager = ColorManager()


# ==================== Legacy Compatibility ==================== #


def get_region_colors() -> List[str]:
    """
    Get colors list for legacy compatibility.

    Returns the current palette colors.
    """
    return color_manager.palette.copy()


# Replace the old colors list
colors = PALETTE_MODERN
