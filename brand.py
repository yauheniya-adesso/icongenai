"""
Adesso brand colors — single source of truth for all visualisations.

Usage:
    from brand import blue, orange, purple  # named constants
    from brand import BRAND                 # dict for loops / matplotlib

In notebooks (which run from notebooks/):
    import sys; sys.path.insert(0, "..")
    from brand import BRAND, blue, purple
"""

blue   = "#006EC7"
purple = "#461EBE"
orange = "#FF9868"
green  = "#76C800"
pink   = "#F566BA"
teal   = "#28DCAA"
brown  = "#887D75"
white  = "#FFFFFF"
black  = "#000000"

# Dark canvas used in matplotlib charts
dark_bg = "#0D1B2A"

BRAND = {
    "blue":    blue,
    "purple":  purple,
    "orange":  orange,
    "green":   green,
    "pink":    pink,
    "teal":    teal,
    "brown":   brown,
    "white":   white,
    "black":   black,
    "dark_bg": dark_bg,
}
