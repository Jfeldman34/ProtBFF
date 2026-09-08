#!/usr/bin/env python
"""Publication figure style: figures4papers house look (spines off, clean sans, editable
vector text, dpi 300) + dataviz-skill chrome (recessive hairline grid, muted axis, ink
tokens) + a colorblind-safe categorical palette (dataviz reference hues). Color follows the
ENTITY (encoder), never rank; identity is reinforced with direct labels so it is never
color-alone."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- dataviz chrome / ink tokens (light surface) ---
SURFACE = "#ffffff"
INK = "#0b0b0b"          # primary text
INK2 = "#52514e"         # secondary text
MUTED = "#898781"        # axis tick labels
GRID = "#e1e0d9"         # hairline gridline (recessive)
AXIS = "#bfbeb7"         # baseline / spine

# --- colorblind-safe categorical hues (dataviz reference palette) ---
# one fixed hue per entity, assigned in importance order; never cycled.
ENCODER_COLOR = {
    "esmc":   "#2a78d6",   # blue   (SOTA / key)
    "prosst": "#eb6834",   # orange (paper original, structure-aware)
    "saprot": "#1baf7a",   # aqua
    "esm2":   "#e87ba4",   # magenta
    "esm3":   "#4a3aa7",   # violet
}
ENCODER_LABEL = {"esmc": "ESM-C", "prosst": "ProSST", "saprot": "SaProt",
                 "esm2": "ESM2", "esm3": "ESM3"}
ENCODER_MARKER = {"esmc": "o", "prosst": "s", "saprot": "D", "esm2": "^", "esm3": "v"}
# semantic roles (figures4papers intent: blue=proposed, red=contrast, neutral=baseline)
PROTBFF = "#2a78d6"
BARE = "#898781"
LEAKY = "#b64342"        # red = leaky / contrast
HONEST = "#1baf7a"       # aqua/green = honest / improvement


def apply_style(base=14):
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        # Nimbus Sans is a metric-compatible Helvetica clone available on the cluster;
        # gives a consistent, standard publication look. (Arial/Helvetica omitted: not
        # installed here, and listing them only spams findfont fallback warnings.)
        "font.family": ["Nimbus Sans", "DejaVu Sans", "sans-serif"],
        "font.size": base, "axes.titlesize": base + 2, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1, "legend.fontsize": base - 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.6,
        "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.size": 4, "ytick.major.size": 4, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1.0, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.2,
        "lines.linewidth": 2.2, "lines.markersize": 8, "lines.markeredgewidth": 1.6,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def panel_label(ax, letter, dx=-0.16, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=17, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, base, formats=("png", "pdf")):
    import os
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for ext in formats:
        fig.savefig(f"{base}.{ext}", dpi=300 if ext == "png" else None)
    print("saved", base + " (" + ",".join(formats) + ")")
