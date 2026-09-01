"""Shared publication-quality matplotlib style for ProtBFF figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
      "grey": "#999999"}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5,
        "axes.titleweight": "bold", "axes.labelweight": "normal",
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.1, "axes.edgecolor": "#333333",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 1.1, "ytick.major.width": 1.1,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8, "grid.alpha": 0.7,
        "lines.linewidth": 2.2, "lines.markersize": 6.5,
    })


def panel_label(ax, letter, dx=-0.13, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=17, fontweight="bold",
            va="top", ha="left")
