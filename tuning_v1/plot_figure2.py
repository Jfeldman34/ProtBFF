#!/usr/bin/env python
"""Figure 2 (leakage analysis) recreated on the MVA all-versus-all split.

A: ProtBFF (dual) performance vs MVA sequence-identity threshold (30-90%), ESM-C & ProSST,
   Pearson (left) and Spearman (right); competitor MVA-60 points shown for reference.
B: structural superposition (model-independent) reused from the original figure.
C: leave-one-experimental-method-out (ESM-C + ProtBFF dual): % of dataset (bars) + Pearson (line).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_style, panel_label, CB
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
OVL = os.path.join(HERE, "..", "..", "protbff_overleaf")
# Original (homology-uncontrolled) split vs honest MVA-60, mean-of-folds Pearson.
# Honest = measured this work; original = the benchmark's leaky-split values.
SLOPE = {  # model: (original, honest, color, marker)
    "ESM-C + ProtBFF":  (None,  0.469, CB["red"],    "o"),
    "ProSST + ProtBFF": (0.514, 0.451, CB["orange"], "s"),
    "ProMIM":           (0.486, 0.399, CB["green"],  "^"),
    "RDE-Network":      (0.480, 0.393, CB["blue"],   "D"),
    "DDAffinity":       (0.485, 0.340, CB["purple"], "v"),
}


def panelA(ax):
    x0, x1 = 0, 1
    for name, (orig, hon, col, mk) in SLOPE.items():
        if orig is not None:
            ax.plot([x0, x1], [orig, hon], "-", color=col, marker=mk, markersize=8,
                    markerfacecolor=col, markeredgecolor=col, linewidth=2.2)
            ax.annotate(f"{name}  {orig:.3f}→{hon:.3f} ({hon - orig:+.3f})", (x1, hon),
                        textcoords="offset points", xytext=(12, 0), va="center",
                        fontsize=9.5, color=col, fontweight="bold")
        else:  # ESM-C: honest only (new encoder, no original-split number)
            ax.plot([x1], [hon], marker=mk, color=col, markersize=10, markerfacecolor=col)
            ax.annotate(f"{name}  {hon:.3f}", (x1, hon), textcoords="offset points", xytext=(10, 0),
                        va="center", fontsize=9.5, color=col, fontweight="bold")
    ax.set_xlim(-0.25, 2.5); ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Original\n(homology-uncontrolled)", "MVA-60\n(homology-aware)"])
    ax.set_ylabel("Pearson Correlation"); ax.set_ylim(0.30, 0.55)
    ax.set_title("Benchmark performance collapses under leakage-corrected evaluation")
    ax.grid(True, axis="y", alpha=0.5); ax.grid(False, axis="x")
    panel_label(ax, "A", dx=-0.08)


def panelC(ax):
    d = json.load(open(os.path.join(OUT, "lomo_esmc.json")))
    methods = sorted(d, key=lambda m: -d[m]["frac"])
    fracs = [d[m]["frac"] * 100 for m in methods]
    pears = [d[m]["pearson"] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, fracs, color=CB["sky"], alpha=0.85, width=0.62, label="% of Dataset")
    for xi, f in zip(x, fracs):
        ax.text(xi, f + 0.6, f"{f:.0f}%", ha="center", va="bottom", fontsize=8.5, color="#2b6a8f")
    ax.set_ylabel("Percentage of Dataset", color="#2b6a8f")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=0)
    ax.set_xlabel("Experimental Method"); ax.set_ylim(0, max(fracs) * 1.18)
    ax.set_title("Leave-One-Experimental-Method-Out (ESM-C + ProtBFF)")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, pears, "-o", color=CB["red"], label="Pearson's r",
             markerfacecolor="white", markeredgecolor=CB["red"], markeredgewidth=1.8)
    ax2.set_ylabel("Pearson Correlation", color=CB["red"]); ax2.set_ylim(0, 1)
    ax.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    panel_label(ax, "C")


def panelB(ax):
    """Reuse the model-independent structural superposition strip from the original figure."""
    src = os.path.join(OVL, "figure_2_protbff.png")
    img = mpimg.imread(src)
    h, w = img.shape[0], img.shape[1]
    crop = img[int(0.355 * h):int(0.62 * h), int(0.06 * w):]   # structures only (exclude orig 'B' label)
    ax.imshow(crop); ax.axis("off")
    panel_label(ax, "B", dx=-0.03, dy=1.02)


def main():
    apply_style()
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 1.0], hspace=0.42)
    panelA(fig.add_subplot(gs[0, 0]))
    panelB(fig.add_subplot(gs[1, 0]))
    panelC(fig.add_subplot(gs[2, 0]))
    base = os.path.join(OVL, "figure_2_protbff_new")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    print(f"saved {base}.png / .pdf")


if __name__ == "__main__":
    main()
