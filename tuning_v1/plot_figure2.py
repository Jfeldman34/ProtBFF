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
THR = [30, 40, 50, 60, 80, 90, 100]   # 100% = no clustering (MVA effectively off -> leakage returns)
ENC = {"esmc": ("ESM-C", CB["red"], "o"), "prosst": ("ProSST", CB["blue"], "s")}


def panelA(axP, axS):
    # ESM-C + ProtBFF (dual) vs ESM-C bare ridge (no ProtBFF), mean-of-folds +/- SEM, vs MVA threshold.
    # 30-90% (homology controlled): flat -> honest, homology-independent.
    # 100% (no clustering, MVA off): near-identical complexes leak back -> both jump = the leakage.
    pf = json.load(open(os.path.join(OUT, "thr_perf_mva_esmc.json")))       # +ProtBFF
    bare = json.load(open(os.path.join(OUT, "bare_esmc_mva.json")))         # true unscaled bare ridge
    xs = [t for t in THR if str(t) in pf and str(t) in bare]
    for key, ax, ylab, tit in [("r", axP, "Pearson Correlation", "Pearson"),
                               ("s", axS, "Spearman Correlation", "Spearman")]:
        # shade the 100% (no-clustering / leakage) region
        if 100 in xs:
            ax.axvspan(95, 103, color=CB["red"], alpha=0.06, zorder=0)
            ax.axvline(95, color="0.6", ls=":", lw=1, zorder=1)
        yp = [pf[str(t)][f"protbff_{key}"] for t in xs]
        ype = [pf[str(t)].get(f"protbff_{key}_sem", 0) for t in xs]
        yb = [bare[str(t)][f"bare_{key}"] for t in xs]
        ybe = [bare[str(t)].get(f"bare_{key}_sem", 0) for t in xs]
        ax.errorbar(xs, yp, yerr=ype, fmt="-o", color=CB["red"], label="ESM-C + ProtBFF",
                    markerfacecolor=CB["red"], markersize=7, lw=2.2, capsize=3, zorder=5)
        ax.errorbar(xs, yb, yerr=ybe, fmt="--s", color=CB["grey"], label="ESM-C (bare, no ProtBFF)",
                    markerfacecolor="white", markeredgecolor=CB["grey"], markeredgewidth=1.6,
                    markersize=7, lw=2.2, capsize=3, zorder=5)
        ax.set_xlabel("MVA Sequence Identity Threshold (%)"); ax.set_ylabel(ylab)
        ax.set_xticks(xs); ax.set_xticklabels([str(t) if t != 100 else "100\n(off)" for t in xs], fontsize=9)
        ax.set_ylim(0.0, 0.7)
        ax.set_title(tit, pad=8)
    axP.legend(loc="lower left", fontsize=8.5)
    for ax in (axP, axS):
        ax.text(99, 0.02, "clustering off:\nleakage returns", ha="center", va="bottom",
                fontsize=7.5, color=CB["red"])
    panel_label(axP, "A")


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
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.85, 1.0], hspace=0.42, wspace=0.28)
    panelA(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
    panelB(fig.add_subplot(gs[1, :]))
    panelC(fig.add_subplot(gs[2, :]))
    base = os.path.join(OVL, "figure_2_protbff_new")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    print(f"saved {base}.png / .pdf")


if __name__ == "__main__":
    main()
