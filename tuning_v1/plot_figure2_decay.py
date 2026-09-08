#!/usr/bin/env python
"""Figure 2 (decay + LOMO). A: every ProtBFF encoder decays from the no-alignment random
split to the honest MVA-60 split (mean-of-folds Pearson +/- SEM, one consistent pipeline).
B: leave-one-experimental-method-out (ESM-C + ProtBFF)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_style, panel_label, CB
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
# encoder -> (label, colour, marker, label_y_offset to de-collide MVA-60 endpoints)
ENC = [
    ("saprot", "SaProt",  "#0072B2", "D",  0.012),
    ("esmc",   "ESM-C",   "#009E73", "s",  -0.006),
    ("prosst", "ProSST",  "#D55E00", "o",  0.000),
    ("esm2",   "ESM2",    "#CC79A7", "^",  0.010),
    ("esm3",   "ESM3",    "#E69F00", "v",  -0.012),
]


def panelA(ax):
    d = json.load(open(os.path.join(OUT, "decay_encoders.json")))
    x0, x1 = 0.0, 1.0
    for enc, lab, col, mk, off in ENC:
        if enc not in d:
            continue
        ns, mv = d[enc]["nosplit"], d[enc]["mva60"]
        lk, lke = ns["protbff_r"], ns.get("protbff_r_sem", 0)
        hn, hne = mv["protbff_r"], mv.get("protbff_r_sem", 0)
        ax.plot([x0, x1], [lk, hn], "-", color=col, lw=2.2, alpha=0.9, zorder=2)
        ax.errorbar([x0], [lk], yerr=[lke], fmt=mk, color=col, markersize=8, capsize=3,
                    markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.8, zorder=4)
        ax.errorbar([x1], [hn], yerr=[hne], fmt=mk, color=col, markersize=8, capsize=3, zorder=4)
        ax.text(x1 + 0.04, hn + off, f"{lab}  ({hn - lk:+.2f})", ha="left", va="center",
                fontsize=9.5, color=col, fontweight="bold")
    ax.set_xlim(-0.18, 1.75); ax.set_ylim(0.35, 0.68)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No sequence alignment\n(random split)", "MVA-60\n(our method)"], fontsize=10.5)
    ax.set_ylabel("Pearson correlation (mean-of-folds)")
    ax.set_title("All encoders decay to MVA-60", pad=10)
    panel_label(ax, "A", dx=-0.13, dy=1.05)


def panelB(ax):
    d = json.load(open(os.path.join(OUT, "lomo_esmc.json")))
    methods = sorted(d, key=lambda m: -d[m]["frac"])
    fracs = [d[m]["frac"] * 100 for m in methods]
    pears = [d[m]["pearson"] for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, fracs, color=CB["sky"], alpha=0.85, width=0.62)
    for xi, f in zip(x, fracs):
        ax.text(xi, f + 0.6, f"{f:.0f}%", ha="center", va="bottom", fontsize=8, color="#2b6a8f")
    ax.set_ylabel("Percentage of dataset", color="#2b6a8f")
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=0, fontsize=9)
    ax.set_xlabel("Experimental method"); ax.set_ylim(0, max(fracs) * 1.18)
    ax.set_title("Leave-One-Experimental-Method-Out (ESM-C + ProtBFF)")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, pears, "-o", color=CB["red"], markerfacecolor="white",
             markeredgecolor=CB["red"], markeredgewidth=1.8)
    ax2.set_ylabel("Pearson correlation", color=CB["red"]); ax2.set_ylim(0, 1)
    ax.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    panel_label(ax, "B")


def main():
    apply_style()
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.30)
    panelA(fig.add_subplot(gs[0, 0]))
    panelB(fig.add_subplot(gs[0, 1]))
    base = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/figure_2_decay_lomo"
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    print("saved", base)


if __name__ == "__main__":
    main()
