#!/usr/bin/env python
"""Supplementary Figure S1: Pearson-correlation counterpart of the DMS few-shot analysis
in Figure 3. Three binding partners (ACE2, REGN10987, LY-CoV555), ProSST + ESM-C, with and
without ProtBFF. ESM-C (sequence-only) covers LY-CoV555 (7KMG), which ESM2 could not fit."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_v2 import (apply_style, panel_label, save, ENCODER_COLOR, ENCODER_LABEL,
                         ENCODER_MARKER)
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "out", "dms_ft")
ENCS = ["prosst", "esmc"]
DMS = [("ace2", "RBD – ACE2"), ("9lyp", "RBD – REGN10987"), ("7kmg", "RBD – LY-CoV555")]


def series(cx, enc, model):
    p = os.path.join(FT, f"{cx}_{enc}", "all_results.json")
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p))[model]
    fr = sorted(r.keys(), key=lambda x: int(str(x).replace("pct", "")))
    x = [int(k.replace("pct", "")) for k in fr]
    y = [(r[k].get("metrics", r[k])).get("pearson_r") for k in fr]   # PEARSON
    return x, y


def dms_panel(ax, cx, title, letter, legend=False):
    for enc in ENCS:
        col, mk = ENCODER_COLOR[enc], ENCODER_MARKER[enc]
        xc, yc = series(cx, enc, "cross_attention")
        xs, ys = series(cx, enc, "simple")
        if xc is None:
            continue
        ax.fill_between(xc, ys, yc, color=col, alpha=0.13, lw=0, zorder=1)
        ax.plot(xc, yc, "-", marker=mk, color=col, label=f"{ENCODER_LABEL[enc]} + ProtBFF", zorder=3)
        ax.plot(xs, ys, "--", marker=mk, color=col, markerfacecolor="white",
                markeredgecolor=col, lw=1.7, label=f"{ENCODER_LABEL[enc]} (bare)", zorder=3)
    ax.set_title(title); ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Pearson correlation"); ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_ylim(-0.02, 1.0)
    if legend:
        ax.legend(loc="upper left", fontsize=10)
    panel_label(ax, letter)


def main():
    apply_style(base=14)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (cx, title), letter, leg in zip(axes, DMS, "ABC", [True, False, False]):
        dms_panel(ax, cx, title, letter, legend=leg)
    fig.subplots_adjust(wspace=0.26, bottom=0.14, top=0.9)
    save(fig, "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/si_s1_pearson")


if __name__ == "__main__":
    main()
