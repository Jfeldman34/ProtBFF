#!/usr/bin/env python
"""Figure 3 (original fine-tuning protocol): ESM2 size sweep (A) + SARS-CoV-2 DMS few-shot (B-D).

Reads tuning_v1/out/dms_ft/{cx}_{enc}/all_results.json (keys cross_attention / simple,
per-pct metrics). Cross-Attention = +ProtBFF (solid), Simple = bare encoder (dashed).
Spearman vs training-set size. Panel A keeps the ESM2 size sweep (leaky split, caveated).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_style, panel_label, CB
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "out", "dms_ft")
OVL = os.path.join(HERE, "..", "..", "protbff_overleaf")
COL = {"prosst": CB["red"], "esmc": CB["green"]}
LAB = {"prosst": "ProSST", "esmc": "ESM-C"}
MK = {"prosst": "o", "esmc": "s"}
SIZES = ["150M", "650M", "3B", "15B"]
ESM2_PROTBFF_A = [0.413, 0.410, 0.418, 0.446]
ESM2_BARE_A = [0.165, 0.204, 0.105, 0.163]


def curve(enc):
    p = os.path.join(FT, f"{enc}", "all_results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    def series(model):
        d = r[model]; fr = sorted(d.keys(), key=lambda x: int(x.replace("pct", "")))
        x = [int(k.replace("pct", "")) for k in fr]
        y = [(d[k].get("metrics", d[k])).get("spearman_r") for k in fr]
        return x[1:], y[1:]   # drop 0% zero-shot
    return series("cross_attention"), series("simple")


def dms_panel(ax, cx, title, letter, encoders=("prosst", "esmc")):
    for enc in encoders:
        c = curve(f"{cx}_{enc}")
        if c is None:
            continue
        (xc, yc), (xs, ys) = c
        col = COL[enc]
        ax.plot(xc, yc, "-", marker=MK[enc], color=col, label=f"{LAB[enc]} + ProtBFF",
                markerfacecolor=col, markeredgecolor=col)
        ax.plot(xs, ys, "--", marker=MK[enc], color=col, label=f"{LAB[enc]}",
                markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.5)
    ax.set_title(title); ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel("Spearman Correlation"); ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax.set_ylim(-0.05, 0.85); ax.legend(loc="lower right", fontsize=9)
    panel_label(ax, letter)


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    ax = axes[0, 0]; x = range(len(SIZES))
    ax.plot(x, ESM2_PROTBFF_A, "-^", color=CB["blue"], label="ESM2 + ProtBFF", markerfacecolor=CB["blue"])
    ax.plot(x, ESM2_BARE_A, "--D", color=CB["blue"], label="ESM2", markerfacecolor="white",
            markeredgecolor=CB["blue"], markeredgewidth=1.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(SIZES)
    ax.set_xlabel("ESM2 Model Size (Parameters)"); ax.set_ylabel("Spearman Correlation")
    ax.set_title("ESM2 Model Size vs Performance on SKEMPI2"); ax.set_ylim(-0.05, 0.6)
    ax.legend(loc="center right", fontsize=9.5); panel_label(ax, "A")

    dms_panel(axes[0, 1], "ace2", "SARS-CoV-2 RBD with ACE2", "B")
    dms_panel(axes[1, 0], "9lyp", "SARS-CoV-2 RBD with REGN10987", "C")
    dms_panel(axes[1, 1], "7kmg", "SARS-CoV-2 RBD with LY-CoV555", "D")

    fig.tight_layout(w_pad=3.0, h_pad=3.5)
    base = os.path.join(OVL, "figure_3_protbff_new")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    print(f"saved {base}.png / .pdf")


if __name__ == "__main__":
    main()
