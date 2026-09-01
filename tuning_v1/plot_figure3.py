#!/usr/bin/env python
"""Figure 3: ESM2 size sweep (A) + SARS-CoV-2 DMS few-shot with the DUAL readout (B-D).

Panels B-D read tuning_v1/out/dms/*.json (protbff = dual ProtBFF, bare = ridge[D|S]).
Panel A keeps the original ESM2 model-size sweep on SKEMPI (leaky split, caveated in text).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_style, panel_label, CB
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "dms")
OVL = os.path.join(HERE, "..", "..", "protbff_overleaf")
PRO, ESM = CB["red"], CB["blue"]

# Panel A: original ESM2 size sweep (Spearman on SKEMPI2), leaky split
SIZES = ["150M", "650M", "3B", "15B"]
ESM2_PROTBFF_A = [0.413, 0.410, 0.418, 0.446]
ESM2_BARE_A = [0.165, 0.204, 0.105, 0.163]


def load(tag):
    p = os.path.join(OUT, f"{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    c = d["curves"]
    fr = sorted(int(k) for k in c)
    return dict(fr=fr, pb=[c[str(k)]["protbff"] for k in fr], pbsd=[c[str(k)]["protbff_sd"] for k in fr],
                ba=[c[str(k)]["bare"] for k in fr], basd=[c[str(k)]["bare_sd"] for k in fr])


def curve(ax, x, y, sd, color, ls, marker, label):
    y, sd = np.array(y), np.array(sd)
    ax.plot(x, y, ls=ls, marker=marker, color=color, label=label,
            markerfacecolor=color if ls == "-" else "white", markeredgecolor=color, markeredgewidth=1.6)
    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, linewidth=0)


def dms_panel(ax, prosst_tag, esm2_tag, title, letter, with_esm2=True):
    pr = load(prosst_tag)
    if pr:
        curve(ax, pr["fr"], pr["pb"], pr["pbsd"], PRO, "-", "o", "ProSST + ProtBFF")
        curve(ax, pr["fr"], pr["ba"], pr["basd"], PRO, "--", "s", "ProSST")
    if with_esm2:
        es = load(esm2_tag)
        if es:
            curve(ax, es["fr"], es["pb"], es["pbsd"], ESM, "-", "^", "ESM2 + ProtBFF")
            curve(ax, es["fr"], es["ba"], es["basd"], ESM, "--", "D", "ESM2")
    ax.set_title(title); ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel("Spearman Correlation")
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax.set_ylim(-0.05, 0.85); ax.legend(loc="lower right", fontsize=9.5)
    panel_label(ax, letter)


def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # Panel A: size sweep
    ax = axes[0, 0]; x = range(len(SIZES))
    ax.plot(x, ESM2_PROTBFF_A, "-^", color=ESM, label="ESM2 + ProtBFF",
            markerfacecolor=ESM, markeredgecolor=ESM)
    ax.plot(x, ESM2_BARE_A, "--D", color=ESM, label="ESM2",
            markerfacecolor="white", markeredgecolor=ESM, markeredgewidth=1.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(SIZES)
    ax.set_xlabel("ESM2 Model Size (Parameters)"); ax.set_ylabel("Spearman Correlation")
    ax.set_title("ESM2 Model Size vs Performance on SKEMPI2")
    ax.set_ylim(-0.05, 0.6); ax.legend(loc="center right", fontsize=9.5)
    panel_label(ax, "A")

    dms_panel(axes[0, 1], "ace2_prosst", "ace2_esm2", "SARS-CoV-2 RBD with ACE2", "B")
    dms_panel(axes[1, 0], "9lyp_prosst", "9lyp_esm2", "SARS-CoV-2 RBD with REGN10987", "C")
    dms_panel(axes[1, 1], "7kmg_prosst", None, "SARS-CoV-2 RBD with LY-CoV555", "D", with_esm2=False)

    fig.tight_layout(w_pad=3.0, h_pad=3.5)
    base = os.path.join(OVL, "figure_3_protbff_new")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    print(f"saved {base}.png / .pdf")


if __name__ == "__main__":
    main()
