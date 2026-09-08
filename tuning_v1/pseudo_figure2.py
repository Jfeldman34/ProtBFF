#!/usr/bin/env python
"""PSEUDO Figure 2 (preview): A = original(no-alignment) -> MVA-threshold sweep;
B = original CD-HIT (leaky) sweep. ESM-C + ProSST only (other encoders computing)."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/tuning_v1")
from figstyle_v2 import apply_style, panel_label, save, ENCODER_COLOR, ENCODER_LABEL, ENCODER_MARKER, MUTED
import matplotlib.pyplot as plt

OUT = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/tuning_v1/out"


def load(fn):
    p = os.path.join(OUT, fn)
    return json.load(open(p)) if os.path.exists(p) else {}


def panelA(ax):
    # MVA sweep: original(100=no clustering) then MVA thresholds 90..30
    order = ["100", "90", "80", "60", "50", "40", "30"]
    xlabels = ["original\n(no align.)", "90", "80", "60", "50", "40", "30"]
    xpos = list(range(len(order)))
    decay = load("decay_encoders.json")
    for enc in ["esmc", "prosst"]:
        d = load(f"thr_perf_mva_{enc}.json")
        col, mk = ENCODER_COLOR[enc], ENCODER_MARKER[enc]
        ys, es, xs = [], [], []
        for i, t in enumerate(order):
            if t in d:
                ys.append(d[t]["protbff_r"]); es.append(d[t].get("protbff_r_sem", 0)); xs.append(i)
            elif t == "100" and enc in decay:          # original from decay run
                ys.append(decay[enc]["nosplit"]["protbff_r"]); es.append(decay[enc]["nosplit"].get("protbff_r_sem", 0)); xs.append(i)
        ax.errorbar(xs, ys, yerr=es, fmt="-" + mk, color=col, capsize=3,
                    label=f"{ENCODER_LABEL[enc]} + ProtBFF")
    ax.axvspan(-0.4, 0.4, color=LEAKY_WASH, zorder=0)   # highlight the leaky 'original'
    ax.set_xticks(xpos); ax.set_xticklabels(xlabels)
    ax.set_xlabel("MVA sequence-identity threshold (%)")
    ax.set_ylabel("Pearson correlation (mean-of-folds)")
    ax.set_ylim(0.3, 0.68); ax.set_title("Original → MVA identity sweep")
    ax.legend(loc="upper right", fontsize=10); panel_label(ax, "A")


def panelB(ax):
    order = ["100", "95", "80", "60", "40"]
    xpos = list(range(len(order)))
    for enc in ["esmc", "prosst"]:
        d = load(f"thr_perf_cdhit_{enc}.json")
        col, mk = ENCODER_COLOR[enc], ENCODER_MARKER[enc]
        ys, es, xs = [], [], []
        for i, t in enumerate(order):
            if t in d:
                ys.append(d[t]["protbff_r"]); es.append(d[t].get("protbff_r_sem", 0)); xs.append(i)
        ax.errorbar(xs, ys, yerr=es, fmt="-" + mk, color=col, capsize=3,
                    label=f"{ENCODER_LABEL[enc]} + ProtBFF")
    ax.set_xticks(xpos); ax.set_xticklabels(["original\n(100)", "95", "80", "60", "40"])
    ax.set_xlabel("CD-HIT sequence-identity threshold (%)")
    ax.set_ylabel("Pearson correlation (mean-of-folds)")
    ax.set_ylim(0.3, 0.68); ax.set_title("CD-HIT (leaky) identity sweep")
    ax.legend(loc="upper right", fontsize=10); panel_label(ax, "B")


LEAKY_WASH = (0.85, 0.30, 0.26, 0.06)


def main():
    apply_style(base=14)
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.26)
    panelA(fig.add_subplot(gs[0, 0]))
    panelB(fig.add_subplot(gs[0, 1]))
    fig.text(0.5, -0.03, "PSEUDO / PREVIEW — ESM-C + ProSST only; ESM2, ESM3, SaProt sweeps still computing",
             ha="center", fontsize=10, color=MUTED, style="italic")
    save(fig, "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/pseudo_figure2")


if __name__ == "__main__":
    main()
