#!/usr/bin/env python
"""Figure 2 (house style v2): A = original(no-alignment) -> MVA identity sweep;
B = original CD-HIT (leaky) sweep; C = performance vs dataset (leave-one-experimental-
method-out), single-axis. ESM-C + ProSST + ProtBFF, mean-of-folds +/- SEM."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_v2 import (apply_style, panel_label, save, ENCODER_COLOR, ENCODER_LABEL,
                         ENCODER_MARKER, MUTED, INK2)
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LEAKY_WASH = (0.85, 0.30, 0.26, 0.06)
ENCS = ["esmc"]                       # ESM-C only; a trained non-PLM competitor is overlaid
COMPETITOR = "#6b6a66"                 # RDE-Network (specialized non-PLM deep predictor)
RDE_LEAKY, RDE_MVA60, RDE_MVA60_SEM = 0.480, 0.393, 0.087 / (10 ** 0.5)   # SEM = SD/sqrt(10)


def _load(fn):
    p = os.path.join(OUT, fn)
    return json.load(open(p)) if os.path.exists(p) else {}


def _sweep(ax, kind, order, xlabels, title, highlight0=False):
    decay = _load("decay_encoders.json")
    for enc in ENCS:
        d = _load(f"thr_perf_{kind}_{enc}.json")
        col, mk = ENCODER_COLOR[enc], ENCODER_MARKER[enc]
        xs, ys, es = [], [], []
        for i, t in enumerate(order):
            if t == "100" and enc in decay:   # SHARED 'original / no-clustering' point (same in A & B)
                v = decay[enc]["nosplit"]
                ys.append(v["protbff_r"]); es.append(v.get("protbff_r_sem", 0)); xs.append(i)
            elif t in d:
                ys.append(d[t]["protbff_r"]); es.append(d[t].get("protbff_r_sem", 0)); xs.append(i)
        ax.errorbar(xs, ys, yerr=es, fmt="-" + mk, color=col, capsize=3,
                    label=f"{ENCODER_LABEL[enc]} + ProtBFF")
    # ESM-C bare (no ProtBFF) overlay -> shows the ProtBFF gain across the sweep
    bare = _load("bare_esmc_mva.json" if kind == "mva" else "bare_esmc_cdhit.json")
    bmva = _load("bare_esmc_mva.json")
    col = ENCODER_COLOR["esmc"]; xs, ys, es = [], [], []
    for i, t in enumerate(order):
        src = bmva if (t == "100" and "100" in bmva) else bare   # shared no-clustering original
        if t in src:
            ys.append(src[t]["bare_r"]); es.append(src[t].get("bare_r_sem", 0)); xs.append(i)
    ax.errorbar(xs, ys, yerr=es, fmt="--o", color=col, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=1.6, lw=1.7, capsize=3,
                label="ESM-C (bare)", zorder=4)
    if highlight0:
        ax.axvspan(-0.4, 0.4, color=LEAKY_WASH, zorder=0)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Pearson correlation (mean-of-folds)"); ax.set_ylim(0.20, 0.68)
    ax.set_title(title)


def _overlay_rde(ax, kind):
    """RDE-Network, a trained non-PLM specialized predictor, evaluated at its leaky benchmark
    and the honest MVA-60 split (the only two splits it was trained on). Shows the leakage
    inflation is not specific to the PLM: a different model family declines the same way."""
    if kind == "mva":                              # leaky original (x=0) -> MVA-60 (x=3)
        ax.errorbar([0, 3], [RDE_LEAKY, RDE_MVA60], yerr=[0, RDE_MVA60_SEM], fmt=":X",
                    color=COMPETITOR, capsize=3, markersize=10, lw=1.9,
                    label="RDE-Network (competitor)", zorder=5)
    else:                                          # CD-HIT: leaky benchmark at 60% (x=3)
        ax.errorbar([3], [RDE_LEAKY], fmt="X", color=COMPETITOR, markersize=10,
                    label="RDE-Network (competitor, leaky)", zorder=5)


def panelA(ax):
    _sweep(ax, "mva", ["100", "90", "80", "60", "50", "40", "30"],
           ["original\n(no align.)", "90", "80", "60", "50", "40", "30"],
           "Original → MVA identity sweep", highlight0=True)
    _overlay_rde(ax, "mva")
    ax.set_xlabel("MVA sequence-identity threshold (%)")
    ax.legend(loc="upper right", fontsize=10); panel_label(ax, "A")


def panelB(ax):
    _sweep(ax, "cdhit", ["100", "95", "80", "60", "40"],
           ["original\n(100)", "95", "80", "60", "40"], "CD-HIT (leaky) identity sweep")
    _overlay_rde(ax, "cdhit")
    ax.set_xlabel("CD-HIT sequence-identity threshold (%)")
    ax.legend(loc="upper right", fontsize=10); panel_label(ax, "B")


def panelC(ax):
    """Performance vs dataset: leave-one-experimental-method-out. Single y-axis (Pearson);
    error bars are the Fisher-z 95% CI of the correlation from that method's test-set size n;
    the % of dataset is shown under each method label (no second axis)."""
    d = _load("lomo_esmc.json")
    methods = sorted(d, key=lambda m: -d[m]["frac"])
    pears = [d[m]["pearson"] for m in methods]
    ns = [d[m]["n"] for m in methods]
    lo, hi = [], []
    for r, n in zip(pears, ns):                       # Fisher z-transform 95% CI on Pearson r
        z, se = np.arctanh(r), 1.0 / np.sqrt(max(n - 3, 1))
        lo.append(r - np.tanh(z - 1.96 * se)); hi.append(np.tanh(z + 1.96 * se) - r)
    x = np.arange(len(methods))
    ax.bar(x, pears, yerr=[lo, hi], color=ENCODER_COLOR["esmc"], alpha=0.9, width=0.66, zorder=3,
           error_kw=dict(ecolor=INK2, lw=1.3, capsize=3))
    for xi, p, h in zip(x, pears, hi):
        ax.text(xi, p + h + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({d[m]['frac']*100:.0f}%)" for m in methods], rotation=0, fontsize=9.5)
    ax.set_xlabel("Held-out experimental method  (% = fraction of dataset)")
    ax.set_ylabel("Pearson correlation"); ax.set_ylim(0, 1.0)
    ax.set_title("Performance vs dataset: leave-one-experimental-method-out (ESM-C + ProtBFF)")
    panel_label(ax, "C", dx=-0.06)


def main():
    apply_style(base=14)
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.38, wspace=0.26)
    panelA(fig.add_subplot(gs[0, 0]))
    panelB(fig.add_subplot(gs[0, 1]))
    panelC(fig.add_subplot(gs[1, :]))
    save(fig, "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/figure_2_final")


if __name__ == "__main__":
    main()
