#!/usr/bin/env python
"""Figure 3 (house style v2): ESM2 size sweep + SARS-CoV-2 DMS few-shot fine-tuning.
Shaded ProtBFF-gain bands, consistent encoder colours, quantitative lift + mean panels."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle_v2 import (apply_style, panel_label, save, ENCODER_COLOR, ENCODER_LABEL,
                         ENCODER_MARKER, MUTED)
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "out", "dms_ft")
ENCS = ["prosst", "esmc"]
DMS = [("ace2", "RBD – ACE2"), ("9lyp", "RBD – REGN10987"), ("7kmg", "RBD – LY-CoV555")]
SIZES = ["150M", "650M", "3B", "15B"]
ESM2_PROTBFF = [0.413, 0.410, 0.418, 0.446]
ESM2_BARE = [0.165, 0.204, 0.105, 0.163]


def series(cx, enc, model):
    p = os.path.join(FT, f"{cx}_{enc}", "all_results.json")
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p))[model]
    fr = sorted(r.keys(), key=lambda x: int(str(x).replace("pct", "")))
    x = [int(k.replace("pct", "")) for k in fr]
    y = [(r[k].get("metrics", r[k])).get("spearman_r") for k in fr]
    keep = [(xi, yi) for xi, yi in zip(x, y) if xi >= 0]   # include 0% (zero-shot)
    return [a for a, _ in keep], [b for _, b in keep]


def dms_panel(ax, cx, title, letter, legend=False):
    for enc in ENCS:
        col = ENCODER_COLOR[enc]; mk = ENCODER_MARKER[enc]
        xc, yc = series(cx, enc, "cross_attention")
        xs, ys = series(cx, enc, "simple")
        if xc is None:
            continue
        ax.fill_between(xc, ys, yc, color=col, alpha=0.13, lw=0, zorder=1)
        ax.plot(xc, yc, "-", marker=mk, color=col, label=f"{ENCODER_LABEL[enc]} + ProtBFF", zorder=3)
        ax.plot(xs, ys, "--", marker=mk, color=col, markerfacecolor="white",
                markeredgecolor=col, lw=1.7, label=f"{ENCODER_LABEL[enc]} (bare)", zorder=3)
    ax.set_title(title); ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Spearman correlation"); ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_ylim(-0.02, 0.85)
    if legend:
        ax.legend(loc="upper left", fontsize=10)
    panel_label(ax, letter)


def esm2_panel(ax):
    col = ENCODER_COLOR["esm2"]; x = list(range(len(SIZES)))
    ax.fill_between(x, ESM2_BARE, ESM2_PROTBFF, color=col, alpha=0.13, lw=0)
    ax.plot(x, ESM2_PROTBFF, "-^", color=col, label="ESM2 + ProtBFF")
    ax.plot(x, ESM2_BARE, "--D", color=col, markerfacecolor="white", markeredgecolor=col,
            lw=1.7, label="ESM2 (bare)")
    ax.set_xticks(x); ax.set_xticklabels(SIZES)
    ax.set_xlabel("ESM2 model size"); ax.set_ylabel("Spearman correlation")
    ax.set_title("ESM2 size sweep (SKEMPI2)"); ax.set_ylim(0.0, 0.6)
    ax.legend(loc="center right", fontsize=10); panel_label(ax, "A")


def mean_panel(ax):
    for enc in ENCS:
        col = ENCODER_COLOR[enc]; mk = ENCODER_MARKER[enc]
        for model, ls, fill, tag in [("cross_attention", "-", col, "+ ProtBFF"),
                                      ("simple", "--", "white", "(bare)")]:
            ys, xg = [], None
            for cx, _ in DMS:
                x, y = series(cx, enc, model)
                if x:
                    ys.append(y); xg = x
            if not ys:
                continue
            ym = np.mean(np.array(ys), axis=0)
            ax.plot(xg, ym, ls, marker=mk, color=col, lw=2.2 if ls == "-" else 1.7,
                    markerfacecolor=fill, markeredgecolor=col, label=f"{ENCODER_LABEL[enc]} {tag}")
    ax.set_title("Mean over three binding partners"); ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Spearman correlation"); ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_ylim(-0.02, 0.7); ax.legend(loc="upper left", fontsize=10)
    panel_label(ax, "C")


def summary_panel(ax):
    labels = ["ACE2", "REGN10987", "LY-CoV555"]; x = np.arange(len(DMS)); w = 0.36
    for i, enc in enumerate(ENCS):
        col = ENCODER_COLOR[enc]
        lifts = []
        for cx, _ in DMS:
            xc, yc = series(cx, enc, "cross_attention"); xs, ys = series(cx, enc, "simple")
            fs = [c - s for xx, c, s in zip(xc, yc, ys) if xx >= 10]   # few-shot lift (exclude zero-shot)
            lifts.append(np.mean(fs) if fs else 0)
        bars = ax.bar(x + (i - 0.5) * w, lifts, w, color=col, label=ENCODER_LABEL[enc], zorder=3)
        for xi, v in zip(x + (i - 0.5) * w, lifts):
            ax.text(xi, v + 0.004, f"+{v:.02f}", ha="center", va="bottom", fontsize=9, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("mean ProtBFF lift (ΔSpearman)")
    ax.set_title("ProtBFF lift across binding partners"); ax.set_ylim(0, 0.20)
    ax.legend(loc="upper right", fontsize=10); panel_label(ax, "D")


def main():
    apply_style(base=15)
    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)
    esm2_panel(fig.add_subplot(gs[0, 0]))                                  # A
    dms_panel(fig.add_subplot(gs[0, 1]), "ace2", "RBD – ACE2", "B", legend=True)  # B
    mean_panel(fig.add_subplot(gs[1, 0]))                                  # C (mean over antibodies)
    summary_panel(fig.add_subplot(gs[1, 1]))                               # D (lift across antibodies)
    save(fig, "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/figure_3_v3")


if __name__ == "__main__":
    main()
