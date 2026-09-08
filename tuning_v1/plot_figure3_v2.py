#!/usr/bin/env python
"""Figure 3 (redesigned): ESM2 size sweep + SARS-CoV-2 DMS few-shot fine-tuning.
Beautiful + informative: unified colours, shaded ProtBFF-gain bands, a quantitative
lift-summary panel, and a mean-across-antibodies panel."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_style, panel_label, CB
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(HERE, "out", "dms_ft")
OVL = os.path.join(HERE, "..", "..", "protbff_overleaf")
ENC = {"prosst": ("ProSST", CB["red"], "o"), "esmc": ("ESM-C", CB["green"], "s")}
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
    # keep 10..80 (drop 0% zero-shot, which is noisy on OOD)
    keep = [(xi, yi) for xi, yi in zip(x, y) if xi >= 10]
    return [a for a, _ in keep], [b for _, b in keep]


def dms_panel(ax, cx, title, letter, legend=False):
    for enc, (lab, col, mk) in ENC.items():
        xc, yc = series(cx, enc, "cross_attention")
        xs, ys = series(cx, enc, "simple")
        if xc is None:
            continue
        ax.fill_between(xc, ys, yc, color=col, alpha=0.12, zorder=1)          # ProtBFF gain band
        ax.plot(xc, yc, "-", marker=mk, color=col, markersize=6, lw=2.2,
                label=f"{lab} + ProtBFF", zorder=3)
        ax.plot(xs, ys, "--", marker=mk, color=col, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=1.4, markersize=6, lw=1.8,
                label=f"{lab} (bare)", zorder=3)
    ax.set_title(title); ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Spearman correlation"); ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax.set_ylim(-0.02, 0.85)
    if legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    panel_label(ax, letter)


def esm2_panel(ax):
    x = range(len(SIZES))
    ax.fill_between(list(x), ESM2_BARE, ESM2_PROTBFF, color=CB["blue"], alpha=0.12)
    ax.plot(x, ESM2_PROTBFF, "-^", color=CB["blue"], markersize=7, lw=2.2, label="ESM2 + ProtBFF")
    ax.plot(x, ESM2_BARE, "--D", color=CB["blue"], markerfacecolor="white",
            markeredgecolor=CB["blue"], markeredgewidth=1.4, markersize=6, lw=1.8, label="ESM2 (bare)")
    ax.set_xticks(list(x)); ax.set_xticklabels(SIZES)
    ax.set_xlabel("ESM2 model size"); ax.set_ylabel("Spearman correlation")
    ax.set_title("ESM2 size sweep (SKEMPI2)"); ax.set_ylim(0.0, 0.6)
    ax.legend(loc="center right", fontsize=8.5); panel_label(ax, "A")


def mean_panel(ax):
    """Mean over the three antibody datasets."""
    for enc, (lab, col, mk) in ENC.items():
        for model, ls, fill, lab2 in [("cross_attention", "-", col, f"{lab} + ProtBFF"),
                                       ("simple", "--", "white", f"{lab} (bare)")]:
            ys = []
            for cx, _ in DMS:
                x, y = series(cx, enc, model)
                if x:
                    ys.append(y)
            if not ys:
                continue
            xg = series(DMS[0][0], enc, model)[0]
            ym = np.mean(np.array(ys), axis=0)
            ax.plot(xg, ym, ls, marker=mk, color=col, markersize=6, lw=2.2 if ls == "-" else 1.8,
                    markerfacecolor=fill, markeredgecolor=col, markeredgewidth=1.4, label=lab2)
    ax.set_title("Mean over three antibodies"); ax.set_xlabel("Training set size (%)")
    ax.set_ylabel("Spearman correlation"); ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax.set_ylim(-0.02, 0.7); ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "F")


def summary_panel(ax):
    """Mean ProtBFF lift (cross_attention - simple, averaged over 10..80%) per dataset x encoder."""
    labels = [t for _, t in DMS]
    x = np.arange(len(DMS)); w = 0.36
    for i, (enc, (lab, col, mk)) in enumerate(ENC.items()):
        lifts = []
        for cx, _ in DMS:
            _, yc = series(cx, enc, "cross_attention")
            _, ys = series(cx, enc, "simple")
            lifts.append(np.mean(np.array(yc) - np.array(ys)) if yc else 0)
        ax.bar(x + (i - 0.5) * w, lifts, w, color=col, alpha=0.85, label=lab)
        for xi, v in zip(x + (i - 0.5) * w, lifts):
            ax.text(xi, v + 0.003, f"+{v:.02f}", ha="center", va="bottom", fontsize=7.5, color=col)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("mean ProtBFF lift (ΔSpearman)")
    ax.set_title("ProtBFF lift across antibodies"); ax.set_ylim(0, 0.20)
    ax.legend(loc="upper right", fontsize=9); panel_label(ax, "E")


def main():
    apply_style()
    fig = plt.figure(figsize=(16.5, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)
    esm2_panel(fig.add_subplot(gs[0, 0]))
    dms_panel(fig.add_subplot(gs[0, 1]), "ace2", "RBD – ACE2", "B", legend=True)
    dms_panel(fig.add_subplot(gs[0, 2]), "9lyp", "RBD – REGN10987", "C")
    dms_panel(fig.add_subplot(gs[1, 0]), "7kmg", "RBD – LY-CoV555", "D")
    summary_panel(fig.add_subplot(gs[1, 1]))
    mean_panel(fig.add_subplot(gs[1, 2]))
    base = os.path.join("/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews", "figure_3_v2")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200 if ext == "png" else None)
    print("saved", base)


if __name__ == "__main__":
    main()
