#!/usr/bin/env python
"""Assembled remade Figure 2: A sweep(+100% jump) | B leakage curve | C deflation
| D moderate-homology | E lysozyme leak schematic."""
import json, os, sys
import numpy as np
sys.path.insert(0, 'tuning_v1')
from figstyle import apply_style, panel_label, CB
import matplotlib.pyplot as plt

OUT = 'tuning_v1/out'
THR = [30, 40, 50, 60, 80, 90, 100]


def panelA(axP, axS):
    pf = json.load(open(f'{OUT}/thr_perf_mva_esmc.json'))
    bare = json.load(open(f'{OUT}/bare_esmc_mva.json'))
    xs = [t for t in THR if str(t) in pf and str(t) in bare]
    for key, ax, ylab, tit in [("r", axP, "Pearson Correlation", "Pearson"),
                               ("s", axS, "Spearman Correlation", "Spearman")]:
        ax.axvspan(95, 103, color=CB["red"], alpha=0.06, zorder=0)
        ax.axvline(95, color="0.6", ls=":", lw=1, zorder=1)
        yp = [pf[str(t)][f"protbff_{key}"] for t in xs]
        ype = [pf[str(t)].get(f"protbff_{key}_sem", 0) for t in xs]
        yb = [bare[str(t)][f"bare_{key}"] for t in xs]
        ybe = [bare[str(t)].get(f"bare_{key}_sem", 0) for t in xs]
        ax.errorbar(xs, yp, yerr=ype, fmt="-o", color=CB["red"], label="ESM-C + ProtBFF",
                    markersize=7, lw=2.2, capsize=3, zorder=5)
        ax.errorbar(xs, yb, yerr=ybe, fmt="--s", color=CB["grey"], label="ESM-C (bare, no ProtBFF)",
                    markerfacecolor="white", markeredgecolor=CB["grey"], markeredgewidth=1.6,
                    markersize=7, lw=2.2, capsize=3, zorder=5)
        ax.set_xlabel("MVA Sequence Identity Threshold (%)"); ax.set_ylabel(ylab)
        ax.set_xticks(xs); ax.set_xticklabels([str(t) if t != 100 else "100\n(off)" for t in xs], fontsize=9)
        ax.set_ylim(0.0, 0.7); ax.set_title(tit, pad=8)
        ax.text(99, 0.02, "clustering off:\nleakage returns", ha="center", va="bottom",
                fontsize=7, color=CB["red"])
    axP.legend(loc="lower left", fontsize=8.5)
    panel_label(axP, "A")


def panelB_leak(ax):
    cdx = [40, 60, 80, 95, 100]; cdy = [6.5, 14.6, 16.2, 32.9, 41.3]
    mvx = [30, 40, 50, 60, 80, 90]; mvy = [0] * 6
    ax.plot(cdx, cdy, '-o', color=CB['grey'], label='CD-HIT (leaky)', markersize=7, lw=2.2)
    ax.plot(mvx, mvy, '-o', color=CB['red'], label='MVA (honest)', markersize=7, lw=2.2)
    ax.set_xlabel("Sequence-identity threshold (%)")
    ax.set_ylabel("% test mutations with an\nidentical chain in training")
    ax.set_title("The leakage itself, measured"); ax.set_ylim(-2, 46)
    ax.legend(loc='upper left', fontsize=9); panel_label(ax, "B")


def panelC_deflate(ax):
    rows = [("ESM-C + ProtBFF", 0.515, 0.469, "pbf"), ("ProSST + ProtBFF", 0.514, 0.451, "pbf"),
            ("ProMIM", 0.486, 0.399, "comp"), ("RDE", 0.480, 0.393, "comp"),
            ("DDAffinity", 0.485, 0.340, "comp"), ("RDE-Linear", 0.369, 0.148, "comp"),
            ("FoldX (physics)", 0.359, 0.359, "phys")]
    rows = sorted(rows, key=lambda r: r[2])
    col = {"pbf": CB["red"], "comp": CB["grey"], "phys": CB["green"]}
    for i, (lab, lk, hn, k) in enumerate(rows):
        d = hn - lk
        if abs(d) < 0.005:
            ax.scatter([hn], [i], s=80, color=col[k], zorder=3)
            ax.text(hn + 0.015, i, "immune", ha='left', va='center', fontsize=7.5, color=col[k]); continue
        ax.plot([lk, hn], [i, i], '-', color=col[k], lw=2, alpha=0.5, zorder=1)
        ax.scatter([lk], [i], s=70, facecolor='white', edgecolor=col[k], linewidth=2, zorder=3)
        ax.scatter([hn], [i], s=80, color=col[k], zorder=3)
        ax.text(min(lk, hn) - 0.01, i, f"{d:+.2f}", ha='right', va='center', fontsize=7.5, color=col[k])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Pearson correlation"); ax.set_xlim(0.0, 0.62)
    ax.set_title("Leaky (CD-HIT-60) → honest (MVA-60)")
    ax.scatter([], [], s=70, facecolor='white', edgecolor='k', label='leaky')
    ax.scatter([], [], s=80, color='k', label='honest')
    ax.legend(loc='lower right', fontsize=8); panel_label(ax, "C")


def panelD_mod(ax):
    groups = ['Pearson', 'Spearman']; has = [0.107, 0.089]; no = [0.262, 0.270]
    x = np.arange(2); w = 0.36
    ax.bar(x - w / 2, has, w, color=CB['red'], alpha=0.85, label='has 30–89% similar\ncomplex in training')
    ax.bar(x + w / 2, no, w, color=CB['sky'], label='nothing similar\nin training')
    for xi, v in zip(x - w / 2, has): ax.text(xi, v + 0.008, f"{v:.3f}", ha='center', fontsize=8)
    for xi, v in zip(x + w / 2, no): ax.text(xi, v + 0.008, f"{v:.3f}", ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 0.34)
    ax.set_ylabel("correlation (MVA-90 test rows)")
    ax.set_title("Moderate homology gives no ΔΔG shortcut", pad=8)
    ax.legend(loc='upper center', fontsize=8, ncol=2); panel_label(ax, "D")


def panelE_lyso(ax):
    ax.set_xlim(0, 10.5); ax.set_ylim(-0.5, 2.9); ax.axis('off')
    lyso = ['1VFB', '3HFM', '1MLC', '1DQJ', '1XGU', '1XGT', '1XGP', '1XGQ']
    cd = {'1VFB': 1, '3HFM': 1, '1MLC': 8, '1DQJ': 9, '1XGU': 9, '1XGT': 9, '1XGP': 9, '1XGQ': 9}
    for y, (title, fmap, leak) in [(2.15, ("CD-HIT: scattered → LEAK", cd, True)),
                                   (0.65, ("MVA: one fold → no leak", {c: 1 for c in lyso}, False))]:
        ax.text(1.0, y + 0.6, title, fontsize=10, fontweight='bold', color=CB['red'] if leak else CB['green'])
        for k in range(1, 11):
            c = '#ffd9d9' if k == 1 else '#eef1f4'
            ax.add_patch(plt.Rectangle((k - 0.45, y - 0.05), 0.9, 0.5, facecolor=c, edgecolor='0.7', lw=0.8))
            ax.text(k, y - 0.25, f"f{k}", ha='center', va='center', fontsize=6.5, color='0.5')
        for comp in lyso:
            ax.scatter([fmap[comp]], [y + 0.2], s=26, color=CB['blue'], zorder=3)
        if leak:
            ax.annotate("", xy=(1, y + 0.2), xytext=(8.5, y + 0.2),
                        arrowprops=dict(arrowstyle='->', color=CB['red'], lw=1.5, connectionstyle="arc3,rad=-0.3"))
            ax.text(5, y + 0.78, "identical antigen in train & test", ha='center', fontsize=7.5, color=CB['red'])
    ax.text(1, -0.4, "pink = held-out test fold;  blue = the 8 identical-antigen complexes", fontsize=7.5, color='0.4')
    panel_label(ax, "E", dx=-0.02, dy=1.0)


def main():
    apply_style()
    fig = plt.figure(figsize=(14, 16.5))
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30, height_ratios=[1, 1, 0.85])
    panelA(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
    panelB_leak(fig.add_subplot(gs[1, 0]))
    panelC_deflate(fig.add_subplot(gs[1, 1]))
    panelD_mod(fig.add_subplot(gs[2, 0]))
    panelE_lyso(fig.add_subplot(gs[2, 1]))
    base = '/n/netscratch/shakhnovich_lab/Lab/jwang/protbff_overleaf/figure_2_protbff_new'
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', bbox_inches='tight')
    print("saved", base + '.{png,pdf}')


if __name__ == '__main__':
    main()
