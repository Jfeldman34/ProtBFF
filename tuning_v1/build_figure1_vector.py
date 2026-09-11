#!/usr/bin/env python
"""Rebuild Figure 1 (ProtBFF schematic). Top half = TRUE VECTOR (matplotlib patches:
boxes, blocks, arrows, trapezoid, text) with the MSE Loss label enlarged and centered
under the DDG output. Bottom half (photographic protein-structure panels) = high-res
raster crop of the original figure embedded in the layout. Output: PDF (vector) + PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon, Circle
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import numpy as np, os

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "Arial", "DejaVu Sans"]

SP = os.path.dirname(os.path.abspath(__file__))
BOTTOM = "/tmp/claude-61673/-n-netscratch-shakhnovich-lab-Lab-jwang-ProtBFF/c1d47fff-9805-412f-8cd4-2f4026b992cd/scratchpad/fig1_bottom.png"

# palette (sampled from the original)
BG_F, BG_E = "#C6D0D9", "#7E96A8"      # blue-gray boxes
Y_F, Y_E   = "#FCEDC6", "#E0BE55"      # yellow encoder / trapezoid
BLK_F, BLK_E = "#EAEAEA", "#BEBEBE"    # embedding blocks
ARR  = "#4E7A8F"                        # thin arrows / connectors
FAT  = "#2F5D7E"                        # fat scaling arrow
INK  = "#1a1a1a"

W, H, TOP = 2250, 1636, 580
fig = plt.figure(figsize=(11.25, 8.18))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0)
ax.set_aspect("equal"); ax.axis("off")

# ---- bottom raster panels ----
img = mpimg.imread(BOTTOM)
ax.imshow(img, extent=[0, W, H, TOP], zorder=0, interpolation="lanczos")

def rbox(x, y, w, h, fc, ec, r=0.02, lw=2, z=2, alpha=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha,
                 boxstyle=f"round,pad=0,rounding_size={min(w,h)*0.28}", mutation_aspect=1))

def blk(x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=BLK_F, ec=BLK_E, lw=1.6, zorder=4,
                 boxstyle=f"round,pad=0,rounding_size={w*0.30}"))

def arrow(x1, y1, x2, y2, c=ARR, lw=2.4, z=5, ms=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                 lw=lw, color=c, zorder=z, shrinkA=0, shrinkB=0))

def line(x1, y1, x2, y2, c=ARR, lw=2.4, z=5):
    ax.add_line(Line2D([x1, x2], [y1, y2], color=c, lw=lw, zorder=z, solid_capstyle="round"))

def dots(cx, cy):
    for dy in (-13, 0, 13):
        ax.add_patch(Circle((cx, cy + dy), 3.4, fc="#5b5b5b", ec="none", zorder=6))

YC = 367  # main horizontal flow axis
DL = 9.5   # dimension-label fontsize

# ProtBFF container
rbox(745, 205, 690, 310, BG_F, BG_E, lw=2.4, z=1, alpha=0.55)
ax.text(1090, 233, "ProtBFF", ha="center", va="center", fontsize=17, fontweight="bold", color=INK, zorder=3)

# Biophysical features box (top) + Encoder box (yellow)
rbox(320, 96, 285, 80, BG_F, BG_E, lw=2.4, z=3)
ax.text(462, 136, "Biophysical\nfeatures", ha="center", va="center", fontsize=15, fontweight="bold", color=INK, zorder=4)
rbox(320, 292, 285, 148, Y_F, Y_E, lw=2.4, z=3)
ax.text(462, 366, "Encoder", ha="center", va="center", fontsize=17, fontweight="bold", color=INK, zorder=4)

# encoder outputs -> subtract
line(605, 330, 690, 348); line(605, 402, 690, 384)
ax.text(648, 312, "Wildtype", ha="center", va="bottom", fontsize=10, color=INK, zorder=4)
ax.text(648, 420, "Mutant type", ha="center", va="top", fontsize=10, color=INK, zorder=4)
ax.add_patch(Circle((712, 366), 24, fc="white", ec=INK, lw=2.0, zorder=5))
ax.text(712, 366, "−", ha="center", va="center", fontsize=20, color=INK, zorder=6)
arrow(736, 366, 786, 366)

# Original E_i block
blk(792, 262, 42, 210)
ax.text(813, 248, r"Original $\mathbf{E_i}$", ha="center", va="bottom", fontsize=11, color=INK, zorder=6)
ax.text(813, 490, "dim d", ha="center", va="top", fontsize=DL, color=INK, zorder=6)

# fat scaling arrow (biophysical scaling)
ax.add_patch(FancyArrowPatch((840, 366), (888, 366), arrowstyle="-|>", mutation_scale=22,
             lw=6, color=FAT, zorder=6, shrinkA=0, shrinkB=0))

# biophysical features feed into the scaled blocks (right, then down)
line(605, 136, 913, 136); arrow(913, 136, 913, 256, lw=2.2)

# stage blocks: 5xd, 5xd', 5xd'
def stage(x, label):
    blk(x, 262, 42, 84); blk(x, 388, 42, 84); dots(x + 21, 367)
    ax.text(x + 21, 490, label, ha="center", va="top", fontsize=DL, color=INK, zorder=6)

stage(892, "dim 5×d")
stage(1024, r"dim 5×d$'$")
stage(1170, r"dim 5×d$'$")

# arrows: stage1 -> stage2 (straight)
arrow(936, 304, 1022, 304); arrow(936, 430, 1022, 430)
# cross-attention X: stage2 -> stage3 (crossed)
arrow(1068, 304, 1168, 430); arrow(1068, 430, 1168, 304)

# stage3 -> Modified E_i (converge)
blk(1320, 262, 42, 210)
ax.text(1341, 248, r"Modified $\mathbf{E_i}$", ha="center", va="bottom", fontsize=11, color=INK, zorder=6)
ax.text(1341, 490, r"dim d$'$", ha="center", va="top", fontsize=DL, color=INK, zorder=6)
arrow(1214, 304, 1318, 344); arrow(1214, 430, 1318, 390)

# container -> trapezoid (MLP readout)
arrow(1435, YC, 1494, YC)
ax.add_patch(Polygon([(1500, 312), (1584, 336), (1584, 398), (1500, 422)],
             closed=True, fc=Y_F, ec=Y_E, lw=2.4, zorder=4))
# trapezoid -> DDG
arrow(1590, YC, 1772, YC)

# DDG output (large) + MSE Loss enlarged, centered under it
ax.text(1852, YC, r"$\mathbf{\Delta\Delta G}$", ha="center", va="center", fontsize=40, color=INK, zorder=6)
ax.text(1852, 470, "MSE Loss", ha="center", va="center", fontsize=21, fontweight="bold", color=INK, zorder=6)

out = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/figure_previews/figure_1_vector"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=200)
print("saved", out + ".pdf / .png")
