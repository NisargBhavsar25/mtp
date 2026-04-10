"""Supplementary Figure: Infographic showing how compound-level Mat2Vec
embeddings are constructed from element embeddings via stoichiometry-weighted
averaging.  Uses real pretrained embedding values for Li6PS5Cl."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

from src.config import OUTPUTS_DIR, journal_savefig

OUT_DIR = str(OUTPUTS_DIR / "final_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Real embedding data (first 6 + last 2 dims from pretrained mat2vec) ──
# Values for d199 and d200 extracted from actual model
elements = {
    "Li": {"w": 6, "vec_head": [0.066, 0.160, -0.051, 0.126, -0.376, 0.598],
                    "vec_tail": [-0.058, 0.215]},
    "P":  {"w": 1, "vec_head": [-0.374, 0.081, 0.081, -0.040, -0.152, 0.073],
                    "vec_tail": [-0.079, -0.001]},
    "S":  {"w": 5, "vec_head": [0.173, -0.161, 0.151, 0.236, -0.035, 0.190],
                    "vec_tail": [0.180, 0.174]},
    "Cl": {"w": 1, "vec_head": [-0.096, -0.107, 0.135, 0.398, -0.170, 0.007],
                    "vec_tail": [-0.323, 0.002]},
}
result_head = [0.061, 0.010, 0.051, 0.176, -0.212, 0.355]
result_tail = [0.011, 0.166]
ndim_show = 6
ndim_total = 200

# ── Colours ───────────────────────────────────────────────────────────
el_colors = {"Li": "#e74c3c", "P": "#f39c12", "S": "#2ecc71", "Cl": "#3498db"}
result_color = "#8e44ad"
box_edge = "#333333"

# ── Layout ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(-0.5, 17.5)
ax.set_ylim(-1.8, 10.5)
ax.set_aspect("equal")
ax.axis("off")

# ── Title ─────────────────────────────────────────────────────────────
ax.text(8.5, 10.0, "Compound Embedding from Element Embeddings",
        fontsize=18, fontweight="bold", ha="center", va="center")
ax.text(8.5, 9.35, r"Example: Li$_6$PS$_5$Cl  (Argyrodite solid electrolyte)",
        fontsize=14, ha="center", va="center", style="italic", color="#555")

# ── Step 1: Parse formula ─────────────────────────────────────────────
box1 = FancyBboxPatch((0.3, 7.6), 3.8, 1.1, boxstyle="round,pad=0.15",
                       facecolor="#e8f4fd", edgecolor=box_edge, linewidth=1.5)
ax.add_patch(box1)
ax.text(2.2, 8.4, "Step 1: Parse Formula", fontsize=12, fontweight="bold",
        ha="center", va="center")
ax.text(2.2, 7.9, r"Li$_6$PS$_5$Cl $\rightarrow$ {Li:6, P:1, S:5, Cl:1}",
        fontsize=11, ha="center", va="center", family="monospace")

# Arrow down
ax.annotate("", xy=(2.2, 6.5), xytext=(2.2, 7.5),
            arrowprops=dict(arrowstyle="-|>", color=box_edge, lw=1.5))

# ── Step 2: Look up element embeddings ────────────────────────────────
box2 = FancyBboxPatch((0.3, 5.4), 3.8, 1.1, boxstyle="round,pad=0.15",
                       facecolor="#e8f4fd", edgecolor=box_edge, linewidth=1.5)
ax.add_patch(box2)
ax.text(2.2, 6.2, "Step 2: Look Up Embeddings", fontsize=12,
        fontweight="bold", ha="center", va="center")
ax.text(2.2, 5.7, "Mat2Vec pretrained (200-dim)", fontsize=11,
        ha="center", va="center", color="#555")

# Arrow right
ax.annotate("", xy=(5.2, 5.95), xytext=(4.2, 5.95),
            arrowprops=dict(arrowstyle="-|>", color=box_edge, lw=1.5))

# ── Element embedding vectors ─────────────────────────────────────────
vec_x = 5.5
cell_w = 1.2
cell_h = 0.65
y_positions = [8.2, 6.7, 5.2, 3.7]
dots_x = vec_x + ndim_show * cell_w + 0.15       # "..." position
tail_x = dots_x + 1.0                             # d199 start
tail_cell_w = 1.2

# Dimension labels on top
for j in range(ndim_show):
    x = vec_x + j * cell_w + (cell_w - 0.08) / 2
    ax.text(x, 8.9, f"d{j+1}", fontsize=10, ha="center", va="center", color="#555")
ax.text(dots_x + 0.3, 8.9, "...", fontsize=12, ha="center", va="center", color="#888")
ax.text(tail_x + (tail_cell_w - 0.08) / 2, 8.9, "d199", fontsize=10,
        ha="center", va="center", color="#555")
ax.text(tail_x + tail_cell_w + (tail_cell_w - 0.08) / 2, 8.9, "d200", fontsize=10,
        ha="center", va="center", color="#555")

for idx, (el, info) in enumerate(elements.items()):
    y = y_positions[idx]
    color = el_colors[el]

    # Light background band for the row
    band = FancyBboxPatch((vec_x - 0.15, y - cell_h / 2 - 0.05),
                           tail_x + 2 * tail_cell_w - vec_x + 0.25, cell_h + 0.1,
                           boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor="none", alpha=0.06)
    ax.add_patch(band)

    # Element label + weight
    ax.text(vec_x - 0.25, y + 0.12, f"{el}", fontsize=16, fontweight="bold",
            ha="right", va="center", color=color)
    ax.text(vec_x - 0.25, y - 0.22, f"w = {info['w']}", fontsize=10,
            ha="right", va="center", color="#666")

    # Head cells (d1–d6)
    for j, val in enumerate(info["vec_head"]):
        x = vec_x + j * cell_w
        rect = FancyBboxPatch((x, y - cell_h / 2), cell_w - 0.08, cell_h,
                               boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor="white",
                               alpha=0.22, linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + (cell_w - 0.08) / 2, y, f"{val:.3f}", fontsize=10,
                ha="center", va="center", fontweight="bold")

    # Dots
    ax.text(dots_x + 0.3, y, "...", fontsize=16, ha="center", va="center",
            color="#888", fontweight="bold")

    # Tail cells (d199, d200)
    for j, val in enumerate(info["vec_tail"]):
        x = tail_x + j * tail_cell_w
        rect = FancyBboxPatch((x, y - cell_h / 2), tail_cell_w - 0.08, cell_h,
                               boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor="white",
                               alpha=0.22, linewidth=0.5)
        ax.add_patch(rect)
        ax.text(x + (tail_cell_w - 0.08) / 2, y, f"{val:.3f}", fontsize=10,
                ha="center", va="center", fontweight="bold")

# ── Arrows from element rows to formula box ───────────────────────────
formula_y = 1.8
mid_vec = vec_x + 4  # middle of the vector area
for idx, (el, info) in enumerate(elements.items()):
    y = y_positions[idx]
    ax.annotate("", xy=(8.5, formula_y + 1.2), xytext=(mid_vec, y - 0.4),
                arrowprops=dict(arrowstyle="-", color=el_colors[el],
                                lw=1.0, alpha=0.3,
                                connectionstyle="arc3,rad=0.12"))

# ── Step 3: Weighted average ──────────────────────────────────────────
box3 = FancyBboxPatch((1.5, formula_y - 0.2), 14.0, 1.4,
                       boxstyle="round,pad=0.15",
                       facecolor="#fef9e7", edgecolor=box_edge, linewidth=1.5)
ax.add_patch(box3)

ax.text(8.5, formula_y + 0.95, "Step 3: Stoichiometry-Weighted Average",
        fontsize=12, fontweight="bold", ha="center", va="center")

formula_text = (
    r"$\mathbf{e}_{\mathrm{compound}} = "
    r"\frac{\sum_{i} w_i \cdot \mathbf{e}_i}{\sum_{i} w_i} = "
    r"\frac{6\,\mathbf{e}_{\mathrm{Li}} + 1\,\mathbf{e}_{\mathrm{P}}"
    r" + 5\,\mathbf{e}_{\mathrm{S}} + 1\,\mathbf{e}_{\mathrm{Cl}}}"
    r"{6+1+5+1}$"
)
ax.text(8.5, formula_y + 0.2, formula_text, fontsize=14,
        ha="center", va="center")

# Arrow down
ax.annotate("", xy=(8.5, -0.1), xytext=(8.5, formula_y - 0.3),
            arrowprops=dict(arrowstyle="-|>", color=box_edge, lw=1.5))

# ── Result vector ─────────────────────────────────────────────────────
res_y = -0.65
res_x = 3.5
cell_w_r = 1.2

# Background box spans head + dots + tail
res_total_w = ndim_show * cell_w_r + 1.0 + 2 * cell_w_r + 1.0
box_res = FancyBboxPatch((res_x - 2.3, res_y - 0.55),
                          res_total_w + 2.8, 1.1,
                          boxstyle="round,pad=0.15",
                          facecolor=result_color, edgecolor=box_edge,
                          alpha=0.10, linewidth=1.5)
ax.add_patch(box_res)

ax.text(res_x - 0.8, res_y, r"$\mathbf{e}_{\mathrm{Li_6PS_5Cl}}$ =",
        fontsize=14, fontweight="bold", ha="right", va="center",
        color=result_color)

# Head cells
for j, val in enumerate(result_head):
    x = res_x + j * cell_w_r
    rect = FancyBboxPatch((x, res_y - cell_h / 2), cell_w_r - 0.1, cell_h,
                           boxstyle="round,pad=0.05",
                           facecolor=result_color, edgecolor="white",
                           alpha=0.25, linewidth=0.5)
    ax.add_patch(rect)
    ax.text(x + (cell_w_r - 0.1) / 2, res_y, f"{val:.3f}", fontsize=11,
            ha="center", va="center", fontweight="bold", color=result_color)

# Dots
res_dots_x = res_x + ndim_show * cell_w_r + 0.15
ax.text(res_dots_x + 0.3, res_y, "...", fontsize=16, ha="center", va="center",
        color="#888", fontweight="bold")

# Tail cells (d199, d200)
res_tail_x = res_dots_x + 0.9
for j, val in enumerate(result_tail):
    x = res_tail_x + j * cell_w_r
    rect = FancyBboxPatch((x, res_y - cell_h / 2), cell_w_r - 0.1, cell_h,
                           boxstyle="round,pad=0.05",
                           facecolor=result_color, edgecolor="white",
                           alpha=0.25, linewidth=0.5)
    ax.add_patch(rect)
    ax.text(x + (cell_w_r - 0.1) / 2, res_y, f"{val:.3f}", fontsize=11,
            ha="center", va="center", fontweight="bold", color=result_color)

# Dimension annotation under result
ax.text(res_x + (ndim_show * cell_w_r + 1.0 + 2 * cell_w_r) / 2, res_y + 0.1 - 0.7,
        "200-dimensional compound embedding vector",
        fontsize=11, ha="center", va="center", color="#666", style="italic")

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "fig_supp_embedding_infographic.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")