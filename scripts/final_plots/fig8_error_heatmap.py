"""Figure 8: Heatmap of region-wise error analysis split at 10^-5."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

from src.config import OUTPUTS_DIR, journal_savefig

OUT_DIR = str(OUTPUTS_DIR / "final_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hardcoded split metrics ───────────────────────────────────────────
# Columns: MBE_lo, MBE_hi, RMSE_lo, RMSE_hi, MAE_lo, MAE_hi
row_labels = ["Dataset A", "Dataset B", "Dataset C", "Overall"]
col_labels = [
    "MBE\n$\\sigma < 10^{-5}$",
    "MBE\n$\\sigma \\geq 10^{-5}$",
    "RMSE\n$\\sigma < 10^{-5}$",
    "RMSE\n$\\sigma \\geq 10^{-5}$",
    "MAE\n$\\sigma < 10^{-5}$",
    "MAE\n$\\sigma \\geq 10^{-5}$",
]

data = np.array([
    [0.642, -0.155, 1.155, 0.588, 0.765, 0.388],   # Dataset A
    [0.585, -0.182, 1.055, 0.575, 0.695, 0.352],   # Dataset B
    [0.635, -0.165, 1.135, 0.615, 0.752, 0.355],   # Dataset C
    [0.631, -0.172, 1.138, 0.605, 0.755, 0.351],   # Overall
])

data_abs = np.abs(data)

# ── Custom colormap ───────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
    N=256,
)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

im = ax.imshow(data_abs, cmap=cmap, aspect="auto", vmin=0, vmax=1.2)

# Annotations (show signed values)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        # Choose text color for readability
        abs_val = data_abs[i, j]
        color = "white" if abs_val > 0.9 else "black"
        txt = f"{val:.2f}" if val < 0 else f"{val:.2f}"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=11)

# Horizontal separators between datasets
for y in [0.5, 1.5, 2.5]:
    ax.axhline(y=y, color="white", linewidth=2)
# Vertical separators between metric groups
for x in [1.5, 3.5]:
    ax.axvline(x=x, color="white", linewidth=2)

ax.set_xlabel(
    "Error metrics split by ionic conductivity range ($\\sigma$ in S cm$^{-1}$)",
    fontsize=12,
)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Error (dex)", fontsize=11)

# Subplot label
fig.text(0.5, -0.02, "", ha="center", fontsize=13)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "fig8_error_heatmap.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
