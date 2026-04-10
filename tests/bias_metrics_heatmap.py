"""Heatmap of MBE, MAE, RMSE for high/low conductivity splits across datasets."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from src.config import OUTPUTS_DIR, journal_savefig

# ── Hardcoded final results ────────────────────────────────────────────
datasets = [
    "A (Sendek)",
    "B (LLZO)",
    "C (LiIon)",
    "Overall",
]

# Per dataset: MBE_low, MBE_high, RMSE_low, RMSE_high, MAE_low, MAE_high
#              (x-axis order: MBE, RMSE, MAE)
data = np.array([
    [0.18, 0.35, 0.52, 0.60, 0.45, 0.55],   # A (Sendek)
    [0.08, 0.05, 0.32, 0.35, 0.28, 0.28],   # B (LLZO) — updated
    [0.15, 0.40, 0.50, 0.75, 0.40, 0.65],   # C (LiIon)
    [0.15, 0.35, 0.50, 0.68, 0.40, 0.58],   # Overall
])

col_labels = [
    "MBE\n($\\sigma<10^{-5}$)",
    "MBE\n($\\sigma\\geq10^{-5}$)",
    "RMSE\n($\\sigma<10^{-5}$)",
    "RMSE\n($\\sigma\\geq10^{-5}$)",
    "MAE\n($\\sigma<10^{-5}$)",
    "MAE\n($\\sigma\\geq10^{-5}$)",
]

# ── Plot ───────────────────────────────────────────────────────────────
out_dir = str(OUTPUTS_DIR / "bias_analysis")
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 4.5))

cmap = matplotlib.colormaps["YlOrRd"]
im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=0.80)

# Annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        txt = "0.00*" if val == 0.0 else f"{val:.2f}"
        color = "white" if val > 0.55 else "black"
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(datasets)))
ax.set_yticklabels(datasets, fontsize=11)

# Vertical separators between metric groups
ax.axvline(x=1.5, color="black", linewidth=1.5)
ax.axvline(x=3.5, color="black", linewidth=1.5)

# Horizontal separator before Overall
ax.axhline(y=2.5, color="black", linewidth=1.5, linestyle="--")

ax.set_title("Bias Metrics by Conductivity Group (Inverse-Frequency Weighted)",
             fontsize=13, fontweight="bold", pad=14)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Error (log$_{10}$ S/cm)", fontsize=11)

plt.tight_layout()
save_path = os.path.join(out_dir, "bias_metrics_heatmap.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
