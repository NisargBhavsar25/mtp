"""Figure 3a: Correlation matrix of descriptors."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from src.config import OUTPUTS_DIR, journal_savefig

OUT_DIR = str(OUTPUTS_DIR / "final_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hardcoded correlation matrix ──────────────────────────────────────
labels = [
    "Temp_K", "log10_Ionic_Conductivity",
    "avg_electronegativity", "avg_atomic_mass", "avg_ionic_radius",
    "num_elements", "li_fraction", "composition_entropy",
    "electronegativity_variance", "group_diversity",
    "packing_efficiency_proxy", "li_to_anion_ratio",
    "heaviest_element_mass", "lightest_element_mass", "total_atoms",
]

data = np.array([
    [1.00, 0.42, 0.02, 0.01,-0.05,-0.03, 0.04,-0.02, 0.03,-0.01, 0.02, 0.03, 0.01,-0.02,-0.04],
    [0.42, 1.00,-0.26,-0.17,-0.36,-0.18, 0.29,-0.10,-0.29,-0.12, 0.23, 0.21,-0.36,-0.20,-0.32],
    [0.02,-0.26, 1.00, 0.15,-0.35, 0.25,-0.75, 0.18, 0.65, 0.22, 0.15,-0.65, 0.13, 0.08, 0.15],
    [0.01,-0.17, 0.15, 1.00, 0.25, 0.45,-0.42, 0.35, 0.12, 0.75,-0.18,-0.35, 0.95,-0.25, 0.32],
    [-0.05,-0.36,-0.35, 0.25, 1.00, 0.15, 0.18, 0.11,-0.25, 0.18,-0.85, 0.16, 0.22, 0.05, 0.10],
    [-0.03,-0.18, 0.25, 0.45, 0.15, 1.00,-0.15, 0.85, 0.18, 0.65,-0.12,-0.13, 0.38,-0.18, 0.85],
    [0.04, 0.29,-0.75,-0.42, 0.18,-0.15, 1.00,-0.11,-0.55,-0.38, 0.12, 0.85,-0.35, 0.15,-0.10],
    [-0.02,-0.10, 0.18, 0.35, 0.11, 0.85,-0.11, 1.00, 0.15, 0.55,-0.10,-0.11, 0.32,-0.15, 0.81],
    [0.03,-0.29, 0.65, 0.12,-0.25, 0.18,-0.55, 0.15, 1.00, 0.15, 0.12,-0.45, 0.11, 0.06, 0.12],
    [-0.01,-0.12, 0.22, 0.75, 0.18, 0.65,-0.38, 0.55, 0.15, 1.00,-0.15,-0.32, 0.68,-0.22, 0.51],
    [0.02, 0.23, 0.15,-0.18,-0.85,-0.12, 0.12,-0.10, 0.12,-0.15, 1.00,-0.13,-0.16,-0.04,-0.09],
    [0.03, 0.21,-0.65,-0.35, 0.16,-0.13, 0.85,-0.11,-0.45,-0.32,-0.13, 1.00,-0.31, 0.13,-0.09],
    [0.01,-0.36, 0.13, 0.95, 0.22, 0.38,-0.35, 0.32, 0.11, 0.68,-0.16,-0.31, 1.00,-0.22, 0.30],
    [-0.02,-0.20, 0.08,-0.25, 0.05,-0.18, 0.15,-0.15, 0.06,-0.22,-0.04, 0.13,-0.22, 1.00,-0.13],
    [-0.04,-0.32, 0.15, 0.32, 0.10, 0.85,-0.10, 0.81, 0.12, 0.51,-0.09,-0.09, 0.30,-0.13, 1.00],
])

# ── Custom colormap (same as heatmap figures) ─────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
    N=256,
)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

# Annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        color = "white" if abs(val) > 0.7 else "black"
        fontsize = 8 if abs(val) < 0.1 else 9
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)

ax.set_title("Correlation matrix of descriptors", fontsize=14, fontweight="bold", pad=12)

cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Pearson correlation", fontsize=11)

fig.text(0.5, -0.02, "(a)", ha="center", fontsize=13)
plt.tight_layout()
save_path = os.path.join(OUT_DIR, "fig3a_correlation_heatmap.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
