"""Plot cross-dataset validation metrics as a grouped bar chart."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import os

from src.config import OUTPUTS_DIR, journal_savefig

# ── Data ───────────────────────────────────────────────────────────────
datasets = ["A (Sendek)\nN=39", "B (LLZO)\nN=117", "C (LiIon)\nN=425", "Overall\nN=581"]
metrics = {
    r"$R^2_{adj}$": [0.85, 0.78, 0.80, 0.81],
    "MAE":          [0.42, 0.28, 0.48, 0.44],
    "RMSE":         [0.50, 0.35, 0.58, 0.53],
    "MBE":          [0.15, 0.06, 0.20, 0.17],
    "STD":          [0.48, 0.34, 0.54, 0.50],
}

metric_names = list(metrics.keys())
n_datasets = len(datasets)
n_metrics = len(metric_names)

colors = ["#2166ac", "#4393c3", "#d6604d", "#f4a582", "#b2abd2"]

# ── Plot ───────────────────────────────────────────────────────────────
out_dir = str(OUTPUTS_DIR / "bias_analysis")
os.makedirs(out_dir, exist_ok=True)

x = np.arange(n_datasets)
bar_width = 0.15
offsets = np.arange(n_metrics) - (n_metrics - 1) / 2

fig, ax = plt.subplots(figsize=(11, 5.5))

for i, (name, vals) in enumerate(metrics.items()):
    pos = x + offsets[i] * bar_width
    bars = ax.bar(pos, vals, bar_width * 0.9, label=name, color=colors[i],
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel("Metric Value (log$_{10}$ S/cm)", fontsize=12)
ax.set_title("Cross-Dataset Validation Metrics (Inverse-Frequency Weighted)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 1.05)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.legend(loc="upper right", ncol=n_metrics, fontsize=10, framealpha=0.9)

# Vertical separator before Overall
ax.axvline(x=2.5, color="grey", linewidth=1.2, linestyle="--", alpha=0.6)

plt.tight_layout()
save_path = os.path.join(out_dir, "validation_metrics_grouped_bar.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
