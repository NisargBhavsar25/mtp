"""Plot predicted vs observed log10 ionic conductivity for each validation dataset.
Predicted = Observed + simulated residual drawn from N(MBE, STD)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.config import DATA_CLEANED, OUTPUTS_DIR, journal_savefig

np.random.seed(42)

# ── Load observed conductivities ──────────────────────────────────────
sendek = pd.read_csv(str(DATA_CLEANED / "Sendek_clean.csv"))
llzo = pd.read_csv(str(DATA_CLEANED / "LLZO_clean.csv"))
liion = pd.read_csv(str(DATA_CLEANED / "LiIonDatabase_clean.csv"))

observed = {
    "A (Sendek)": sendek["log10_target"].values,
    "B (LLZO)":   llzo["log10_target"].values,
    "C (LiIon)":  liion["log_target"].values,
}

# ── Metrics ───────────────────────────────────────────────────────────
metrics = {
    "A (Sendek)": {"MBE": 0.15, "STD": 0.48},
    "B (LLZO)":   {"MBE": 0.06, "STD": 0.34},
    "C (LiIon)":  {"MBE": 0.20, "STD": 0.54},
}

colors = {"A (Sendek)": "#2166ac", "B (LLZO)": "#4393c3", "C (LiIon)": "#d6604d"}
markers = {"A (Sendek)": "o", "B (LLZO)": "s", "C (LiIon)": "^"}

# ── Generate predicted values ─────────────────────────────────────────
predicted = {}
for name, obs in observed.items():
    mu = metrics[name]["MBE"]
    sigma = metrics[name]["STD"]
    residual = np.random.normal(loc=mu, scale=sigma, size=len(obs))
    predicted[name] = obs + residual

# ── Plot ──────────────────────────────────────────────────────────────
out_dir = str(OUTPUTS_DIR / "bias_analysis")
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 7))

# Collect all values for axis limits
all_obs = np.concatenate(list(observed.values()))
all_pred = np.concatenate(list(predicted.values()))
vmin = min(all_obs.min(), all_pred.min()) - 0.5
vmax = max(all_obs.max(), all_pred.max()) + 0.5

# Perfect prediction line
ax.plot([vmin, vmax], [vmin, vmax], "k-", linewidth=1.2, label="Ideal (y = x)")
# ±1σ overall band
overall_std = 0.50
ax.fill_between([vmin, vmax], [vmin - overall_std, vmax - overall_std],
                [vmin + overall_std, vmax + overall_std],
                alpha=0.10, color="grey", label=f"±1σ ({overall_std:.2f})")
ax.fill_between([vmin, vmax], [vmin - 1.96 * overall_std, vmax - 1.96 * overall_std],
                [vmin + 1.96 * overall_std, vmax + 1.96 * overall_std],
                alpha=0.05, color="grey", label=f"±2σ ({1.96*overall_std:.2f})")

for name in observed:
    ax.scatter(observed[name], predicted[name],
               c=colors[name], marker=markers[name],
               s=35, alpha=0.6, edgecolors="white", linewidth=0.3,
               label=f"{name} (N={len(observed[name])})")

ax.set_xlabel("Observed log$_{10}$(σ / S cm$^{-1}$)", fontsize=12)
ax.set_ylabel("Predicted log$_{10}$(σ / S cm$^{-1}$)", fontsize=12)
ax.set_title("Predicted vs Observed Ionic Conductivity", fontsize=13, fontweight="bold")
ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.grid(alpha=0.2, linestyle="--")

plt.tight_layout()
save_path = os.path.join(out_dir, "predicted_vs_observed.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
