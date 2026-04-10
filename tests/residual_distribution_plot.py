"""Plot estimated residual distributions for each validation dataset
using MBE (mean) and STD (standard deviation) to define normal distributions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

from src.config import OUTPUTS_DIR, journal_savefig

# ── Data ───────────────────────────────────────────────────────────────
datasets = {
    "A (Sendek)\nN=39":      {"MBE": 0.15, "STD": 0.48, "N": 39},
    "B (LLZO)\nN=117":       {"MBE": 0.06, "STD": 0.34, "N": 117},
    "C (LiIon)\nN=425":      {"MBE": 0.20, "STD": 0.54, "N": 425},
    "Overall\nN=581":        {"MBE": 0.17, "STD": 0.50, "N": 581},
}

colors = ["#2166ac", "#4393c3", "#d6604d", "#333333"]
x = np.linspace(-2.5, 2.5, 500)

# ── Plot ───────────────────────────────────────────────────────────────
out_dir = str(OUTPUTS_DIR / "bias_analysis")
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
axes = axes.ravel()

for i, (name, stats) in enumerate(datasets.items()):
    ax = axes[i]
    mu, sigma = stats["MBE"], stats["STD"]
    pdf = norm.pdf(x, loc=mu, scale=sigma)

    ax.fill_between(x, pdf, alpha=0.3, color=colors[i])
    ax.plot(x, pdf, color=colors[i], linewidth=2)

    # Mark MBE
    ax.axvline(mu, color=colors[i], linestyle="--", linewidth=1.5, label=f"MBE={mu:.2f}")
    # Mark zero
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

    # Shade ±1σ region
    x_1s = np.linspace(mu - sigma, mu + sigma, 200)
    ax.fill_between(x_1s, norm.pdf(x_1s, loc=mu, scale=sigma),
                    alpha=0.15, color=colors[i], label=f"±1σ = ±{sigma:.2f}")

    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-2.5, 2.5)

fig.supxlabel("Residual (Predicted − Actual) [log$_{10}$ S/cm]", fontsize=12)
fig.supylabel("Probability Density", fontsize=12)
fig.suptitle("Estimated Residual Distributions by Dataset",
             fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
save_path = os.path.join(out_dir, "residual_distributions.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")

# ── Also create an overlay version ────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))

for i, (name, stats) in enumerate(datasets.items()):
    mu, sigma = stats["MBE"], stats["STD"]
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    label = name.replace("\n", " ")
    ax2.plot(x, pdf, color=colors[i], linewidth=2, label=label)
    ax2.fill_between(x, pdf, alpha=0.15, color=colors[i])

ax2.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.5, label="Zero bias")
ax2.set_xlabel("Residual (Predicted − Actual) [log$_{10}$ S/cm]", fontsize=12)
ax2.set_ylabel("Probability Density", fontsize=12)
ax2.set_title("Estimated Residual Distributions — All Datasets Overlaid",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="upper right")
ax2.set_xlim(-2.5, 2.5)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
save_path2 = os.path.join(out_dir, "residual_distributions_overlay.png")
journal_savefig(save_path2)
plt.close()
print(f"Saved: {save_path2}")
