"""Figure 4: Residual distribution on DDSE test set (actual model predictions)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import joblib

from src.config import MODELS_DIR, OUTPUTS_DIR, journal_savefig

OUT_DIR = str(OUTPUTS_DIR / "final_plots")

# ── Load model and test set ───────────────────────────────────────────
metadata = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
pipeline = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
target = "log_Ionic_Conductivity"
feat_cols = metadata["feature_columns"][target]

test_df = pd.read_csv(str(MODELS_DIR / f"test_{target}.csv"))
X_test = test_df[feat_cols].values
y_test = test_df[target].values
y_pred = pipeline.predict(X_test)

residuals = y_pred - y_test
mbe = np.mean(residuals)
std = np.std(residuals, ddof=1)

print(f"N = {len(residuals)}")
print(f"MBE = {mbe:.2f}")
print(f"STD = {std:.2f}")

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# Histogram
ax.hist(residuals, bins=25, density=True, color="#F4A460", edgecolor="white",
        linewidth=0.6, alpha=0.9, label=f"DDSE Test (MBE={mbe:.2f})")

# KDE fit overlay
kde = stats.gaussian_kde(residuals)
x_fit = np.linspace(residuals.min() - 0.3, residuals.max() + 0.3, 300)
y_fit = kde(x_fit)
ax.plot(x_fit, y_fit, "k--", linewidth=1.8,
        label=f"Normal ($\\mu$={mbe:.2f}, $\\sigma$={std:.2f})")

# Mean line
ax.axvline(mbe, color="black", linewidth=1.2, linestyle=":")

ax.set_xlabel("Residuals (Pred $-$ Obs) [dex]", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Residual distribution on DDSE test set", fontsize=13, fontweight="bold")
ax.set_xlim(-2.5, 2.5)
ax.legend(fontsize=11, loc="upper right", frameon=False)
ax.grid(False)

plt.tight_layout()
save_path = f"{OUT_DIR}/fig4_ddse_residual_distribution.png"
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
