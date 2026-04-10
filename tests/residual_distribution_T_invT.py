"""Residual distribution histograms for T vs 1/T models, split at 10^-5."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from src.config import DATA_CLEANED, OUTPUTS_DIR, MODELS_DIR, journal_savefig
from src.training.train_best_save import DDSEModelTrainer

SELECTED_PHYSICAL = [
    "avg_ionic_radius", "li_fraction", "electronegativity_variance",
    "packing_efficiency_proxy", "heaviest_element_mass", "composition_entropy",
]
csv_path = str(DATA_CLEANED / "ddse_compositional_clean.csv")
out_dir = str(OUTPUTS_DIR / "model_comparison")
os.makedirs(out_dir, exist_ok=True)
SPLIT = -5.0

def get_residuals(features, csv):
    trainer = DDSEModelTrainer(csv, feature_subset=features)
    trainer.train_and_save_best_models(use_sample_weights=False)
    test = pd.read_csv(str(MODELS_DIR / "test_log_Ionic_Conductivity.csv"))
    pipe = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
    meta = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
    feat = meta["feature_columns"]["log_Ionic_Conductivity"]
    yt = test["log_Ionic_Conductivity"].values
    yp = pipe.predict(test[feat].values)
    return yt, yp - yt

# T model
print("Training T model...")
yt_T, res_T = get_residuals(SELECTED_PHYSICAL + ["Temp_K"], csv_path)

# 1/T model
print("Training 1/T model...")
df = pd.read_csv(csv_path)
df["inv_Temp_K"] = 1.0 / df["Temp_K"]
tmp_csv = str(DATA_CLEANED / "ddse_compositional_clean_inv_T.csv")
df.to_csv(tmp_csv, index=False)
yt_invT, res_invT = get_residuals(SELECTED_PHYSICAL + ["inv_Temp_K"], tmp_csv)

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

bins = np.linspace(-3.5, 3.5, 40)

for col, (yt, res, label) in enumerate([
    (yt_T, res_T, "With T"),
    (yt_invT, res_invT, "With 1/T"),
]):
    low = yt < SPLIT
    high = yt >= SPLIT

    # Top row: overall
    ax = axes[0, col]
    ax.hist(res, bins=bins, color="#2166ac", edgecolor="white", alpha=0.8, density=True)
    ax.axvline(np.mean(res), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean = {np.mean(res):+.3f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{label} — All (N={len(res)})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual (Predicted - Observed)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, linestyle="--")

    # Bottom row: split
    ax = axes[1, col]
    ax.hist(res[low], bins=bins, color="#d6604d", edgecolor="white", alpha=0.7,
            density=True, label=rf"Low ($<10^{{-5}}$) N={low.sum()}, $\mu$={np.mean(res[low]):+.3f}")
    ax.hist(res[high], bins=bins, color="#4393c3", edgecolor="white", alpha=0.7,
            density=True, label=rf"High ($\geq 10^{{-5}}$) N={high.sum()}, $\mu$={np.mean(res[high]):+.3f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{label} — By Conductivity Group", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual (Predicted - Observed)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, linestyle="--")

fig.suptitle("Residual Distribution: T vs 1/T", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
journal_savefig(os.path.join(out_dir, "residual_distribution_T_invT.png"))
plt.close()
print(f"Saved: {os.path.join(out_dir, 'residual_distribution_T_invT.png')}")
