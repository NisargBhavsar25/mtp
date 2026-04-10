"""Full comparison of T vs 1/T models:
1. Residual vs observed conductivity
2. Predicted vs observed
3. Bias metrics heatmap split at 10^-5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import joblib

from src.config import DATA_CLEANED, OUTPUTS_DIR, MODELS_DIR, journal_savefig
from src.training.train_best_save import DDSEModelTrainer
from sklearn.metrics import r2_score

SELECTED_PHYSICAL = [
    "avg_ionic_radius", "li_fraction", "electronegativity_variance",
    "packing_efficiency_proxy", "heaviest_element_mass", "composition_entropy",
]

csv_path = str(DATA_CLEANED / "ddse_compositional_clean.csv")
out_dir = str(OUTPUTS_DIR / "model_comparison")
os.makedirs(out_dir, exist_ok=True)

SPLIT = -5.0  # log10(10^-5)

def compute_split_metrics(y_true, y_pred):
    errors = y_pred - y_true
    low = y_true < SPLIT
    high = y_true >= SPLIT
    def m(mask):
        e = errors[mask]
        return {
            "N": int(mask.sum()),
            "MBE": np.mean(e),
            "MAE": np.mean(np.abs(e)),
            "RMSE": np.sqrt(np.mean(e**2)),
        }
    return m(low), m(high)

# ── Train T model ─────────────────────────────────────────────────────
print("=" * 60)
print("Training model with Temp_K")
print("=" * 60)
trainer_T = DDSEModelTrainer(csv_path, feature_subset=SELECTED_PHYSICAL + ["Temp_K"])
trainer_T.train_and_save_best_models(use_sample_weights=False)

test_T = pd.read_csv(str(MODELS_DIR / "test_log_Ionic_Conductivity.csv"))
pipe_T = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
meta_T = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
feat_T = meta_T["feature_columns"]["log_Ionic_Conductivity"]
y_true_T = test_T["log_Ionic_Conductivity"].values
y_pred_T = pipe_T.predict(test_T[feat_T].values)
res_T = y_pred_T - y_true_T
low_T, high_T = compute_split_metrics(y_true_T, y_pred_T)

# ── Train 1/T model ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training model with 1/Temp_K")
print("=" * 60)
df = pd.read_csv(csv_path)
df["inv_Temp_K"] = 1.0 / df["Temp_K"]
tmp_csv = str(DATA_CLEANED / "ddse_compositional_clean_inv_T.csv")
df.to_csv(tmp_csv, index=False)

trainer_invT = DDSEModelTrainer(tmp_csv, feature_subset=SELECTED_PHYSICAL + ["inv_Temp_K"])
trainer_invT.train_and_save_best_models(use_sample_weights=False)

test_invT = pd.read_csv(str(MODELS_DIR / "test_log_Ionic_Conductivity.csv"))
pipe_invT = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
meta_invT = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
feat_invT = meta_invT["feature_columns"]["log_Ionic_Conductivity"]
y_true_invT = test_invT["log_Ionic_Conductivity"].values
y_pred_invT = pipe_invT.predict(test_invT[feat_invT].values)
res_invT = y_pred_invT - y_true_invT
low_invT, high_invT = compute_split_metrics(y_true_invT, y_pred_invT)

# ══════════════════════════════════════════════════════════════════════
# PLOT 1: Residual vs Observed (side by side)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for ax, yt, res, title in [
    (axes[0], y_true_T, res_T, "With T"),
    (axes[1], y_true_invT, res_invT, "With 1/T"),
]:
    ax.scatter(yt, res, s=20, alpha=0.5, c="#2166ac", edgecolors="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-np.std(res, ddof=1), np.std(res, ddof=1), alpha=0.08, color="grey", label=r"$\pm 1\sigma$")
    ax.axhspan(-1.96*np.std(res, ddof=1), 1.96*np.std(res, ddof=1), alpha=0.04, color="grey", label=r"$\pm 2\sigma$")
    ax.axvline(SPLIT, color="red", linewidth=1, linestyle="--", alpha=0.6, label=r"$10^{-5}$ split")
    ax.set_xlabel(r"Observed log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(fontsize=9, loc="upper left")

axes[0].set_ylabel("Residual (Predicted - Observed)", fontsize=12)
fig.suptitle("Residual vs Observed Conductivity", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
journal_savefig(os.path.join(out_dir, "residual_vs_observed_T_invT.png"))
plt.close()
print("Saved: residual_vs_observed_T_invT.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 2: Predicted vs Observed (side by side)
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, yt, yp, title in [
    (axes[0], y_true_T, y_pred_T, "With T"),
    (axes[1], y_true_invT, y_pred_invT, "With 1/T"),
]:
    all_v = np.concatenate([yt, yp])
    vmin, vmax = all_v.min() - 0.5, all_v.max() + 0.5
    ax.plot([vmin, vmax], [vmin, vmax], "k-", linewidth=1.2, label="Ideal")
    std = np.std(yp - yt, ddof=1)
    ax.fill_between([vmin, vmax], [vmin - std, vmax - std], [vmin + std, vmax + std],
                    alpha=0.1, color="grey", label=rf"$\pm 1\sigma$ ({std:.2f})")
    ax.scatter(yt, yp, s=20, alpha=0.5, c="#2166ac", edgecolors="white", linewidth=0.3)
    ax.set_xlabel(r"Observed log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=12)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(fontsize=9, loc="upper left")

axes[0].set_ylabel(r"Predicted log$_{10}$($\sigma$ / S cm$^{-1}$)", fontsize=12)
fig.suptitle("Predicted vs Observed Ionic Conductivity", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
journal_savefig(os.path.join(out_dir, "predicted_vs_observed_T_invT.png"))
plt.close()
print("Saved: predicted_vs_observed_T_invT.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 3: Bias metrics heatmap split at 10^-5
# ══════════════════════════════════════════════════════════════════════
row_labels = [
    r"With T — Low ($<10^{-5}$)" + f"  N={low_T['N']}",
    r"With T — High ($\geq10^{-5}$)" + f"  N={high_T['N']}",
    r"With 1/T — Low ($<10^{-5}$)" + f"  N={low_invT['N']}",
    r"With 1/T — High ($\geq10^{-5}$)" + f"  N={high_invT['N']}",
]
col_labels = ["MBE", "RMSE", "MAE"]

data = np.array([
    [low_T["MBE"],    low_T["RMSE"],    low_T["MAE"]],
    [high_T["MBE"],   high_T["RMSE"],   high_T["MAE"]],
    [low_invT["MBE"], low_invT["RMSE"], low_invT["MAE"]],
    [high_invT["MBE"],high_invT["RMSE"],high_invT["MAE"]],
])

# Use absolute values for coloring, show signed for MBE
data_abs = np.abs(data)

fig, ax = plt.subplots(figsize=(9, 5))
cmap = matplotlib.colormaps["YlOrRd"]
im = ax.imshow(data_abs, cmap=cmap, aspect="auto", vmin=0, vmax=1.0)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        color = "white" if data_abs[i, j] > 0.6 else "black"
        sign = "+" if val > 0 else ""
        ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=12)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)

ax.axhline(y=1.5, color="black", linewidth=2)

ax.set_title(r"Bias Metrics by Conductivity Group: T vs 1/T (split at $10^{-5}$)",
             fontsize=13, fontweight="bold", pad=12)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("|Error| (log$_{10}$ S/cm)", fontsize=11)

plt.tight_layout()
journal_savefig(os.path.join(out_dir, "bias_heatmap_T_invT.png"))
plt.close()
print("Saved: bias_heatmap_T_invT.png")

# Print summary
print("\n" + "=" * 60)
print("SPLIT METRICS SUMMARY")
print("=" * 60)
print(f"  {'':40s} {'MBE':>8s} {'RMSE':>8s} {'MAE':>8s} {'N':>6s}")
print(f"  {'-'*66}")
print(f"  {'With T   — Low  (< 10^-5)':<40s} {low_T['MBE']:>+8.4f} {low_T['RMSE']:>8.4f} {low_T['MAE']:>8.4f} {low_T['N']:>6d}")
print(f"  {'With T   — High (>= 10^-5)':<40s} {high_T['MBE']:>+8.4f} {high_T['RMSE']:>8.4f} {high_T['MAE']:>8.4f} {high_T['N']:>6d}")
print(f"  {'With 1/T — Low  (< 10^-5)':<40s} {low_invT['MBE']:>+8.4f} {low_invT['RMSE']:>8.4f} {low_invT['MAE']:>8.4f} {low_invT['N']:>6d}")
print(f"  {'With 1/T — High (>= 10^-5)':<40s} {high_invT['MBE']:>+8.4f} {high_invT['RMSE']:>8.4f} {high_invT['MAE']:>8.4f} {high_invT['N']:>6d}")
