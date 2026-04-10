"""Figure 3b: SHAP violin plot for physical features (7 features).
Uses actual SHAP values from the trained model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import shap
import os

from src.config import MODELS_DIR, OUTPUTS_DIR, DATA_CLEANED, journal_savefig

OUT_DIR = str(OUTPUTS_DIR / "final_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model and rebuild features ───────────────────────────────────
metadata = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
pipeline = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
target = "log_Ionic_Conductivity"
feat_cols = metadata["feature_columns"][target]

# Identify physical feature indices and names
phys_features = []
phys_idx = []
for i, c in enumerate(feat_cols):
    if c.startswith("orig_"):
        phys_features.append(c[5:])  # strip orig_ prefix
        phys_idx.append(i)

print(f"Physical features ({len(phys_features)}): {phys_features}")

# ── Load test set and compute SHAP ────────────────────────────────────
test_df = pd.read_csv(str(MODELS_DIR / f"test_{target}.csv"))
X_test = test_df[feat_cols].values

model = pipeline.named_steps["model"]
imputer = pipeline.named_steps["imputer"]
scaler = pipeline.named_steps["scaler"]

X_transformed = scaler.transform(imputer.transform(X_test))

print("Running SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)
print(f"SHAP values shape: {shap_values.shape}")

# Extract physical feature SHAP values and feature values
shap_phys = shap_values[:, phys_idx]
fval_phys = X_transformed[:, phys_idx]

# Sort by mean |SHAP|
mean_abs = np.mean(np.abs(shap_phys), axis=0)
sort_idx = np.argsort(mean_abs)  # ascending for bottom-to-top
shap_sorted = shap_phys[:, sort_idx]
fval_sorted = fval_phys[:, sort_idx]
names_sorted = [phys_features[i] for i in sort_idx]

n_features = len(names_sorted)

# ── Custom colormap (same as heatmap) ─────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging",
    ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
    N=256,
)

# ── Plot: violin with color by feature value ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

for i in range(n_features):
    sv = shap_sorted[:, i]
    fv = fval_sorted[:, i]

    # Normalise feature values to [0, 1] for coloring
    fv_min, fv_max = fv.min(), fv.max()
    if fv_max - fv_min > 0:
        fv_norm = (fv - fv_min) / (fv_max - fv_min)
    else:
        fv_norm = np.full_like(fv, 0.5)

    # Sort by SHAP value for smooth violin outline
    sort_sv = np.argsort(sv)
    sv_s = sv[sort_sv]
    fv_s = fv_norm[sort_sv]

    # KDE for violin shape
    try:
        kde = gaussian_kde(sv, bw_method=0.3)
        x_grid = np.linspace(sv.min() - 0.05, sv.max() + 0.05, 200)
        density_grid = kde(x_grid)
        density_grid = density_grid / density_grid.max() * 0.38
    except Exception:
        x_grid = np.linspace(sv.min(), sv.max(), 200)
        density_grid = np.full_like(x_grid, 0.1)

    # Draw filled violin outline (light grey background)
    ax.fill_betweenx(
        [i - d for d in density_grid], x_grid,
        [i + d for d in density_grid], x_grid,
        alpha=0.0,  # invisible, just for shape reference
    )

    # Scatter with KDE-based jitter (smooth spread)
    kde_at_pts = kde(sv)
    kde_at_pts = kde_at_pts / kde_at_pts.max() * 0.35
    jitter = np.random.uniform(-1, 1, size=len(sv)) * kde_at_pts

    # Sort by feature value so high values render on top
    order = np.argsort(fv_norm)

    ax.scatter(
        sv[order],
        i + jitter[order],
        c=fv_norm[order],
        cmap=cmap,
        s=12,
        alpha=0.7,
        edgecolors="none",
        vmin=0, vmax=1,
        rasterized=True,
    )

    # Add smoothed violin outline
    ax.plot(x_grid, i + density_grid, color="grey", linewidth=0.5, alpha=0.4)
    ax.plot(x_grid, i - density_grid, color="grey", linewidth=0.5, alpha=0.4)

ax.set_yticks(range(n_features))
ax.set_yticklabels(names_sorted, fontsize=11)
ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

# Clip x-axis to show main distribution
all_shap = shap_sorted.flatten()
q_lo, q_hi = np.percentile(all_shap, [1, 99])
margin = (q_hi - q_lo) * 0.15
ax.set_xlim(q_lo - margin, q_hi + margin)

ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
ax.set_title("SHAP Feature Importance — Physical Features",
             fontsize=13, fontweight="bold")
ax.grid(False)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Low", "", "High"])
cbar.set_label("Feature value", fontsize=11)

fig.text(0.5, -0.02, "(b)", ha="center", fontsize=13)
plt.tight_layout()
save_path = os.path.join(OUT_DIR, "fig3b_shap_violin.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
