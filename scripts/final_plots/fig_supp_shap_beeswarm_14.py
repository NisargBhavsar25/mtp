"""Supplementary: SHAP beeswarm for the 14 descriptors used in correlation
analysis (includes Temp_K, excludes is_mixture and formula_complexity).
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
from scipy.stats import gaussian_kde

from src.config import MODELS_DIR, OUTPUTS_DIR, journal_savefig

OUT_DIR = str(OUTPUTS_DIR / "final_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── The 14 descriptors matching the correlation matrix (fig 3a) ───────
KEEP_FEATURES = [
    "Temp_K",
    "avg_electronegativity",
    "avg_atomic_mass",
    "avg_ionic_radius",
    "num_elements",
    "li_fraction",
    "composition_entropy",
    "electronegativity_variance",
    "group_diversity",
    "packing_efficiency_proxy",
    "li_to_anion_ratio",
    "heaviest_element_mass",
    "lightest_element_mass",
    "total_atoms",
]

# ── Load model and data ──────────────────────────────────────────────
MODEL_DIR_14 = MODELS_DIR / "model_14feat"
metadata = joblib.load(str(MODEL_DIR_14 / "model_metadata.joblib"))
pipeline = joblib.load(str(MODEL_DIR_14 / "ddse_model_log_Ionic_Conductivity.joblib"))
target = "log_Ionic_Conductivity"
feat_cols = metadata["feature_columns"][target]

# Map orig_ prefixed names to display names, filter to our 14
keep_idx = []
keep_names = []
for i, c in enumerate(feat_cols):
    if c.startswith("orig_"):
        name = c[5:]  # strip orig_ prefix
        if name in KEEP_FEATURES:
            keep_idx.append(i)
            keep_names.append(name)

print(f"Selected {len(keep_names)} features: {keep_names}")

# ── Compute SHAP ─────────────────────────────────────────────────────
test_df = pd.read_csv(str(MODEL_DIR_14 / f"test_{target}.csv"))
X_test = test_df[feat_cols].values

model = pipeline.named_steps["model"]
imputer = pipeline.named_steps["imputer"]
scaler = pipeline.named_steps["scaler"]
X_transformed = scaler.transform(imputer.transform(X_test))

print("Running SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# Extract selected features
shap_sel = shap_values[:, keep_idx]
fval_sel = X_transformed[:, keep_idx]

# Sort by mean |SHAP| (ascending for bottom-to-top)
mean_abs = np.mean(np.abs(shap_sel), axis=0)
sort_idx = np.argsort(mean_abs)
shap_sorted = shap_sel[:, sort_idx]
fval_sorted = fval_sel[:, sort_idx]
names_sorted = [keep_names[i] for i in sort_idx]
n_features = len(names_sorted)

# ── Colormap (blue-to-red matching the reference image) ──────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "shap_br", ["#3B4CC0", "#8CAAD0", "#C9B4D0", "#DD6E6E", "#E8443A"], N=256
)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 9))

np.random.seed(42)

for i in range(n_features):
    sv = shap_sorted[:, i]
    fv = fval_sorted[:, i]

    # Normalise feature values to [0, 1]
    fv_min, fv_max = fv.min(), fv.max()
    fv_norm = (fv - fv_min) / (fv_max - fv_min) if fv_max - fv_min > 0 else np.full_like(fv, 0.5)

    # KDE-based jitter for beeswarm shape
    try:
        kde = gaussian_kde(sv, bw_method=0.3)
        kde_at_pts = kde(sv)
        kde_at_pts = kde_at_pts / kde_at_pts.max() * 0.38
    except Exception:
        kde_at_pts = np.full_like(sv, 0.15)

    jitter = np.random.uniform(-1, 1, size=len(sv)) * kde_at_pts

    # Sort by feature value so high values render on top
    order = np.argsort(fv_norm)

    ax.scatter(
        sv[order], i + jitter[order],
        c=fv_norm[order], cmap=cmap,
        s=10, alpha=0.7, edgecolors="none",
        vmin=0, vmax=1, rasterized=True,
    )

    # Violin outline
    try:
        x_grid = np.linspace(sv.min() - 0.02, sv.max() + 0.02, 200)
        density = kde(x_grid)
        density = density / density.max() * 0.38
        ax.plot(x_grid, i + density, color="grey", linewidth=0.4, alpha=0.3)
        ax.plot(x_grid, i - density, color="grey", linewidth=0.4, alpha=0.3)
    except Exception:
        pass

ax.set_yticks(range(n_features))
ax.set_yticklabels(names_sorted, fontsize=11)
ax.axvline(0, color="grey", linewidth=0.8, alpha=0.5)
ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
ax.set_title("SHAP Beeswarm — Physical Features",
             fontsize=13, fontweight="bold")
ax.grid(False)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Low", "", "High"])
cbar.set_label("Feature value", fontsize=11)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "fig_supp_shap_beeswarm_14.png")
journal_savefig(save_path)
plt.close()
print(f"Saved: {save_path}")
