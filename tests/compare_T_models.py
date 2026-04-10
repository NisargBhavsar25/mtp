"""Train both T and 1/T models, compute evaluation metrics, and plot comparison."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from src.config import DATA_CLEANED, OUTPUTS_DIR, MODELS_DIR, journal_savefig
from src.training.train_best_save import DDSEModelTrainer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SELECTED_PHYSICAL = [
    "avg_ionic_radius",
    "li_fraction",
    "electronegativity_variance",
    "packing_efficiency_proxy",
    "heaviest_element_mass",
    "composition_entropy",
]

csv_path = str(DATA_CLEANED / "ddse_compositional_clean.csv")

def compute_all_metrics(y_true, y_pred):
    errors = y_pred - y_true
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - 2))
    return {
        "R2_adj": r2_adj,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MBE": np.mean(errors),
        "STD": np.std(errors, ddof=1),
    }

results = {}

# ── Model with T ──────────────────────────────────────────────────────
print("=" * 60)
print("Training model with Temp_K")
print("=" * 60)
features_T = SELECTED_PHYSICAL + ["Temp_K"]
trainer_T = DDSEModelTrainer(csv_path, feature_subset=features_T)
trainer_T.train_and_save_best_models(use_sample_weights=False)

import joblib
test_T = pd.read_csv(str(MODELS_DIR / "test_log_Ionic_Conductivity.csv"))
pipe_T = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
meta_T = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
feat_T = meta_T["feature_columns"]["log_Ionic_Conductivity"]
y_true_T = test_T["log_Ionic_Conductivity"].values
y_pred_T = pipe_T.predict(test_T[feat_T].values)
results["With T"] = compute_all_metrics(y_true_T, y_pred_T)
print(f"Metrics (T): {results['With T']}")

# ── Model with 1/T ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training model with 1/Temp_K")
print("=" * 60)
df = pd.read_csv(csv_path)
df["inv_Temp_K"] = 1.0 / df["Temp_K"]
tmp_csv = str(DATA_CLEANED / "ddse_compositional_clean_inv_T.csv")
df.to_csv(tmp_csv, index=False)

features_invT = SELECTED_PHYSICAL + ["inv_Temp_K"]
trainer_invT = DDSEModelTrainer(tmp_csv, feature_subset=features_invT)
trainer_invT.train_and_save_best_models(use_sample_weights=False)

test_invT = pd.read_csv(str(MODELS_DIR / "test_log_Ionic_Conductivity.csv"))
pipe_invT = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
meta_invT = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
feat_invT = meta_invT["feature_columns"]["log_Ionic_Conductivity"]
y_true_invT = test_invT["log_Ionic_Conductivity"].values
y_pred_invT = pipe_invT.predict(test_invT[feat_invT].values)
results["With 1/T"] = compute_all_metrics(y_true_invT, y_pred_invT)
print(f"Metrics (1/T): {results['With 1/T']}")

# ── Plot comparison ───────────────────────────────────────────────────
out_dir = str(OUTPUTS_DIR / "model_comparison")
os.makedirs(out_dir, exist_ok=True)

metric_names = ["R2_adj", "MAE", "RMSE", "MBE", "STD"]
display_names = [r"$R^2_{adj}$", "MAE", "RMSE", "MBE", "STD"]
model_labels = ["With T", "With 1/T"]
colors = ["#2166ac", "#d6604d"]

vals = {m: [results[m][k] for k in metric_names] for m in model_labels}

x = np.arange(len(metric_names))
bar_width = 0.32

fig, ax = plt.subplots(figsize=(10, 5.5))

for i, (label, color) in enumerate(zip(model_labels, colors)):
    offset = (i - 0.5) * bar_width
    v = vals[label]
    bars = ax.bar(x + offset, v, bar_width * 0.9, label=label, color=color,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, v):
        y_pos = bar.get_height()
        va = "bottom" if val >= 0 else "top"
        nudge = 0.015 if val >= 0 else -0.015
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + nudge,
                f"{val:.3f}", ha="center", va=va, fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=12)
ax.set_ylabel("Metric Value", fontsize=12)
ax.set_title("Model Comparison: T vs 1/T (DDSE Test Set)",
             fontsize=13, fontweight="bold", pad=12)
ax.axhline(0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.legend(fontsize=11, loc="upper right")

plt.tight_layout()
save_path = os.path.join(out_dir, "T_vs_invT_metrics.png")
journal_savefig(save_path)
plt.close()
print(f"\nSaved: {save_path}")

# Print summary table
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Metric':<10s} {'With T':>10s} {'With 1/T':>10s}")
print(f"  {'-'*32}")
for k, d in zip(metric_names, display_names):
    print(f"  {k:<10s} {results['With T'][k]:>10.4f} {results['With 1/T'][k]:>10.4f}")
