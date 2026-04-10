"""Evaluate model on validation sets filtered to DDSE's training range.

Filters:
1. Target range: keep only samples where log10(IC) falls within DDSE's range
2. Temperature range: keep only samples where Temp_K falls within DDSE's range

Reports both filtered and unfiltered metrics side-by-side.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib

from src.config import MODELS_DIR, DATA_CLEANED
from src.evaluation.calculate_metrics import calculate_metrics
from src.features.get_composition import parse_mixture_formula
import src.features.get_composition as gc
from gensim.models import Word2Vec
from src.config import MAT2VEC_PRETRAINED

# ── Load model ─────────────────────────────────────────────────────────
metadata = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
pipeline = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
target_name = "log_Ionic_Conductivity"
feature_cols = metadata["feature_columns"][target_name]

# ── Determine DDSE training range ──────────────────────────────────────
ddse = pd.read_csv(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
ddse = ddse[ddse["Temp_K"] >= 293]
ic = ddse["Ionic_Conductivity"].copy().replace(0, 1e-12)
ic[ic <= 0] = 1e-12
ddse_log_target = np.log10(ic)

TARGET_MIN = ddse_log_target.min()
TARGET_MAX = ddse_log_target.max()
TEMP_MIN = ddse["Temp_K"].min()
TEMP_MAX = ddse["Temp_K"].max()

print("=" * 80)
print("DDSE TRAINING RANGE")
print("=" * 80)
print(f"  log10(IC) : [{TARGET_MIN:.2f}, {TARGET_MAX:.2f}]")
print(f"  Temp_K    : [{TEMP_MIN:.1f}, {TEMP_MAX:.1f}]")

# ── Mat2vec model ──────────────────────────────────────────────────────
m2v_model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
dim = 200


def featurize_and_predict(df, formula_col, target_col):
    """Build features and predict for a validation dataframe."""
    # Temperature
    if "Temp_K" not in df.columns and "temperature" in df.columns:
        df["Temp_K"] = df["temperature"]
    elif "Temp_K" not in df.columns:
        df["Temp_K"] = 298.0

    # Compositional features
    df_enh = gc.enhance_composition_features_fixed(df.copy(), formula_col)

    # Rename to orig_ prefix
    exclude_rename = [formula_col, "Temp_K", "doi", target_col,
                      "temperature", "target", "log_target",
                      "conductivity", "log10_target", "log10_predict",
                      "Ionic_Conductivity", "residue",
                      "ID", "source", "family", "ChemicalFamily",
                      "formula", "comp", "IC", "ic"]
    rename_map = {col: f"orig_{col}" for col in df_enh.columns if col not in exclude_rename}
    rename_map["Temp_K"] = "orig_Temp_K"
    df_model = df_enh.rename(columns=rename_map)

    # Mat2vec embeddings
    emb_list = []
    for formula in df_enh[formula_col]:
        try:
            elements = parse_mixture_formula(str(formula))
            vecs, wts = [], []
            for el, amt in elements.items():
                if el in m2v_model.wv:
                    vecs.append(m2v_model.wv[el])
                    wts.append(amt)
            if vecs:
                emb = np.average(vecs, axis=0, weights=wts)
                if len(emb) != dim:
                    r = np.zeros(dim)
                    r[:min(len(emb), dim)] = emb[:min(len(emb), dim)]
                    emb = r
            else:
                emb = np.zeros(dim)
        except Exception:
            emb = np.zeros(dim)
        emb_list.append(emb)

    m2v_df = pd.DataFrame(np.array(emb_list), columns=[f"mat2vec_{i}" for i in range(dim)])
    df_model = pd.concat([df_model.reset_index(drop=True), m2v_df], axis=1)

    X = df_model.reindex(columns=feature_cols, fill_value=0)
    y_pred = pipeline.predict(X.values)

    # Get temperature values for filtering
    temp_col = "orig_Temp_K" if "orig_Temp_K" in df_model.columns else None
    temps = df_model[temp_col].values if temp_col else np.full(len(df), 298.0)

    return y_pred, temps


def evaluate_dataset(name, csv_path, formula_col, target_col):
    """Evaluate on full and filtered validation set."""
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        print(f"\n--- {name}: target column '{target_col}' not found, skipping ---")
        return None

    y_true = df[target_col].values

    # Get predictions and temperatures
    y_pred, temps = featurize_and_predict(df.copy(), formula_col, target_col)

    # Remove NaN targets
    valid = ~np.isnan(y_true)
    y_true_clean = y_true[valid]
    y_pred_clean = y_pred[valid]
    temps_clean = temps[valid]

    # ── Unfiltered metrics ──
    metrics_full = calculate_metrics(y_true_clean, y_pred_clean)

    # ── Filter: target range only ──
    target_mask = (y_true_clean >= TARGET_MIN) & (y_true_clean <= TARGET_MAX)
    n_target = target_mask.sum()

    metrics_target = None
    if n_target >= 5:
        metrics_target = calculate_metrics(y_true_clean[target_mask], y_pred_clean[target_mask])

    # ── Filter: target + temperature range ──
    temp_mask = (temps_clean >= TEMP_MIN) & (temps_clean <= TEMP_MAX)
    both_mask = target_mask & temp_mask
    n_both = both_mask.sum()

    metrics_both = None
    if n_both >= 5:
        metrics_both = calculate_metrics(y_true_clean[both_mask], y_pred_clean[both_mask])

    # ── Print results ──
    print(f"\n{'=' * 80}")
    print(f"  {name} DATASET")
    print(f"{'=' * 80}")

    print(f"\n  Target range in this set: [{y_true_clean.min():.2f}, {y_true_clean.max():.2f}]")
    print(f"  Temp range in this set:   [{temps_clean.min():.1f}, {temps_clean.max():.1f}]")

    print(f"\n  {'Condition':<35s} {'N':>5s} {'R_adj^2':>9s} {'MAE':>8s} {'RMSE':>8s} {'MBE':>8s}")
    print(f"  {'-'*70}")

    print(f"  {'Unfiltered':<35s} {metrics_full['N']:>5d} {metrics_full['R_adj^2']:>9.4f} "
          f"{metrics_full['MAE']:>8.4f} {metrics_full['RMSE']:>8.4f} {metrics_full['MBE']:>8.4f}")

    if metrics_target:
        print(f"  {'Target range filtered':<35s} {metrics_target['N']:>5d} {metrics_target['R_adj^2']:>9.4f} "
              f"{metrics_target['MAE']:>8.4f} {metrics_target['RMSE']:>8.4f} {metrics_target['MBE']:>8.4f}")
    else:
        print(f"  {'Target range filtered':<35s} {'N/A (< 5 samples)':>40s}")

    if metrics_both:
        print(f"  {'Target + Temp range filtered':<35s} {metrics_both['N']:>5d} {metrics_both['R_adj^2']:>9.4f} "
              f"{metrics_both['MAE']:>8.4f} {metrics_both['RMSE']:>8.4f} {metrics_both['MBE']:>8.4f}")
    else:
        print(f"  {'Target + Temp range filtered':<35s} {'N/A (< 5 samples)':>40s}")

    # Show how many samples were kept
    print(f"\n  Samples kept: {n_target}/{len(y_true_clean)} (target filter), "
          f"{n_both}/{len(y_true_clean)} (target+temp filter)")

    return {
        "name": name,
        "full": metrics_full,
        "target_filtered": metrics_target,
        "both_filtered": metrics_both,
        "n_total": len(y_true_clean),
        "n_target": n_target,
        "n_both": n_both,
    }


# ── DDSE test set baseline ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("  DDSE TEST SET (baseline)")
print("=" * 80)

test_path = MODELS_DIR / f"test_{target_name}.csv"
if test_path.exists():
    test_df = pd.read_csv(test_path)
    X_test = test_df[feature_cols].values
    y_test = test_df[target_name].values
    y_pred_test = pipeline.predict(X_test)
    m_test = calculate_metrics(y_test, y_pred_test)
    print(f"\n  N={m_test['N']}  R_adj^2={m_test['R_adj^2']:.4f}  "
          f"MAE={m_test['MAE']:.4f}  RMSE={m_test['RMSE']:.4f}  MBE={m_test['MBE']:.4f}")

# ── Validation sets ────────────────────────────────────────────────────
validation_sets = [
    ("LLZO", DATA_CLEANED / "LLZO_clean.csv", "compound", "log10_target"),
    ("Sendek", DATA_CLEANED / "Sendek_clean.csv", "comp", "log10_target"),
    ("LiIon", DATA_CLEANED / "LiIonDatabase_clean.csv", "composition", "log_target"),
]

all_results = []
for name, csv_path, formula_col, target_col in validation_sets:
    if csv_path.exists():
        result = evaluate_dataset(name, csv_path, formula_col, target_col)
        if result:
            all_results.append(result)

# ── Summary ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("  SUMMARY COMPARISON")
print("=" * 80)
print(f"\n  {'Dataset':<12s} {'Condition':<30s} {'N':>5s} {'R_adj^2':>9s} {'MAE':>8s} {'RMSE':>8s}")
print(f"  {'-'*68}")

if test_path.exists():
    print(f"  {'DDSE Test':<12s} {'Hold-out (20%)':<30s} {m_test['N']:>5d} {m_test['R_adj^2']:>9.4f} "
          f"{m_test['MAE']:>8.4f} {m_test['RMSE']:>8.4f}")

for r in all_results:
    print(f"  {r['name']:<12s} {'Unfiltered':<30s} {r['full']['N']:>5d} {r['full']['R_adj^2']:>9.4f} "
          f"{r['full']['MAE']:>8.4f} {r['full']['RMSE']:>8.4f}")
    if r["target_filtered"]:
        m = r["target_filtered"]
        print(f"  {'':<12s} {'Target filtered':<30s} {m['N']:>5d} {m['R_adj^2']:>9.4f} "
              f"{m['MAE']:>8.4f} {m['RMSE']:>8.4f}")
    if r["both_filtered"]:
        m = r["both_filtered"]
        print(f"  {'':<12s} {'Target + Temp filtered':<30s} {m['N']:>5d} {m['R_adj^2']:>9.4f} "
              f"{m['MAE']:>8.4f} {m['RMSE']:>8.4f}")
