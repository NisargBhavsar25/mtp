"""Compare unweighted vs inverse-frequency-weighted training.

Trains both variants, evaluates on DDSE test set and cross-dataset validation
(filtered to DDSE target range), and prints a side-by-side comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib

from src.config import MODELS_DIR, DATA_CLEANED, MAT2VEC_PRETRAINED
from src.evaluation.calculate_metrics import calculate_metrics
from src.features.get_composition import parse_mixture_formula
import src.features.get_composition as gc
from gensim.models import Word2Vec


# ── Helpers ────────────────────────────────────────────────────────────
m2v_model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
DIM = 200


def featurize(df, formula_col, feature_cols):
    """Build feature matrix for a validation DataFrame."""
    if "Temp_K" not in df.columns and "temperature" in df.columns:
        df["Temp_K"] = df["temperature"]
    elif "Temp_K" not in df.columns:
        df["Temp_K"] = 298.0

    df_enh = gc.enhance_composition_features_fixed(df.copy(), formula_col)

    exclude = [formula_col, "Temp_K", "doi",
               "temperature", "target", "log_target",
               "conductivity", "log10_target", "log10_predict",
               "Ionic_Conductivity", "residue",
               "ID", "source", "family", "ChemicalFamily",
               "formula", "comp", "IC", "ic"]
    rename_map = {c: f"orig_{c}" for c in df_enh.columns if c not in exclude}
    rename_map["Temp_K"] = "orig_Temp_K"
    df_model = df_enh.rename(columns=rename_map)

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
                if len(emb) != DIM:
                    r = np.zeros(DIM)
                    r[:min(len(emb), DIM)] = emb[:min(len(emb), DIM)]
                    emb = r
            else:
                emb = np.zeros(DIM)
        except Exception:
            emb = np.zeros(DIM)
        emb_list.append(emb)

    m2v_df = pd.DataFrame(np.array(emb_list), columns=[f"mat2vec_{i}" for i in range(DIM)])
    df_model = pd.concat([df_model.reset_index(drop=True), m2v_df], axis=1)
    X = df_model.reindex(columns=feature_cols, fill_value=0)
    return X.values


def evaluate_validation(pipeline, feature_cols, target_min, target_max):
    """Evaluate pipeline on all validation sets, return dict of metrics."""
    validation_sets = [
        ("LLZO", DATA_CLEANED / "LLZO_clean.csv", "compound", "log10_target"),
        ("Sendek", DATA_CLEANED / "Sendek_clean.csv", "comp", "log10_target"),
        ("LiIon", DATA_CLEANED / "LiIonDatabase_clean.csv", "composition", "log_target"),
    ]

    results = {}
    for name, csv_path, formula_col, target_col in validation_sets:
        if not csv_path.exists():
            continue
        vdf = pd.read_csv(csv_path)
        if target_col not in vdf.columns:
            continue

        y_true = vdf[target_col].values
        X_val = featurize(vdf.copy(), formula_col, feature_cols)

        valid = ~np.isnan(y_true)
        y_true_c = y_true[valid]
        X_val_c = X_val[valid]
        y_pred = pipeline.predict(X_val_c)

        # Unfiltered
        m_full = calculate_metrics(y_true_c, y_pred)

        # Filtered to training range
        mask = (y_true_c >= target_min) & (y_true_c <= target_max)
        m_filt = calculate_metrics(y_true_c[mask], y_pred[mask]) if mask.sum() >= 5 else None

        results[name] = {"full": m_full, "filtered": m_filt, "n_filt": int(mask.sum())}

    return results


# ══════════════════════════════════════════════════════════════════════
#  TRAIN BOTH VARIANTS
# ══════════════════════════════════════════════════════════════════════
from src.training.train_best_save import DDSEModelTrainer
import tempfile, shutil

csv_path = str(DATA_CLEANED / "ddse_compositional_clean.csv")

results_table = []

for label, weighted in [("Unweighted (baseline)", False), ("Inverse-Frequency Weighted", True)]:
    print("\n" + "=" * 80)
    print(f"  TRAINING: {label}")
    print("=" * 80)

    # Train into a temp directory so we don't overwrite the saved model
    tmp_dir = tempfile.mkdtemp(prefix="mtp_compare_")

    trainer = DDSEModelTrainer(csv_path)
    trainer.train_and_save_best_models(save_dir=tmp_dir, use_sample_weights=weighted)

    # Load what was saved
    meta = joblib.load(str(Path(tmp_dir) / "model_metadata.joblib"))
    pipe = joblib.load(str(Path(tmp_dir) / "ddse_model_log_Ionic_Conductivity.joblib"))
    target = "log_Ionic_Conductivity"
    feat_cols = meta["feature_columns"][target]
    cfg = meta["model_configs"][target]

    # Test set
    test_df = pd.read_csv(str(Path(tmp_dir) / f"test_{target}.csv"))
    X_test = test_df[feat_cols].values
    y_test = test_df[target].values
    y_pred_test = pipe.predict(X_test)
    m_test = calculate_metrics(y_test, y_pred_test)

    # DDSE target range for filtering
    ddse_df = pd.read_csv(csv_path)
    ddse_df = ddse_df[ddse_df["Temp_K"] >= 293]
    ic = ddse_df["Ionic_Conductivity"].copy().replace(0, 1e-12)
    ic[ic <= 0] = 1e-12
    t_min = np.log10(ic).min()
    t_max = np.log10(ic).max()

    # Validation
    val_results = evaluate_validation(pipe, feat_cols, t_min, t_max)

    # Collect rows
    results_table.append({
        "Variant": label,
        "Model": cfg["best_model"],
        "Features": cfg["features"],
        "DDSE_Test_R2": m_test["R_adj^2"],
        "DDSE_Test_MAE": m_test["MAE"],
        "DDSE_Test_RMSE": m_test["RMSE"],
    })

    for ds_name in ["LLZO", "Sendek", "LiIon"]:
        if ds_name in val_results:
            vr = val_results[ds_name]
            results_table[-1][f"{ds_name}_R2_full"] = vr["full"]["R_adj^2"]
            results_table[-1][f"{ds_name}_MAE_full"] = vr["full"]["MAE"]
            if vr["filtered"]:
                results_table[-1][f"{ds_name}_R2_filt"] = vr["filtered"]["R_adj^2"]
                results_table[-1][f"{ds_name}_MAE_filt"] = vr["filtered"]["MAE"]
                results_table[-1][f"{ds_name}_N_filt"] = vr["n_filt"]

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  SIDE-BY-SIDE COMPARISON")
print("=" * 80)

for row in results_table:
    print(f"\n  --- {row['Variant']} ({row['Model']} + {row['Features']}) ---")
    print(f"  {'Metric':<35s} {'Value':>10s}")
    print(f"  {'-'*48}")
    print(f"  {'DDSE Test R_adj^2':<35s} {row['DDSE_Test_R2']:>10.4f}")
    print(f"  {'DDSE Test MAE':<35s} {row['DDSE_Test_MAE']:>10.4f}")
    print(f"  {'DDSE Test RMSE':<35s} {row['DDSE_Test_RMSE']:>10.4f}")

    for ds in ["LLZO", "Sendek", "LiIon"]:
        if f"{ds}_R2_full" in row:
            print(f"  {f'{ds} R2 (unfiltered)':<35s} {row[f'{ds}_R2_full']:>10.4f}")
            print(f"  {f'{ds} MAE (unfiltered)':<35s} {row[f'{ds}_MAE_full']:>10.4f}")
        if f"{ds}_R2_filt" in row:
            n_key = f"{ds}_N_filt"
            n_val = row[n_key]
            r2_filt = row[f"{ds}_R2_filt"]
            mae_filt = row[f"{ds}_MAE_filt"]
            label_r2 = f"{ds} R2 (filtered, N={n_val})"
            label_mae = f"{ds} MAE (filtered)"
            print(f"  {label_r2:<35s} {r2_filt:>10.4f}")
            print(f"  {label_mae:<35s} {mae_filt:>10.4f}")

# Delta summary
if len(results_table) == 2:
    base, weighted = results_table[0], results_table[1]
    print(f"\n\n  {'DELTA (Weighted - Baseline)':<35s}")
    print(f"  {'-'*48}")
    print(f"  {'DDSE Test R_adj^2':<35s} {weighted['DDSE_Test_R2'] - base['DDSE_Test_R2']:>+10.4f}")
    for ds in ["LLZO", "Sendek", "LiIon"]:
        k_full = f"{ds}_R2_full"
        k_filt = f"{ds}_R2_filt"
        if k_full in base and k_full in weighted:
            print(f"  {f'{ds} R2 (unfiltered)':<35s} {weighted[k_full] - base[k_full]:>+10.4f}")
        if k_filt in base and k_filt in weighted:
            print(f"  {f'{ds} R2 (filtered)':<35s} {weighted[k_filt] - base[k_filt]:>+10.4f}")
