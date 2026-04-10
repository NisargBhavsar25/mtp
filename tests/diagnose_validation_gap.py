"""Diagnose why test set R²=0.78 but validation R²≈0.

Check for:
1. Target scale mismatch (log10 vs ln vs raw)
2. Feature distribution shift
3. Temperature distribution differences
4. Prediction distribution sanity
5. Whether the model is just predicting a constant
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib
from src.config import MODELS_DIR, DATA_CLEANED

print("=" * 80)
print("DIAGNOSIS: Why does the model fail on validation sets?")
print("=" * 80)

# ── 1. CHECK TARGET SCALES ────────────────────────────────────────────
print("\n\n>>> CHECK 1: TARGET SCALE COMPARISON <<<")
print("-" * 60)

# DDSE training target
ddse = pd.read_csv(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
ddse = ddse[ddse["Temp_K"] >= 293]
ic = ddse["Ionic_Conductivity"].copy().replace(0, 1e-12)
ic[ic <= 0] = 1e-12
ddse_log_target = np.log10(ic)

print(f"DDSE log10(IC) — training target:")
print(f"  min={ddse_log_target.min():.2f}, max={ddse_log_target.max():.2f}, "
      f"mean={ddse_log_target.mean():.2f}, std={ddse_log_target.std():.2f}")

# LLZO
llzo = pd.read_csv(str(DATA_CLEANED / "LLZO_clean.csv"))
print(f"\nLLZO columns: {list(llzo.columns)}")
print(f"LLZO 'log10_target':")
print(f"  min={llzo['log10_target'].min():.2f}, max={llzo['log10_target'].max():.2f}, "
      f"  mean={llzo['log10_target'].mean():.2f}, std={llzo['log10_target'].std():.2f}")
if 'conductivity' in llzo.columns:
    llzo_recomputed = np.log10(llzo['conductivity'].clip(lower=1e-15))
    print(f"LLZO log10(conductivity) recomputed:")
    print(f"  min={llzo_recomputed.min():.2f}, max={llzo_recomputed.max():.2f}, "
          f"  mean={llzo_recomputed.mean():.2f}, std={llzo_recomputed.std():.2f}")

# Sendek
sendek = pd.read_csv(str(DATA_CLEANED / "Sendek_clean.csv"))
print(f"\nSendek columns: {list(sendek.columns)}")
if 'log10_target' in sendek.columns:
    print(f"Sendek 'log10_target':")
    print(f"  min={sendek['log10_target'].min():.2f}, max={sendek['log10_target'].max():.2f}, "
          f"  mean={sendek['log10_target'].mean():.2f}, std={sendek['log10_target'].std():.2f}")
if 'IC' in sendek.columns:
    print(f"Sendek 'IC' (raw):")
    print(f"  min={sendek['IC'].min():.4e}, max={sendek['IC'].max():.4e}")
    sendek_log = np.log10(sendek['IC'].clip(lower=1e-15))
    print(f"Sendek log10(IC) recomputed:")
    print(f"  min={sendek_log.min():.2f}, max={sendek_log.max():.2f}, "
          f"  mean={sendek_log.mean():.2f}, std={sendek_log.std():.2f}")

# LiIon
liion = pd.read_csv(str(DATA_CLEANED / "LiIonDatabase_clean.csv"))
print(f"\nLiIon columns: {list(liion.columns)}")
print(f"LiIon 'log_target':")
print(f"  min={liion['log_target'].min():.2f}, max={liion['log_target'].max():.2f}, "
      f"  mean={liion['log_target'].mean():.2f}, std={liion['log_target'].std():.2f}")
if 'target' in liion.columns:
    liion_recomputed = np.log10(liion['target'].clip(lower=1e-15))
    print(f"LiIon log10(target) recomputed:")
    print(f"  min={liion_recomputed.min():.2f}, max={liion_recomputed.max():.2f}, "
          f"  mean={liion_recomputed.mean():.2f}, std={liion_recomputed.std():.2f}")
    # CHECK: is log_target actually log10 or ln?
    ratio = liion['log_target'].iloc[0] / liion_recomputed.iloc[0]
    print(f"  Ratio log_target / log10(target) for row 0: {ratio:.4f}")
    print(f"  (If ≈1.0 → same scale. If ≈2.303 → log_target is ln, not log10)")


# ── 2. CHECK WHAT THE MODEL PREDICTS ON VALIDATION SETS ───────────────
print("\n\n>>> CHECK 2: PREDICTION DISTRIBUTIONS <<<")
print("-" * 60)

metadata = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
pipeline = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
target_name = "log_Ionic_Conductivity"
feature_cols = metadata["feature_columns"][target_name]

# Load test set
test_df = pd.read_csv(str(MODELS_DIR / f"test_{target_name}.csv"))
X_test = test_df[feature_cols].values
y_test = test_df[target_name].values
y_pred_test = pipeline.predict(X_test)

print(f"DDSE Test Set predictions:")
print(f"  y_true: min={y_test.min():.2f}, max={y_test.max():.2f}, mean={y_test.mean():.2f}")
print(f"  y_pred: min={y_pred_test.min():.2f}, max={y_pred_test.max():.2f}, mean={y_pred_test.mean():.2f}")

# Now predict on validation — using minimal approach to isolate issues
from src.features.get_composition import parse_mixture_formula
from gensim.models import Word2Vec
from src.config import MAT2VEC_PRETRAINED
import src.features.get_composition as gc

m2v_model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
dim = 200

def predict_validation(df, formula_col, target_col, name):
    """Generate features and predict for a validation set."""
    y_true = df[target_col].dropna().values

    # Temperature
    if "Temp_K" not in df.columns and "temperature" in df.columns:
        df["Temp_K"] = df["temperature"]
    elif "Temp_K" not in df.columns:
        df["Temp_K"] = 298.0

    # Compositional features
    df_enh = gc.enhance_composition_features_fixed(df.copy(), formula_col)

    # Rename
    exclude_rename = [formula_col, "Temp_K", "doi", target_col,
                      "temperature", "target", "log_target",
                      "conductivity", "log10_target", "log10_predict",
                      "Ionic_Conductivity", "residue",
                      "ID", "source", "family", "ChemicalFamily",
                      "formula", "comp", "IC"]
    rename_map = {col: f"orig_{col}" for col in df_enh.columns if col not in exclude_rename}
    rename_map["Temp_K"] = "orig_Temp_K"
    df_model = df_enh.rename(columns=rename_map)

    # Mat2vec
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
        except:
            emb = np.zeros(dim)
        emb_list.append(emb)

    m2v_df = pd.DataFrame(np.array(emb_list), columns=[f"mat2vec_{i}" for i in range(dim)])
    df_model = pd.concat([df_model.reset_index(drop=True), m2v_df], axis=1)

    X = df_model.reindex(columns=feature_cols, fill_value=0)

    # Check for missing required columns
    present = [c for c in feature_cols if c in df_model.columns]
    missing = [c for c in feature_cols if c not in df_model.columns]
    filled_zero = len(missing)

    y_pred = pipeline.predict(X.values)

    print(f"\n{name}:")
    print(f"  Target '{target_col}': min={y_true.min():.2f}, max={y_true.max():.2f}, mean={y_true.mean():.2f}, std={y_true.std():.2f}")
    print(f"  Predictions:          min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}, std={y_pred.std():.2f}")
    print(f"  Features present: {len(present)}/{len(feature_cols)}, filled with 0: {filled_zero}")
    if missing:
        print(f"  Missing features (filled 0): {missing[:10]}...")

    # Check temperature
    temp_col = "orig_Temp_K" if "orig_Temp_K" in df_model.columns else None
    if temp_col:
        temps = df_model[temp_col]
        print(f"  Temperature: min={temps.min():.1f}, max={temps.max():.1f}, mean={temps.mean():.1f}")

    return y_true, y_pred

predict_validation(llzo.copy(), "compound", "log10_target", "LLZO")
predict_validation(liion.copy(), "composition", "log_target", "LiIon")
predict_validation(sendek.copy(), "comp", "log10_target", "Sendek")


# ── 3. CHECK TRAINING LOG TRANSFORM ──────────────────────────────────
print("\n\n>>> CHECK 3: LOG TRANSFORM VERIFICATION <<<")
print("-" * 60)

meta_log_info = metadata.get("log_transformation_info", {})
print(f"Metadata says: {meta_log_info}")
print(f"  'transformation' field: {meta_log_info.get('transformation')}")
print()
print("CRITICAL: If training used np.log10 but metadata says 'natural_log', ")
print("or if validation targets use a different base, we have a scale mismatch!")

# Verify by checking actual code path
sample_ic = 0.001  # 1e-3
log10_val = np.log10(sample_ic)
ln_val = np.log(sample_ic)
print(f"\n  For IC = {sample_ic}:")
print(f"    log10 = {log10_val:.4f}")
print(f"    ln    = {ln_val:.4f}")
print(f"    ratio ln/log10 = {ln_val/log10_val:.4f} (should be 2.3026 if different)")

# Check LiIon specifically
if 'target' in liion.columns:
    sample_idx = 0
    raw_target = liion['target'].iloc[sample_idx]
    stored_log = liion['log_target'].iloc[sample_idx]
    computed_log10 = np.log10(max(raw_target, 1e-15))
    computed_ln = np.log(max(raw_target, 1e-15))
    print(f"\n  LiIon row 0: raw={raw_target:.4e}")
    print(f"    stored log_target = {stored_log:.4f}")
    print(f"    computed log10    = {computed_log10:.4f}")
    print(f"    computed ln       = {computed_ln:.4f}")
    if abs(stored_log - computed_log10) < 0.01:
        print(f"    --> log_target IS log10 (matches)")
    elif abs(stored_log - computed_ln) < 0.01:
        print(f"    --> log_target IS ln (MISMATCH with model!)")
    else:
        print(f"    --> UNKNOWN scale!")


# ── 4. MATERIAL TYPE OVERLAP ─────────────────────────────────────────
print("\n\n>>> CHECK 4: MATERIAL TYPE OVERLAP <<<")
print("-" * 60)

if 'Material_Type' in ddse.columns:
    ddse_types = set(ddse['Material_Type'].dropna().unique())
    print(f"DDSE material types ({len(ddse_types)}): {sorted(ddse_types)}")

# Check what families are in validation sets
if 'family' in liion.columns:
    liion_families = set(liion['family'].dropna().unique())
    print(f"\nLiIon families ({len(liion_families)}): {sorted(liion_families)}")

# ── 5. FEATURE VALUE COMPARISON ───────────────────────────────────────
print("\n\n>>> CHECK 5: FEATURE DISTRIBUTION COMPARISON <<<")
print("-" * 60)

# Compare key feature distributions between DDSE and validation sets
# We already have ddse loaded with features
key_features = ['Temp_K', 'Ionic_Conductivity']
print(f"\n{'Feature':<25s} {'DDSE mean':>12s} {'DDSE std':>12s} {'LLZO mean':>12s} {'LiIon mean':>12s} {'Sendek mean':>12s}")
print("-" * 85)

for feat in key_features:
    ddse_mean = ddse[feat].mean() if feat in ddse.columns else float('nan')
    ddse_std = ddse[feat].std() if feat in ddse.columns else float('nan')

    llzo_val = float('nan')
    if feat == 'Temp_K' and 'temperature' in llzo.columns:
        llzo_val = llzo['temperature'].mean()
    elif feat in llzo.columns:
        llzo_val = llzo[feat].mean()
    elif feat == 'Ionic_Conductivity' and 'conductivity' in llzo.columns:
        llzo_val = llzo['conductivity'].mean()

    liion_val = float('nan')
    if feat == 'Temp_K' and 'temperature' in liion.columns:
        liion_val = liion['temperature'].mean()
    elif feat in liion.columns:
        liion_val = liion[feat].mean()
    elif feat == 'Ionic_Conductivity' and 'target' in liion.columns:
        liion_val = liion['target'].mean()

    sendek_val = float('nan')
    if feat in sendek.columns:
        sendek_val = sendek[feat].mean()
    elif feat == 'Ionic_Conductivity' and 'IC' in sendek.columns:
        sendek_val = sendek['IC'].mean()

    print(f"{feat:<25s} {ddse_mean:>12.4f} {ddse_std:>12.4f} {llzo_val:>12.4f} {liion_val:>12.4f} {sendek_val:>12.4f}")


# ── 6. CHECK SENDEK TARGET UNITS ────────────────────────────────────
print("\n\n>>> CHECK 6: SENDEK DATA DEEP DIVE <<<")
print("-" * 60)
print(f"Sendek first 10 rows:")
print(sendek.head(10).to_string())
print(f"\nSendek 'IC' stats: {sendek['IC'].describe()}")
if 'log10_target' in sendek.columns:
    print(f"\nSendek 'log10_target' stats: {sendek['log10_target'].describe()}")
    # Verify: is log10_target = log10(IC)?
    sendek_check = np.log10(sendek['IC'].clip(lower=1e-15))
    print(f"\nlog10(IC) vs stored log10_target (first 5):")
    for i in range(min(5, len(sendek))):
        print(f"  Row {i}: IC={sendek['IC'].iloc[i]:.4e}, "
              f"log10(IC)={sendek_check.iloc[i]:.4f}, "
              f"stored={sendek['log10_target'].iloc[i]:.4f}, "
              f"diff={abs(sendek_check.iloc[i] - sendek['log10_target'].iloc[i]):.4f}")
