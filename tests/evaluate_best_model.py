"""Evaluate the best trained model on train/test splits and validation datasets."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.config import MODELS_DIR, DATA_CLEANED
from src.evaluation.calculate_metrics import calculate_metrics


def main():
    # Load model and metadata
    metadata = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
    pipeline = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
    target = "log_Ionic_Conductivity"
    feature_cols = metadata["feature_columns"][target]
    config = metadata["model_configs"][target]

    print("=" * 80)
    print(f"MODEL: {config['best_model']} | Features: {config['features']}")
    print("=" * 80)

    # ── 1. Test set (saved during training) ───────────────────────────
    test_path = MODELS_DIR / f"test_{target}.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        X_test = test_df[feature_cols].values
        y_test = test_df[target].values

        y_pred = pipeline.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)

        print(f"\n--- DDSE Test Set (20% hold-out) ---")
        print(f"  N = {metrics['N']}")
        print(f"  R_adj^2 = {metrics['R_adj^2']:.4f}")
        print(f"  MAE     = {metrics['MAE']:.4f}")
        print(f"  RMSE    = {metrics['RMSE']:.4f}")
        print(f"  MBE     = {metrics['MBE']:.4f}")
        print(f"  STD     = {metrics['STD']:.4f}")
    else:
        print(f"\nTest set not found at {test_path}")

    # ── 2. Full DDSE dataset (train + test, recomputed) ──────────────
    print("\n--- Full DDSE Dataset (recomputed predictions) ---")
    from src.features.get_composition import parse_mixture_formula
    from gensim.models import Word2Vec
    from src.config import MAT2VEC_PRETRAINED

    df = pd.read_csv(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
    df = df[df["Temp_K"] >= 293].copy()

    ic = df["Ionic_Conductivity"].copy().replace(0, 1e-12)
    ic[ic <= 0] = 1e-12
    df["log_Ionic_Conductivity"] = np.log10(ic)

    exclude = ["electrolyte", "doi", "Ea_eV", "Ionic_Conductivity", "log_Ionic_Conductivity"]
    orig_features = [c for c in df.columns if c not in exclude and df[c].dtype in ["int64", "float64", "bool"]]
    orig_data = df[orig_features].fillna(df[orig_features].median())

    m2v_model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
    dim = 200
    embeddings = []
    for formula in df["electrolyte"]:
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
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(dim))
        except Exception:
            embeddings.append(np.zeros(dim))
    embeddings = np.array(embeddings)

    orig_cols = [f"orig_{c}" for c in orig_features]
    m2v_cols = [f"mat2vec_{i}" for i in range(dim)]
    X_full = np.hstack([orig_data.values, embeddings])
    X_df = pd.DataFrame(X_full, columns=orig_cols + m2v_cols)
    X_df = X_df.reindex(columns=feature_cols, fill_value=0)

    y_full = df["log_Ionic_Conductivity"].values
    y_pred_full = pipeline.predict(X_df.values)

    metrics_full = calculate_metrics(y_full, y_pred_full)
    print(f"  N = {metrics_full['N']}")
    print(f"  R_adj^2 = {metrics_full['R_adj^2']:.4f}")
    print(f"  MAE     = {metrics_full['MAE']:.4f}")
    print(f"  RMSE    = {metrics_full['RMSE']:.4f}")
    print(f"  MBE     = {metrics_full['MBE']:.4f}")
    print(f"  STD     = {metrics_full['STD']:.4f}")

    # ── 3. Cross-dataset validation ──────────────────────────────────
    # These datasets have log10_target and log10_predict columns from
    # prior predictions. We re-predict using the current model.
    import src.features.get_composition as gc

    validation_sets = [
        ("LLZO", DATA_CLEANED / "LLZO_clean.csv", "compound", "log10_target"),
        ("Sendek", DATA_CLEANED / "Sendek_clean.csv", "comp", "log10_target"),
        ("LiIon", DATA_CLEANED / "LiIonDatabase_clean.csv", "composition", "log_target"),
    ]

    print("\n" + "=" * 80)
    print("CROSS-DATASET VALIDATION")
    print("=" * 80)

    all_results = []

    for name, csv_path, formula_col, target_col in validation_sets:
        if not csv_path.exists():
            print(f"\n--- {name}: file not found, skipping ---")
            continue

        vdf = pd.read_csv(csv_path)

        if target_col not in vdf.columns:
            print(f"\n--- {name}: target column '{target_col}' not found, skipping ---")
            continue

        y_true = vdf[target_col].values

        # Add Temp_K if missing
        if "Temp_K" not in vdf.columns and "temperature" in vdf.columns:
            vdf["Temp_K"] = vdf["temperature"]
        elif "Temp_K" not in vdf.columns:
            vdf["Temp_K"] = 298.0

        # Generate compositional features
        vdf_enhanced = gc.enhance_composition_features_fixed(vdf, formula_col)

        # Rename to orig_ prefix
        rename_map = {col: f"orig_{col}" for col in vdf_enhanced.columns
                      if col not in [formula_col, "Temp_K", "doi", target_col,
                                     "temperature", "target", "log_target",
                                     "conductivity", "log10_target", "log10_predict",
                                     "Ionic_Conductivity", "residue",
                                     "ID", "source", "family", "ChemicalFamily",
                                     "formula", "comp", "IC"]}
        rename_map["Temp_K"] = "orig_Temp_K"
        vdf_model = vdf_enhanced.rename(columns=rename_map)

        # Generate mat2vec embeddings
        m2v_dict = {}
        for idx, row in vdf_enhanced.iterrows():
            try:
                elements = parse_mixture_formula(str(row[formula_col]))
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
            for i, val in enumerate(emb):
                col_name = f"mat2vec_{i}"
                if col_name not in m2v_dict:
                    m2v_dict[col_name] = []
                m2v_dict[col_name].append(val)

        for col, values in m2v_dict.items():
            vdf_model[col] = values

        # Align columns and predict
        X_val = vdf_model.reindex(columns=feature_cols, fill_value=0)

        # Remove NaN targets
        mask = ~np.isnan(y_true)
        y_true_clean = y_true[mask]
        X_val_clean = X_val.values[mask]

        y_pred_val = pipeline.predict(X_val_clean)
        metrics_val = calculate_metrics(y_true_clean, y_pred_val)

        print(f"\n--- {name} Dataset ---")
        print(f"  N = {metrics_val['N']}")
        print(f"  R_adj^2 = {metrics_val['R_adj^2']:.4f}")
        print(f"  MAE     = {metrics_val['MAE']:.4f}")
        print(f"  RMSE    = {metrics_val['RMSE']:.4f}")
        print(f"  MBE     = {metrics_val['MBE']:.4f}")
        print(f"  STD     = {metrics_val['STD']:.4f}")

        all_results.append({"Dataset": name, **metrics_val})

    # ── Summary table ────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        summary = pd.DataFrame(all_results)
        cols = ["Dataset", "N", "R_adj^2", "MAE", "RMSE", "MBE", "STD"]
        summary = summary[[c for c in cols if c in summary.columns]]
        print(summary.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
