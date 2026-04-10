"""
SHAP Analysis for the best trained model.
==========================================
Runs TreeExplainer on the winning XGBoost pipeline, aggregates the 200
mat2vec dimensions into a single "Mat2Vec_Embedding" feature, and produces:

1. Mean |SHAP| bar chart
2. Beeswarm plot
3. CSV of SHAP feature importance values

Usage:
    python -m src.analysis.shap_analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

from src.config import MODELS_DIR, OUTPUTS_DIR, DATA_CLEANED, journal_savefig

# ── output directory ──────────────────────────────────────────────────
SHAP_DIR = str(OUTPUTS_DIR / "shap_analysis")
os.makedirs(SHAP_DIR, exist_ok=True)


def load_model_and_data():
    """Load the trained pipeline, metadata, and rebuild full feature matrix."""
    metadata = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
    pipeline = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))

    target = "log_Ionic_Conductivity"
    feature_cols = metadata["feature_columns"][target]
    config = metadata["model_configs"][target]

    print(f"Model: {config['best_model']}")
    print(f"Features: {config['features']}")
    print(f"Training R2: {config['score']:.3f}")
    print(f"Total feature columns: {len(feature_cols)}")

    # Rebuild the full dataset with embeddings to get SHAP on all data
    from src.features.get_composition import parse_mixture_formula
    from gensim.models import Word2Vec
    from src.config import MAT2VEC_PRETRAINED

    df = pd.read_csv(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
    df = df[df["Temp_K"] >= 293].copy()

    # log target
    ic = df["Ionic_Conductivity"].copy()
    ic = ic.replace(0, 1e-12)
    ic[ic <= 0] = 1e-12
    df["log_Ionic_Conductivity"] = np.log10(ic)

    y = df[target].values

    # Original features
    exclude = ["electrolyte", "doi", "Ea_eV", "Ionic_Conductivity", "log_Ionic_Conductivity"]
    orig_features = [c for c in df.columns if c not in exclude and df[c].dtype in ["int64", "float64", "bool"]]
    orig_data = df[orig_features].fillna(df[orig_features].median())

    # Mat2vec embeddings
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
                    r[: min(len(emb), dim)] = emb[: min(len(emb), dim)]
                    emb = r
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(dim))
        except Exception:
            embeddings.append(np.zeros(dim))
    embeddings = np.array(embeddings)

    # Build combined matrix matching training column order
    orig_cols = [f"orig_{c}" for c in orig_features]
    m2v_cols = [f"mat2vec_{i}" for i in range(dim)]
    X_full = np.hstack([orig_data.values, embeddings])
    all_cols = orig_cols + m2v_cols

    X_df = pd.DataFrame(X_full, columns=all_cols)
    # Align to trained feature order
    X_df = X_df.reindex(columns=feature_cols, fill_value=0)

    print(f"Dataset shape: {X_df.shape}")
    print(f"Target samples: {len(y)}")

    return pipeline, X_df, y, feature_cols, orig_cols, m2v_cols


def run_shap(pipeline, X_df):
    """Run SHAP TreeExplainer on the underlying XGBoost model."""
    # Extract the fitted model from the pipeline
    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    # Transform features through imputer + scaler (same as pipeline does)
    X_imputed = imputer.transform(X_df.values)
    X_scaled = scaler.transform(X_imputed)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

    print("Running SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)
    print(f"SHAP values shape: {shap_values.shape}")

    return shap_values, X_scaled_df


def aggregate_mat2vec_shap(shap_values, X_scaled_df, feature_cols, m2v_cols):
    """
    Collapse 200 mat2vec SHAP columns into one 'Mat2Vec_Embedding' feature.

    SHAP value for the aggregate = sum of individual mat2vec SHAP values per sample.
    Feature value for the aggregate = L2 norm of the mat2vec vector (for colouring).
    """
    m2v_mask = np.array([c in m2v_cols for c in feature_cols])
    orig_mask = ~m2v_mask

    # Aggregated SHAP: sum across mat2vec dims per sample
    shap_m2v_agg = shap_values[:, m2v_mask].sum(axis=1, keepdims=True)
    shap_orig = shap_values[:, orig_mask]
    shap_agg = np.hstack([shap_orig, shap_m2v_agg])

    # Feature values for colouring: keep originals, use L2 norm for mat2vec
    fval_m2v_agg = np.linalg.norm(X_scaled_df.values[:, m2v_mask], axis=1, keepdims=True)
    fval_orig = X_scaled_df.values[:, orig_mask]
    fval_agg = np.hstack([fval_orig, fval_m2v_agg])

    # Column names
    orig_names = [c for c, m in zip(feature_cols, orig_mask) if m]
    agg_names = orig_names + ["Mat2Vec_Embedding"]

    # Clean up display names: strip "orig_" prefix
    display_names = []
    for n in agg_names:
        if n.startswith("orig_"):
            display_names.append(n[5:])
        else:
            display_names.append(n)

    return shap_agg, fval_agg, agg_names, display_names


def generate_plots_and_csv(shap_agg, fval_agg, display_names):
    """Generate bar chart, beeswarm, and CSV."""

    n_features = shap_agg.shape[1]

    # ── 1. Mean |SHAP| values ────────────────────────────────────────
    mean_abs_shap = np.mean(np.abs(shap_agg), axis=0)
    importance_df = pd.DataFrame({
        "Feature": display_names,
        "Mean_|SHAP|": mean_abs_shap,
    }).sort_values("Mean_|SHAP|", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE (Mean |SHAP|)")
    print("=" * 60)
    print(importance_df.to_string(index=False))

    importance_df.to_csv(os.path.join(SHAP_DIR, "shap_feature_importance.csv"), index=False)

    # ── 2. Bar chart ─────────────────────────────────────────────────
    sorted_df = importance_df.sort_values("Mean_|SHAP|", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sorted_df["Feature"], sorted_df["Mean_|SHAP|"], color="steelblue")
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title("SHAP Feature Importance for log$_{10}$(Ionic Conductivity)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    journal_savefig(os.path.join(SHAP_DIR, "shap_bar_chart.png"))
    plt.close()
    print(f"\nSaved: shap_bar_chart.png")

    # ── 3. Beeswarm plot ─────────────────────────────────────────────
    # Sort features by importance for display
    sort_idx = np.argsort(mean_abs_shap)[::-1]  # descending
    shap_sorted = shap_agg[:, sort_idx]
    fval_sorted = fval_agg[:, sort_idx]
    names_sorted = [display_names[i] for i in sort_idx]

    # Build a shap.Explanation object for the beeswarm
    explanation = shap.Explanation(
        values=shap_sorted,
        data=fval_sorted,
        feature_names=names_sorted,
    )

    fig, ax = plt.subplots(figsize=(12, 9))
    shap.plots.beeswarm(explanation, max_display=n_features, show=False)
    ax = plt.gca()
    ax.set_title(
        "SHAP Beeswarm — Feature Impact on log$_{10}$(Ionic Conductivity)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    journal_savefig(os.path.join(SHAP_DIR, "shap_beeswarm.png"))
    plt.close()
    print(f"Saved: shap_beeswarm.png")

    # ── 4. Summary bar via SHAP library ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(explanation, max_display=n_features, show=False)
    ax = plt.gca()
    ax.set_title(
        "SHAP Feature Importance (Mean |SHAP|)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    journal_savefig(os.path.join(SHAP_DIR, "shap_bar_summary.png"))
    plt.close()
    print(f"Saved: shap_bar_summary.png")

    print(f"\nAll outputs saved to: {SHAP_DIR}")

    return importance_df


def generate_raw_plots(shap_values, X_scaled_df, feature_cols):
    """Generate plots with all 216 features (mat2vec dimensions NOT aggregated).

    Only the top 30 features by mean |SHAP| are shown to keep the plots
    readable.
    """
    print("\n" + "=" * 60)
    print("RAW SHAP (all 216 features, top 30 shown)")
    print("=" * 60)

    # Clean display names
    display_names = []
    for c in feature_cols:
        if c.startswith("orig_"):
            display_names.append(c[5:])
        else:
            display_names.append(c)

    max_display = 30

    # Mean |SHAP| ranking
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    raw_df = pd.DataFrame({
        "Feature": display_names,
        "Mean_|SHAP|": mean_abs,
    }).sort_values("Mean_|SHAP|", ascending=False).reset_index(drop=True)

    print(f"Top {max_display} features:")
    print(raw_df.head(max_display).to_string(index=False))
    raw_df.to_csv(os.path.join(SHAP_DIR, "shap_feature_importance_raw.csv"), index=False)

    # Sort for plotting
    sort_idx = np.argsort(mean_abs)[::-1]
    shap_sorted = shap_values[:, sort_idx]
    fval_sorted = X_scaled_df.values[:, sort_idx]
    names_sorted = [display_names[i] for i in sort_idx]

    explanation = shap.Explanation(
        values=shap_sorted,
        data=fval_sorted,
        feature_names=names_sorted,
    )

    # ── Beeswarm (top 30) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_title(
        "SHAP Beeswarm — All Features (Top 30)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    journal_savefig(os.path.join(SHAP_DIR, "shap_beeswarm_raw.png"))
    plt.close()
    print(f"Saved: shap_beeswarm_raw.png")

    # ── Bar chart (top 30) ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.bar(explanation, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_title(
        "SHAP Feature Importance — All Features (Top 30)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    journal_savefig(os.path.join(SHAP_DIR, "shap_bar_raw.png"))
    plt.close()
    print(f"Saved: shap_bar_raw.png")

    return raw_df


def generate_physical_only_plots(shap_values, X_scaled_df, feature_cols):
    """Generate SHAP plots for physical/compositional features only.

    Excludes Temp_K and all mat2vec dimensions.
    """
    print("\n" + "=" * 60)
    print("SHAP — Physical Features Only (excl. Mat2Vec)")
    print("=" * 60)

    # Select physical features (all orig_* columns including Temp_K)
    keep_idx = []
    keep_names = []
    for i, c in enumerate(feature_cols):
        if c.startswith("orig_"):
            keep_idx.append(i)
            keep_names.append(c[5:])  # strip orig_ prefix

    keep_idx = np.array(keep_idx)
    shap_phys = shap_values[:, keep_idx]
    fval_phys = X_scaled_df.values[:, keep_idx]

    # Sort by mean |SHAP|
    mean_abs = np.mean(np.abs(shap_phys), axis=0)
    sort_idx = np.argsort(mean_abs)[::-1]
    shap_sorted = shap_phys[:, sort_idx]
    fval_sorted = fval_phys[:, sort_idx]
    names_sorted = [keep_names[i] for i in sort_idx]

    phys_df = pd.DataFrame({
        "Feature": names_sorted,
        "Mean_|SHAP|": mean_abs[sort_idx],
    }).reset_index(drop=True)
    print(phys_df.to_string(index=False))
    phys_df.to_csv(os.path.join(SHAP_DIR, "shap_physical_features.csv"), index=False)

    explanation = shap.Explanation(
        values=shap_sorted,
        data=fval_sorted,
        feature_names=names_sorted,
    )

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=len(keep_names), show=False)
    ax = plt.gca()
    ax.set_title(
        "SHAP Beeswarm — Physical Features (excl. Temperature)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    journal_savefig(os.path.join(SHAP_DIR, "shap_beeswarm_physical.png"))
    plt.close()
    print("Saved: shap_beeswarm_physical.png")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(explanation, max_display=len(keep_names), show=False)
    ax = plt.gca()
    ax.set_title(
        "SHAP Feature Importance — Physical Features (excl. Temperature)",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    journal_savefig(os.path.join(SHAP_DIR, "shap_bar_physical.png"))
    plt.close()
    print("Saved: shap_bar_physical.png")

    return phys_df


def main():
    print("=" * 60)
    print("SHAP ANALYSIS — Best Model")
    print("=" * 60)

    pipeline, X_df, y, feature_cols, orig_cols, m2v_cols = load_model_and_data()
    shap_values, X_scaled_df = run_shap(pipeline, X_df)

    # ── Aggregated version (mat2vec collapsed) ────────────────────────
    shap_agg, fval_agg, agg_names, display_names = aggregate_mat2vec_shap(
        shap_values, X_scaled_df, feature_cols, m2v_cols
    )
    importance_df = generate_plots_and_csv(shap_agg, fval_agg, display_names)

    # ── Raw version (all 216 features individually) ───────────────────
    raw_df = generate_raw_plots(shap_values, X_scaled_df, feature_cols)

    # ── Physical features only (no Temp_K, no Mat2Vec) ────────────────
    phys_df = generate_physical_only_plots(shap_values, X_scaled_df, feature_cols)

    return importance_df, raw_df, phys_df


if __name__ == "__main__":
    main()
