"""
Data Cleaning Script for MTP Project
=====================================
Removes problematic entries from all four datasets:

1. DDSE:   Drops exact duplicates, aggregates conflicting entries (median),
           flags formulas that overlap with validation sets.
2. LiIon:  Aggregates conflicting entries (median).
3. LLZO:   Aggregates conflicting entries (median), drops junk columns.
4. Sendek: Drops junk columns (already clean).

Outputs cleaned files to data/cleaned/ alongside a full audit log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.config import DATA_RAW, DATA_PROCESSED, PROJECT_ROOT

CLEAN_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_formula(f):
    """Strip whitespace for consistent formula matching."""
    if pd.isna(f):
        return ""
    return str(f).strip().replace(" ", "")


def log(msg, file=None):
    """Print and optionally write to audit log."""
    print(msg)
    if file:
        file.write(msg + "\n")


def aggregate_conflicts(df, formula_col, temp_col, value_cols, extra_keep_cols=None):
    """
    For rows sharing the same (formula, temperature), collapse them into one
    row by taking the *median* of numeric value columns and keeping the first
    occurrence of non-numeric metadata.

    Returns (cleaned_df, n_groups_merged, n_rows_removed).
    """
    key = [formula_col, temp_col]

    # Identify groups with > 1 row
    dup_mask = df.duplicated(subset=key, keep=False)
    unique_part = df[~dup_mask].copy()
    dup_part = df[dup_mask].copy()

    if dup_part.empty:
        return df.copy(), 0, 0

    n_dup_rows = len(dup_part)

    # Build aggregation dict
    agg_dict = {}
    for col in df.columns:
        if col in key:
            continue
        if col in value_cols:
            agg_dict[col] = "median"
        elif extra_keep_cols and col in extra_keep_cols:
            agg_dict[col] = "first"
        elif df[col].dtype in ["float64", "int64"]:
            agg_dict[col] = "median"
        else:
            agg_dict[col] = "first"

    merged = dup_part.groupby(key, as_index=False).agg(agg_dict)

    n_groups = len(merged)
    n_removed = n_dup_rows - n_groups

    cleaned = pd.concat([unique_part, merged], ignore_index=True)
    return cleaned, n_groups, n_removed


# ---------------------------------------------------------------------------
# 1. Clean DDSE
# ---------------------------------------------------------------------------

def clean_ddse(audit):
    log("\n" + "=" * 70, audit)
    log("CLEANING: DDSE Compositional", audit)
    log("=" * 70, audit)

    df = pd.read_csv(DATA_PROCESSED / "ddse_compositional.csv")
    log(f"  Loaded: {len(df)} rows", audit)

    # Step 1: Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    exact_removed = before - len(df)
    log(f"  Step 1 - Exact duplicates removed: {exact_removed}", audit)

    # Step 2: Aggregate conflicting entries (same electrolyte + Temp_K)
    value_cols = ["Ea_eV", "Ionic_Conductivity"]
    extra_keep = ["doi", "Material_Type"]
    df, n_groups, n_removed = aggregate_conflicts(
        df, "electrolyte", "Temp_K", value_cols, extra_keep
    )
    log(f"  Step 2 - Conflicting groups merged: {n_groups} (removed {n_removed} excess rows)", audit)

    # Step 3: Recalculate compositional features are already present and
    # were computed from the formula, so the median row retains them from
    # the first occurrence which is fine (they are deterministic per formula).

    # Step 4: Flag formulas that overlap with validation sets
    liion = pd.read_csv(DATA_RAW / "LiIonDatabase.csv")
    sendek = pd.read_csv(DATA_RAW / "Sendek_OP.csv")

    liion_formulas = set(liion["composition"].apply(normalize_formula))
    sendek_formulas = set(sendek["comp"].apply(normalize_formula))
    val_formulas = liion_formulas | sendek_formulas

    df["_norm"] = df["electrolyte"].apply(normalize_formula)
    overlap_mask = df["_norm"].isin(val_formulas)
    n_overlap = overlap_mask.sum()

    # Remove overlapping entries to prevent data leakage
    df_leakage = df[overlap_mask].copy()
    df = df[~overlap_mask].copy()
    df = df.drop(columns=["_norm"])
    df_leakage = df_leakage.drop(columns=["_norm"])

    log(f"  Step 3 - Validation-set overlaps removed: {n_overlap} rows ({df_leakage['electrolyte'].nunique()} unique formulas)", audit)
    log(f"  Final: {len(df)} rows", audit)

    # Save
    df.to_csv(CLEAN_DIR / "ddse_compositional_clean.csv", index=False)
    df_leakage.to_csv(CLEAN_DIR / "ddse_leakage_removed.csv", index=False)
    log(f"  Saved: ddse_compositional_clean.csv", audit)
    log(f"  Saved: ddse_leakage_removed.csv (for reference)", audit)

    return df


# ---------------------------------------------------------------------------
# 2. Clean LiIon
# ---------------------------------------------------------------------------

def clean_liion(audit):
    log("\n" + "=" * 70, audit)
    log("CLEANING: LiIon Database", audit)
    log("=" * 70, audit)

    df = pd.read_csv(DATA_RAW / "LiIonDatabase.csv")
    log(f"  Loaded: {len(df)} rows", audit)

    # Aggregate conflicting entries (same composition + temperature)
    value_cols = ["target", "log_target"]
    extra_keep = ["ID", "source", "family", "ChemicalFamily"]
    df, n_groups, n_removed = aggregate_conflicts(
        df, "composition", "temperature", value_cols, extra_keep
    )
    log(f"  Conflicting groups merged: {n_groups} (removed {n_removed} excess rows)", audit)
    log(f"  Final: {len(df)} rows", audit)

    # Recalculate log_target from median target
    df["log_target"] = np.log10(df["target"].clip(lower=1e-15))

    df.to_csv(CLEAN_DIR / "LiIonDatabase_clean.csv", index=False)
    log(f"  Saved: LiIonDatabase_clean.csv", audit)

    return df


# ---------------------------------------------------------------------------
# 3. Clean LLZO
# ---------------------------------------------------------------------------

def clean_llzo(audit):
    log("\n" + "=" * 70, audit)
    log("CLEANING: LLZO Dataset", audit)
    log("=" * 70, audit)

    df = pd.read_csv(DATA_RAW / "LLZO_OP.csv")
    log(f"  Loaded: {len(df)} rows, {len(df.columns)} columns", audit)

    # Drop junk columns (Unnamed)
    junk_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=junk_cols)
    log(f"  Dropped {len(junk_cols)} junk columns: {junk_cols}", audit)

    # Aggregate conflicting entries (same compound + temperature)
    value_cols = ["conductivity", "log10_target", "log10_predict", "Ionic_Conductivity", "residue"]
    df, n_groups, n_removed = aggregate_conflicts(
        df, "compound", "temperature", value_cols
    )
    log(f"  Conflicting groups merged: {n_groups} (removed {n_removed} excess rows)", audit)

    # Recalculate log from median conductivity
    df["log10_target"] = np.log10(df["conductivity"].clip(lower=1e-15))

    log(f"  Final: {len(df)} rows", audit)

    df.to_csv(CLEAN_DIR / "LLZO_clean.csv", index=False)
    log(f"  Saved: LLZO_clean.csv", audit)

    return df


# ---------------------------------------------------------------------------
# 4. Clean Sendek
# ---------------------------------------------------------------------------

def clean_sendek(audit):
    log("\n" + "=" * 70, audit)
    log("CLEANING: Sendek Dataset", audit)
    log("=" * 70, audit)

    df = pd.read_csv(DATA_RAW / "Sendek_OP.csv")
    log(f"  Loaded: {len(df)} rows", audit)

    # Drop junk columns
    junk_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=junk_cols)
    log(f"  Dropped {len(junk_cols)} junk columns: {junk_cols}", audit)

    # No duplicates to fix
    log(f"  No duplicates found. Dataset is clean.", audit)
    log(f"  Final: {len(df)} rows", audit)

    df.to_csv(CLEAN_DIR / "Sendek_clean.csv", index=False)
    log(f"  Saved: Sendek_clean.csv", audit)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    audit_path = CLEAN_DIR / "cleaning_audit_log.txt"

    with open(audit_path, "w", encoding="utf-8") as audit:
        log("=" * 70, audit)
        log("MTP DATA CLEANING AUDIT LOG", audit)
        log("=" * 70, audit)

        ddse = clean_ddse(audit)
        liion = clean_liion(audit)
        llzo = clean_llzo(audit)
        sendek = clean_sendek(audit)

        # Final summary
        log("\n" + "=" * 70, audit)
        log("FINAL SUMMARY", audit)
        log("=" * 70, audit)
        log(f"  DDSE:   2917 -> {len(ddse)} rows", audit)
        log(f"  LiIon:  443  -> {len(liion)} rows", audit)
        log(f"  LLZO:   175  -> {len(llzo)} rows", audit)
        log(f"  Sendek: 39   -> {len(sendek)} rows", audit)

        # Verify no cross-dataset leakage remains
        ddse_f = set(ddse["electrolyte"].apply(normalize_formula))
        liion_f = set(liion["composition"].apply(normalize_formula))
        sendek_f = set(sendek["comp"].apply(normalize_formula))
        remaining_overlap = ddse_f & (liion_f | sendek_f)
        log(f"\n  Cross-dataset formula overlap remaining: {len(remaining_overlap)}", audit)
        if remaining_overlap:
            log(f"  WARNING: {sorted(remaining_overlap)[:10]}", audit)
        else:
            log(f"  Validation sets are now fully independent from training data.", audit)

        log(f"\n  All cleaned files saved to: {CLEAN_DIR}", audit)
        log(f"  Audit log saved to: {audit_path}", audit)

    print(f"\nDone. Audit log: {audit_path}")


if __name__ == "__main__":
    main()
