"""Check if parse_mixture_formula handles validation dataset formulas correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.features.get_composition import parse_mixture_formula
from src.config import DATA_CLEANED

# Sample formulas from each dataset
datasets = [
    ("LLZO", DATA_CLEANED / "LLZO_clean.csv", "compound"),
    ("Sendek", DATA_CLEANED / "Sendek_clean.csv", "comp"),
    ("LiIon", DATA_CLEANED / "LiIonDatabase_clean.csv", "composition"),
    ("DDSE", DATA_CLEANED / "ddse_compositional_clean.csv", "electrolyte"),
]

for name, path, col in datasets:
    df = pd.read_csv(path)
    formulas = df[col].tolist()

    print("=" * 80)
    print(f"  {name} — column '{col}' — {len(formulas)} formulas")
    print("=" * 80)

    empty_count = 0
    error_count = 0
    success_count = 0
    empty_examples = []

    for i, f in enumerate(formulas):
        try:
            result = parse_mixture_formula(str(f))
            if not result or all(v == 0 for v in result.values()):
                empty_count += 1
                if len(empty_examples) < 10:
                    empty_examples.append((i, f, result))
            else:
                success_count += 1
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"  ERROR row {i}: '{f}' -> {e}")

    print(f"  Success: {success_count}/{len(formulas)}")
    print(f"  Empty/zero: {empty_count}/{len(formulas)}")
    print(f"  Errors: {error_count}/{len(formulas)}")

    if empty_examples:
        print(f"\n  Empty parse examples:")
        for idx, formula, result in empty_examples:
            print(f"    Row {idx}: '{formula}' -> {result}")

    # Show first 5 parsed results for sanity
    print(f"\n  First 5 parsed:")
    for i in range(min(5, len(formulas))):
        try:
            r = parse_mixture_formula(str(formulas[i]))
            print(f"    '{formulas[i]}' -> {dict(list(r.items())[:6])}{'...' if len(r) > 6 else ''}")
        except Exception as e:
            print(f"    '{formulas[i]}' -> ERROR: {e}")
    print()
