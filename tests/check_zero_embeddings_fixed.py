"""Verify the fix: re-run zero-embedding check using parse_mixture_formula
(the same parser now used in train_best_save.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from src.config import MAT2VEC_PRETRAINED
from src.features.get_composition import parse_mixture_formula

df = pd.read_csv('data/cleaned/ddse_compositional_clean.csv')
model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
dim = 200

zero_count = 0
success_count = 0
zero_entries = []

for i, formula in enumerate(df['electrolyte']):
    try:
        elements = parse_mixture_formula(str(formula))
        token_embeddings = []
        weights = []
        for element, amount in elements.items():
            if element in model.wv:
                token_embeddings.append(model.wv[element])
                weights.append(amount)

        if token_embeddings:
            success_count += 1
        else:
            zero_count += 1
            zero_entries.append((i, formula.strip(), list(elements.keys())))
    except Exception as e:
        zero_count += 1
        zero_entries.append((i, formula.strip() if isinstance(formula, str) else str(formula), f'error: {e}'))

print(f"Total entries:     {len(df)}")
print(f"Non-zero embeddings: {success_count} ({success_count/len(df)*100:.1f}%)")
print(f"Zero embeddings:     {zero_count} ({zero_count/len(df)*100:.1f}%)")

if zero_entries:
    print(f"\nRemaining zero-array entries:")
    for idx, formula, info in zero_entries:
        print(f"  Row {idx}: {formula}  -> {info}")
else:
    print("\nNo zero-array embeddings. All formulas parsed successfully.")
