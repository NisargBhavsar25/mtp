"""Check zero embeddings using the ACTUAL training code path from train_best_save.py.

This replicates DDSEModelTrainer.generate_mat2vec_embeddings() exactly,
which catches parse errors and returns np.zeros(200) for them.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from pymatgen.core import Composition
from gensim.models import Word2Vec
from src.config import MAT2VEC_PRETRAINED

# Load data
df = pd.read_csv('data/cleaned/ddse_compositional_clean.csv')
print(f"Total entries: {len(df)}")

# Load mat2vec model
model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
dim = 200

# Replicate generate_mat2vec_embeddings() EXACTLY
zero_indices = []
partial_indices = []
success_indices = []

for i, formula in enumerate(df['electrolyte']):
    try:
        comp = Composition(str(formula))
        elements = comp.get_el_amt_dict()

        token_embeddings = []
        weights = []
        missing = []

        for element, amount in elements.items():
            if element in model.wv:
                token_embeddings.append(model.wv[element])
                weights.append(amount)
            else:
                missing.append(element)

        if token_embeddings:
            emb = np.average(token_embeddings, axis=0, weights=weights)
            if len(emb) != dim:
                resized = np.zeros(dim)
                m = min(len(emb), dim)
                resized[:m] = emb[:m]
                emb = resized

            if missing:
                partial_indices.append((i, formula.strip(), missing))
            else:
                success_indices.append(i)
        else:
            zero_indices.append((i, formula.strip(), 'no_elements_in_vocab'))

    except Exception as e:
        # This is the fallback path — returns zeros
        zero_indices.append((i, formula.strip(), f'exception: {str(e)[:80]}'))

print(f"\n{'='*70}")
print(f"TRAINING-PATH EMBEDDING RESULTS")
print(f"{'='*70}")
print(f"  Successful (non-zero embedding):  {len(success_indices):>5d}  ({len(success_indices)/len(df)*100:.1f}%)")
print(f"  Partial (some elements missing):  {len(partial_indices):>5d}  ({len(partial_indices)/len(df)*100:.1f}%)")
print(f"  ZERO ARRAYS (embedding = 0):      {len(zero_indices):>5d}  ({len(zero_indices)/len(df)*100:.1f}%)")
print(f"  Total:                            {len(df):>5d}")

# Categorize zero-array reasons
parse_errors = [z for z in zero_indices if 'exception' in z[2]]
vocab_missing = [z for z in zero_indices if z[2] == 'no_elements_in_vocab']

print(f"\n  Zero-array breakdown:")
print(f"    Parse errors (pymatgen can't parse): {len(parse_errors)}")
print(f"    Vocab miss (no elements in model):   {len(vocab_missing)}")

# Show examples by category
if parse_errors:
    print(f"\n{'='*70}")
    print(f"PARSE ERROR EXAMPLES (first 30 of {len(parse_errors)})")
    print(f"{'='*70}")

    # Categorize parse error types
    mixture_errors = []
    fraction_errors = []
    phase_errors = []
    other_errors = []

    for idx, formula, reason in parse_errors:
        if '-' in formula and any(c.isdigit() for c in formula.split('-')[0][-3:]):
            mixture_errors.append((idx, formula, reason))
        elif '/' in formula:
            fraction_errors.append((idx, formula, reason))
        elif formula.startswith(('alpha', 'beta', 'gamma')):
            phase_errors.append((idx, formula, reason))
        else:
            other_errors.append((idx, formula, reason))

    print(f"\n  By type:")
    print(f"    Mixture formulas (with '-'):  {len(mixture_errors)}")
    print(f"    Fraction formulas (with '/'):  {len(fraction_errors)}")
    print(f"    Phase prefix (alpha/beta):     {len(phase_errors)}")
    print(f"    Other:                         {len(other_errors)}")

    for idx, formula, reason in parse_errors[:30]:
        print(f"    Row {idx:>5d}: {formula:<55s} {reason}")

# Now test with the custom parser (parse_mixture_formula)
print(f"\n{'='*70}")
print(f"RE-TEST: Using parse_mixture_formula for failed entries")
print(f"{'='*70}")

from src.features.get_composition import parse_mixture_formula

recovered = 0
still_zero = 0
custom_errors = []

for idx, formula, reason in zero_indices:
    try:
        comp = parse_mixture_formula(formula)
        if not comp:
            still_zero += 1
            continue

        token_embeddings = []
        weights = []
        for element, amount in comp.items():
            if element in model.wv:
                token_embeddings.append(model.wv[element])
                weights.append(amount)

        if token_embeddings:
            recovered += 1
        else:
            still_zero += 1
    except Exception as e:
        custom_errors.append((idx, formula, str(e)[:80]))
        still_zero += 1

print(f"  Recovered with custom parser:    {recovered:>5d}")
print(f"  Still zero after custom parser:  {still_zero:>5d}")
print(f"  Custom parser errors:            {len(custom_errors):>5d}")

if custom_errors:
    print(f"\n  Custom parser failures:")
    for idx, formula, err in custom_errors[:10]:
        print(f"    Row {idx:>5d}: {formula:<55s} {err}")

total_nonzero = len(success_indices) + len(partial_indices) + recovered
print(f"\n{'='*70}")
print(f"SUMMARY: With custom parser, {total_nonzero}/{len(df)} ({total_nonzero/len(df)*100:.1f}%) would get non-zero embeddings")
print(f"         Irreducible zeros: {still_zero}/{len(df)} ({still_zero/len(df)*100:.1f}%)")
print(f"{'='*70}")
