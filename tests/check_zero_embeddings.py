"""Check how many DDSE entries produce zero-array mat2vec embeddings."""

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
sample_key = list(model.wv.key_to_index.keys())[0]
dim = len(model.wv[sample_key])
print(f"Mat2vec dim: {dim}")
print(f"Vocab size: {len(model.wv.key_to_index)}")

zero_formulas = []       # all elements missing → zero array
partial_formulas = []    # some elements missing
parse_errors = []        # formula couldn't be parsed

for i, formula in enumerate(df['electrolyte']):
    try:
        comp = Composition(str(formula).strip())
        elements = comp.get_el_amt_dict()

        found = []
        missing = []
        for element in elements:
            if element in model.wv:
                found.append(element)
            else:
                missing.append(element)

        if not found:
            zero_formulas.append((i, formula.strip(), list(elements.keys()), 'all_missing'))
        elif missing:
            partial_formulas.append((i, formula.strip(), missing, found))

    except Exception as e:
        parse_errors.append((i, formula, str(e)))

successful = len(df) - len(zero_formulas) - len(partial_formulas) - len(parse_errors)

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"  Full embeddings (all elements in vocab):   {successful}")
print(f"  Partial embeddings (some elements missing): {len(partial_formulas)}")
print(f"  Zero-array embeddings (no elements found):  {len(zero_formulas)}")
print(f"  Parse errors (formula unparseable):          {len(parse_errors)}")
print(f"  Total:                                       {len(df)}")

if zero_formulas:
    print(f"\n{'='*70}")
    print(f"ZERO-ARRAY ENTRIES ({len(zero_formulas)})")
    print(f"{'='*70}")
    for idx, formula, elems, reason in zero_formulas:
        print(f"  Row {idx:>5d}: {formula:<50s} elements={elems}")

if partial_formulas:
    print(f"\n{'='*70}")
    print(f"PARTIAL EMBEDDINGS ({len(partial_formulas)})")
    print(f"{'='*70}")

    # Aggregate missing elements
    all_missing = {}
    for idx, formula, missing, found in partial_formulas:
        for elem in missing:
            if elem not in all_missing:
                all_missing[elem] = []
            all_missing[elem].append(formula)

    print(f"\n  Unique missing elements: {sorted(all_missing.keys())}")
    print(f"\n  Missing element frequency:")
    for elem, formulas_list in sorted(all_missing.items(), key=lambda x: -len(x[1])):
        print(f"    {elem:>3s}: {len(formulas_list):>4d} formulas  (e.g. {formulas_list[0]})")

    print(f"\n  First 30 partial entries:")
    for idx, formula, missing, found in partial_formulas[:30]:
        print(f"    Row {idx:>5d}: {formula:<50s} missing={missing}  found={found}")

if parse_errors:
    print(f"\n{'='*70}")
    print(f"PARSE ERRORS ({len(parse_errors)})")
    print(f"{'='*70}")
    for idx, formula, err in parse_errors[:20]:
        print(f"  Row {idx:>5d}: {formula:<50s} error={err}")
