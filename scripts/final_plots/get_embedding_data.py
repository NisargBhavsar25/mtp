"""Extract real mat2vec embedding values for infographic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from gensim.models import Word2Vec
from src.config import MAT2VEC_PRETRAINED
from src.features.get_composition import parse_mixture_formula

model = Word2Vec.load(str(MAT2VEC_PRETRAINED))

# Use Li6PS5Cl (argyrodite - very relevant to solid-state electrolytes)
formula = "Li6PS5Cl"
elements = parse_mixture_formula(formula)
print(f"Formula: {formula}")
print(f"Parsed: {elements}")
print()

# Show first 5 dims of each element embedding
for el, amt in elements.items():
    vec = model.wv[el]
    print(f"{el} (x{amt}): first 5 dims = {vec[:5].round(4)}")
    print(f"  full shape = {vec.shape}")

# Compute weighted average
vecs = []
weights = []
for el, amt in elements.items():
    vecs.append(model.wv[el])
    weights.append(amt)

result = np.average(vecs, axis=0, weights=weights)
print(f"\nWeighted average (first 5 dims): {result[:5].round(4)}")
print(f"Total weight: {sum(weights)}")

# Print more dims for the infographic
print("\n--- For infographic (first 8 dims) ---")
for el, amt in elements.items():
    vec = model.wv[el]
    vals = ", ".join([f"{v:.3f}" for v in vec[:8]])
    print(f"{el} (w={amt}): [{vals}, ...]")

vals = ", ".join([f"{v:.3f}" for v in result[:8]])
print(f"\nResult: [{vals}, ...]")
