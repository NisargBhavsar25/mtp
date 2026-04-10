import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np; from gensim.models import Word2Vec; from src.config import MAT2VEC_PRETRAINED
from src.features.get_composition import parse_mixture_formula
model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
elements = parse_mixture_formula("Li6PS5Cl")
vecs, weights = [], []
for el, amt in elements.items():
    v = model.wv[el]
    print(f"{el} d199={v[198]:.3f}  d200={v[199]:.3f}")
    vecs.append(v); weights.append(amt)
result = np.average(vecs, axis=0, weights=weights)
print(f"Result d199={result[198]:.3f}  d200={result[199]:.3f}")