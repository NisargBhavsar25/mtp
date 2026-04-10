"""Train a model with all 14 physical descriptors + mat2vec for SHAP analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.training.train_best_save import DDSEModelTrainer
from src.config import DATA_CLEANED, MODELS_DIR

FEATURES_14 = [
    "Temp_K",
    "avg_electronegativity",
    "avg_atomic_mass",
    "avg_ionic_radius",
    "num_elements",
    "li_fraction",
    "composition_entropy",
    "electronegativity_variance",
    "group_diversity",
    "packing_efficiency_proxy",
    "li_to_anion_ratio",
    "heaviest_element_mass",
    "lightest_element_mass",
    "total_atoms",
]

save_dir = str(MODELS_DIR / "model_14feat")

trainer = DDSEModelTrainer(
    str(DATA_CLEANED / "ddse_compositional_clean.csv"),
    feature_subset=FEATURES_14,
)
trainer.train_and_save_best_models(save_dir=save_dir, use_sample_weights=False)
