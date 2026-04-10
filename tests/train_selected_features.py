import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_best_save import DDSEModelTrainer
from src.config import DATA_CLEANED

SELECTED_FEATURES = [
    "avg_ionic_radius",
    "li_fraction",
    "electronegativity_variance",
    "packing_efficiency_proxy",
    "heaviest_element_mass",
    "composition_entropy",
    "Temp_K",
]

csv_path = str(DATA_CLEANED / "ddse_compositional_clean.csv")
trainer = DDSEModelTrainer(csv_path, feature_subset=SELECTED_FEATURES)
trainer.train_and_save_best_models(use_sample_weights=False)
