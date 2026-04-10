import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.training.train_best_save import DDSEModelTrainer
from src.config import DATA_CLEANED

csv_path = str(DATA_CLEANED / "ddse_compositional_clean.csv")

# Pre-add 1/T column to the CSV data so the trainer picks it up
df = pd.read_csv(csv_path)
df["inv_Temp_K"] = 1.0 / df["Temp_K"]
tmp_csv = str(DATA_CLEANED / "ddse_compositional_clean_inv_T.csv")
df.to_csv(tmp_csv, index=False)

SELECTED_FEATURES = [
    "avg_ionic_radius",
    "li_fraction",
    "electronegativity_variance",
    "packing_efficiency_proxy",
    "heaviest_element_mass",
    "composition_entropy",
    "inv_Temp_K",
]

trainer = DDSEModelTrainer(tmp_csv, feature_subset=SELECTED_FEATURES)
trainer.train_and_save_best_models(use_sample_weights=False)
