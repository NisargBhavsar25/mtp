import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.train_best_save import DDSEModelTrainer
from src.config import DATA_CLEANED

trainer = DDSEModelTrainer(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
trainer.train_and_save_best_models()
