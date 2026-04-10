import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predict_properties import ConductivityPredictor

predictor = ConductivityPredictor()
predictor.predict_csv("tests/quick_input.csv", "tests/quick_output.csv")
