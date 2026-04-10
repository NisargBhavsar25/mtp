"""Table 2: Test results on 20% unseen DDSE data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import joblib

from src.config import MODELS_DIR

metadata = joblib.load(str(MODELS_DIR / "model_metadata.joblib"))
pipeline = joblib.load(str(MODELS_DIR / "ddse_model_log_Ionic_Conductivity.joblib"))
target = "log_Ionic_Conductivity"
feat_cols = metadata["feature_columns"][target]

test_df = pd.read_csv(str(MODELS_DIR / f"test_{target}.csv"))
X_test = test_df[feat_cols].values
y_test = test_df[target].values
y_pred = pipeline.predict(X_test)

errors = y_pred - y_test
n = len(y_test)
ss_res = np.sum(errors**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - ss_res / ss_tot
r2_adj = 1 - ((1 - r2) * (n - 1) / (n - 2))

mtv = np.mean(y_test)
mpv = np.mean(y_pred)
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
mbe = np.mean(errors)
std = np.std(errors, ddof=1)

print("Table 2: Test results on 20% of unseen data of DDSE dataset (Internal test)")
print()
print(f"Data set\tN\tR²_adj\tMTV\tMPV\tMAE\tRMSE\tMBE\tSTD")
print(f"DDSE\t{n}\t{r2_adj:.2f}\t{mtv:.2f}\t{mpv:.2f}\t{mae:.2f}\t{rmse:.2f}\t{mbe:.2f}\t{std:.2f}")
print()
print("--- Copy-paste friendly (tab-separated) ---")
print()
print(f"DDSE\t{n}\t{r2_adj:.2f}\t{mtv:.2f}\t{mpv:.2f}\t{mae:.2f}\t{rmse:.2f}\t{mbe:.2f}\t{std:.2f}")
