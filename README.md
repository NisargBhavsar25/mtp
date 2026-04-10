# MTP — Machine Learning for Ionic Conductivity Prediction in Solid-State Electrolytes

A machine learning pipeline for predicting **ionic conductivity** of solid-state lithium-ion electrolyte materials. The approach combines compositional feature engineering with [mat2vec](https://github.com/materialsintelligence/mat2vec) word embeddings and classical ML regressors to screen candidate materials at scale.

Trained on the Diverse Solid-State Electrolyte (DDSE) dataset (2,515 entries after cleaning), the models are validated against three independent datasets (LiIon, LLZO, Sendek) and applied to 2M+ compounds from the [Materials Project](https://materialsproject.org/).

---

## Project Structure

```
MTP/
├── src/                        # Source code
│   ├── config.py               # Centralized path configuration
│   ├── data_cleaning.py        # Data deduplication & leakage removal
│   ├── features/
│   │   └── get_composition.py  # Compositional feature extraction
│   ├── training/
│   │   ├── train_best_save.py  # Main training pipeline (RF, GB, XGB, Ridge)
│   │   ├── ddse_train.py       # GridSearchCV across 10+ algorithms
│   │   └── ddse_vec_train.py   # Mat2vec-only training
│   ├── inference/
│   │   └── predict_properties.py  # Predict IC for new compositions
│   ├── evaluation/
│   │   ├── validation_plots.py    # Parity, residual & Q-Q plots
│   │   └── calculate_metrics.py   # MAE, RMSE, MBE, R²_adj, STD
│   ├── analysis/
│   │   ├── data_analysis.py       # EDA with feature importance
│   │   └── data_visualizer.py     # Publication-quality figures
│   └── scraping/
│       ├── ddse.py             # Selenium scraper for DDSE database
│       └── mp_id.py            # Materials Project API queries
│
├── data/
│   ├── raw/                    # Original unprocessed source files
│   ├── cleaned/                # Deduplicated, leakage-free datasets
│   ├── processed/              # Feature-engineered intermediates
│   ├── results/                # Model prediction outputs
│   ├── embeddings/             # Precomputed mat2vec .npy arrays
│   ├── validation/             # Cross-dataset validation CSVs
│   └── mat2vec_models/         # Pretrained mat2vec .model files
│
├── models/                     # Trained model artifacts (.joblib)
├── notebooks/                  # Jupyter notebooks for EDA & cleaning
├── outputs/                    # Generated plots and figures
│   ├── acs_plots/              # ACS-style publication figures
│   ├── combined_regression_plots/
│   ├── regression_plots/
│   ├── visualizations/
│   ├── results/
│   └── misc_plots/
│
├── docs/                       # Papers, corrections, error summaries
├── Report/                     # LaTeX thesis/report
├── mat2vec/                    # mat2vec library (external)
├── archive/                    # Legacy/miscellaneous files
└── requirements.txt
```

---

## Installation

**Prerequisites:** Python 3.8+ with pip or conda.

```bash
# Clone the repository
git clone https://github.com/<your-username>/MTP.git && cd MTP

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Mat2vec setup
git clone https://github.com/materialsintelligence/mat2vec.git
cd mat2vec && pip install -e . && cd ..
```

### Environment Variables

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `MATERIALS_PROJECT_API_KEY` | For MP queries only | [Get your key here](https://materialsproject.org/api) |

The API key is only needed if you want to run `src/scraping/mp_id.py` to query the Materials Project database.

---

## Quick Start

All scripts are run from the project root directory.

### 1. Clean the data

Removes duplicates, resolves conflicting entries, and eliminates training/validation data leakage.

```bash
python -m src.data_cleaning
```

Outputs cleaned CSVs and an audit log to `data/cleaned/`.

### 2. Train models

Trains RandomForest, GradientBoosting, XGBoost, and Ridge regressors for log(IC) prediction, with optional mat2vec embeddings.

```bash
python -m src.training.train_best_save
```

Saves trained models to `models/`.

### 3. Predict properties

Predict ionic conductivity for new compositions.

```bash
python -m src.inference.predict_properties
```

### 4. Evaluate

Generate validation plots and compute metrics across datasets.

```bash
python -m src.evaluation.validation_plots
python -m src.evaluation.calculate_metrics
```

### 5. Visualize

Generate exploratory and publication-quality figures.

```bash
python src/analysis/data_visualizer.py
```

---

## Pipeline Overview

```
Data Collection ──> Cleaning ──> Feature Engineering ──> Training ──> Inference ──> Evaluation
    │                  │               │                    │             │             │
  DDSE DB         Dedup, leak      Compositional        RF, GB,       Predict       Parity,
  LiIon DB        removal,         descriptors +        XGB, Ridge    on new        residual,
  LLZO             conflict        mat2vec 200D         per target    formulas      Q-Q plots
  Sendek           aggregation     embeddings                                       + metrics
  Materials Proj
```

---

## Datasets

| Dataset | Source | Rows (cleaned) | Target Variables |
|---------|--------|---------------:|------------------|
| DDSE | Diverse Solid-State Electrolyte database | 2,515 | Ionic Conductivity |
| LiIon | Li-ion conductor literature compilation | 425 | Ionic Conductivity |
| LLZO | Li₇La₃Zr₂O₁₂ garnet variants | 117 | Ionic Conductivity |
| Sendek | Sendek et al. screening dataset | 39 | Ionic Conductivity |

The DDSE dataset serves as the primary training set. LiIon, LLZO, and Sendek are used for cross-dataset validation. See `data/cleaned/cleaning_audit_log.txt` for detailed cleaning statistics.

---

## Features

### Compositional Descriptors (from `get_composition.py`)

Extracted from chemical formulas using an extended periodic table:

- **Elemental statistics:** average atomic mass, electronegativity, ionic radius
- **Composition metrics:** number of elements, composition entropy, Li fraction
- **Derived features:** electronegativity variance, Li-to-anion ratio, packing efficiency proxy, mass-weighted electronegativity, ionic radius range, group diversity

### Mat2vec Embeddings

200-dimensional word embeddings trained on materials science literature. Each element in a formula is looked up in the pretrained Word2Vec model, and element vectors are combined via weighted averaging (weighted by stoichiometric coefficients).

---

## Models

The training pipeline evaluates multiple regression algorithms via GridSearchCV with 5-fold cross-validation:

| Algorithm | Library |
|-----------|---------|
| Random Forest | scikit-learn |
| Gradient Boosting | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| Ridge | scikit-learn |
| ElasticNet | scikit-learn |
| SVR | scikit-learn |
| KNN | scikit-learn |
| MLP | scikit-learn |

**Target:**
- `log_Ionic_Conductivity` — log₁₀(Ionic Conductivity) (S/cm)

Trained models are saved as `.joblib` files in `models/`.

---

## Key Dependencies

| Package | Role |
|---------|------|
| `scikit-learn` | ML models, pipelines, preprocessing, metrics |
| `pandas` / `numpy` | Data manipulation and numerical computing |
| `pymatgen` | Chemical formula parsing (`Composition`) |
| `gensim` | Loading pretrained Word2Vec mat2vec models |
| `mat2vec` | Materials text processing |
| `xgboost` / `lightgbm` | Gradient boosting regressors |
| `matplotlib` / `seaborn` | Visualization |
| `scipy` | Statistical analysis (Q-Q plots, distributions) |
| `joblib` | Model serialization |
| `mp-api` | Materials Project API queries |
| `selenium` | Web scraping for DDSE database |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `analysis.ipynb` | Comprehensive exploratory data analysis with ACS-style plots |
| `ddse_cleaning.ipynb` | Interactive data cleaning and inspection |
| `temp_analysis.ipynb` | Temperature-dependent conductivity analysis |

---

## Tests

Run the test suite with:

```bash
pytest tests/ -v
```

---

## License

This project is released for academic and research purposes. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this work in your research, please cite:

```
@misc{mtp2026,
  title={Machine Learning for Ionic Conductivity Prediction in Solid-State Electrolytes},
  year={2026},
  url={https://github.com/<your-username>/MTP}
}
```
