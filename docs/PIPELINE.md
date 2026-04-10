# Pipeline Documentation

Detailed technical documentation for the MTP ionic conductivity prediction pipeline.

---

## 1. Data Collection

### Primary Training Data — DDSE

The Diverse Solid-State Electrolyte (DDSE) database is scraped using Selenium (`src/scraping/ddse.py`) from a Streamlit-hosted web app. Three JSON files are collected:

- `ddse_exp_ic.json` — experimental ionic conductivity measurements
- `ddse_exp_ea.json` — experimental activation energy measurements
- `ddse_exp_type.json` — electrolyte type classifications

These are merged into `ddse_original.csv`, then enriched with compositional features to produce `ddse_compositional.csv`.

### Validation Datasets

| Dataset | Source File | Description |
|---------|------------|-------------|
| LiIon | `LiIonDatabase.csv` / `.xlsx` | Literature compilation of Li-ion conductors |
| LLZO | `LLZO_compound.xlsx` | Li₇La₃Zr₂O₁₂ garnet-type electrolyte variants |
| Sendek | `Sendek_OP.csv` | Screening dataset from Sendek et al. |

### Materials Project

`src/scraping/mp_id.py` queries the Materials Project API for all Li-containing materials, extracting structure and stability data. This produces `MP_Li_Materials.csv` (~2M entries) used for large-scale screening.

---

## 2. Data Cleaning (`src/data_cleaning.py`)

The cleaning pipeline addresses three categories of data quality issues. All operations are logged to `data/cleaned/cleaning_audit_log.txt`.

### Exact Duplicate Removal

Rows that are identical across all columns (formula, temperature, conductivity, activation energy) are deduplicated, keeping the first occurrence.

- **DDSE:** 46 exact duplicates removed

### Conflicting Entry Aggregation

When the same formula appears at the same temperature but with different property values, entries are aggregated by taking the **median** of numeric columns. This is more robust to outliers than mean aggregation.

- **DDSE:** 20 conflicting groups → 33 excess rows removed
- **LiIon:** 15 conflicting groups → 18 excess rows removed
- **LLZO:** 22 conflicting groups → 58 excess rows removed

### Cross-Dataset Leakage Prevention

Chemical formulas appearing in both the DDSE training set and any validation set (LiIon, LLZO, Sendek) are removed from DDSE to ensure validation results reflect true generalization.

- **323 rows removed** (94 unique formulas)
- Removed entries saved separately in `ddse_leakage_removed.csv` for reference

### Junk Column Removal

Unnamed/empty columns (artifacts from Excel exports) are dropped from LLZO and Sendek datasets.

### Final Cleaned Dataset Sizes

| Dataset | Before | After |
|---------|-------:|------:|
| DDSE | 2,917 | 2,515 |
| LiIon | 443 | 425 |
| LLZO | 175 | 117 |
| Sendek | 39 | 39 |

---

## 3. Feature Engineering (`src/features/get_composition.py`)

Chemical formulas are parsed into numerical descriptors using a custom composition parser with an extended periodic table covering 60+ elements relevant to battery materials.

### Compositional Descriptors

| Feature | Description |
|---------|-------------|
| `avg_atomic_mass` | Weighted average atomic mass |
| `avg_electronegativity` | Weighted average Pauling electronegativity |
| `avg_ionic_radius` | Weighted average ionic radius |
| `num_elements` | Number of distinct elements |
| `li_fraction` | Molar fraction of lithium |
| `composition_entropy` | Shannon entropy of the composition (-Σ xᵢ ln xᵢ) |
| `electronegativity_variance` | Variance of electronegativities across elements |
| `li_to_anion_ratio` | Ratio of Li to anion (O, S, F, Cl, Br, I) content |
| `packing_efficiency_proxy` | Ratio of smallest to largest ionic radius |
| `mass_weighted_electronegativity` | Electronegativity weighted by atomic mass fraction |
| `ionic_radius_range` | Max minus min ionic radius |
| `group_diversity` | Number of distinct periodic table groups represented |

Additional raw features from the dataset: `Temp_K` (measurement temperature) and `Material_Type`.

### Mat2vec Embeddings

200-dimensional word embeddings from a Word2Vec model trained on ~2M materials science abstracts ([Tshitoyan et al., Nature 2019](https://doi.org/10.1038/s41586-019-1335-8)).

**Embedding generation process:**
1. Parse formula into elements using `pymatgen.core.Composition`
2. Look up each element token in the pretrained Word2Vec model
3. Compute weighted average of element vectors, weighted by stoichiometric coefficients
4. Result: one 200D vector per formula

The pretrained model files are stored in `data/mat2vec_models/` and the mat2vec library is in the `mat2vec/` directory.

---

## 4. Model Training

### Main Pipeline (`src/training/train_best_save.py`)

The `DDSEModelTrainer` class handles the full training workflow:

1. **Load and filter data** — only entries with Temp_K >= 293 K are kept
2. **Create log target** — `log_Ionic_Conductivity = log₁₀(Ionic_Conductivity)`
3. **Generate mat2vec embeddings** — 200D vectors appended as features
4. **Train/test split** — 80/20 stratified split (random_state=42)
5. **Pipeline construction** — `SimpleImputer → StandardScaler → Regressor`
6. **Model training** — trains on `log_Ionic_Conductivity` target
7. **Model selection** — evaluates RF, GB, XGBoost, Ridge, ElasticNet
8. **Save artifacts** — best model saved as `.joblib`, plus train/test CSVs and mat2vec metadata

### GridSearchCV Exploration (`src/training/ddse_train.py`)

A broader hyperparameter search across 10+ algorithms:
- Linear: LinearRegression, Ridge, Lasso, ElasticNet
- Tree-based: RandomForest, GradientBoosting, XGBoost, LightGBM
- Other: SVR, KNN, MLP

Uses 5-fold cross-validation with R², MAE, and RMSE as scoring metrics.

### Mat2vec-Only Training (`src/training/ddse_vec_train.py`)

Trains models using only mat2vec embeddings (no compositional features) to evaluate the standalone predictive power of materials word embeddings.

---

## 5. Inference (`src/inference/predict_properties.py`)

The `Mat2VecGenerator` class loads trained models and generates predictions for new compositions:

1. Parse input formula using `get_composition.py`
2. Generate mat2vec embedding
3. Combine compositional features + embedding
4. Apply trained pipeline (impute → scale → predict)
5. Output predicted `log₁₀(Ionic Conductivity)`

---

## 6. Evaluation

### Metrics (`src/evaluation/calculate_metrics.py`)

Computed for each validation dataset:

| Metric | Description |
|--------|-------------|
| N | Number of data points |
| R²_adj | Adjusted R-squared (accounts for number of predictors) |
| MTV | Mean true value |
| MPV | Mean predicted value |
| MAE | Mean absolute error |
| RMSE | Root mean square error |
| MBE | Mean bias error (systematic over/under-prediction) |
| STD | Standard deviation of residuals |

### Validation Plots (`src/evaluation/validation_plots.py`)

Generates four diagnostic plots for each validation run:

1. **Residual distribution (KDE)** — overlaid density plots for each dataset
2. **Parity plot** — predicted vs. actual with ideal fit line
3. **Residuals vs. predicted** — checks for heteroscedasticity
4. **Q-Q plot** — normality check of residuals

Also computes MAPE (Mean Absolute Percentage Error) and saves a summary CSV.

---

## 7. Analysis & Visualization

### Exploratory Data Analysis (`src/analysis/data_analysis.py`)

The `DDSEDataAnalyzer` class provides:
- Distribution analysis of all features
- Correlation analysis
- Feature importance via mutual information and random forest
- Material type breakdowns

### Publication Figures (`src/analysis/data_visualizer.py`)

Generates 12 publication-quality plots saved to `outputs/visualizations/`:
- Conductivity vs. activation energy (colored by material type)
- Conductivity distribution by material type (box plots)
- Correlation heatmap of electrolyte properties
- Conductivity vs. temperature, entropy, Li fraction
- Material type distribution (bar chart)
- Multi-panel relationship overview
- High-conductivity materials analysis
- Activation energy and ionic conductivity distributions
- Cumulative frequency distribution

---

## 8. File Naming Conventions

| Suffix | Meaning |
|--------|---------|
| `_clean.csv` | Cleaned dataset (deduplicated, leakage-free) |
| `_results.csv` | Model predictions appended to original data |
| `_summary.csv` | Aggregated metrics or statistics |
| `_OP.csv` / `_OP_py.csv` | OpenPhonon format / Python-processed |
| `.joblib` | Serialized scikit-learn model or metadata |
| `.npy` | NumPy array (precomputed embeddings) |
