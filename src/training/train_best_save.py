import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
import os
from src.config import DATA_CLEANED, MODELS_DIR, MAT2VEC_PRETRAINED
from src.features.get_composition import parse_mixture_formula
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
import warnings
warnings.filterwarnings('ignore')

# Mat2vec imports
try:
    from pymatgen.core import Composition
    from mat2vec.processing import MaterialsTextProcessor
    from gensim.models import Word2Vec
    from xgboost import XGBRegressor
    MAT2VEC_AVAILABLE = True
    XGBOOST_AVAILABLE = True
    print("✅ All required libraries available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install mat2vec xgboost pymatgen")
    # For safety, define flags if imports fail so class can still load
    MAT2VEC_AVAILABLE = False
    XGBOOST_AVAILABLE = False

class DDSEModelTrainer:
    def __init__(self, csv_file_path, random_state=42, feature_subset=None):
        """Initialize trainer for log(Ionic_Conductivity) prediction.

        Parameters
        ----------
        feature_subset : list[str] or None
            If provided, only these column names are used as original features.
            All must exist in the CSV. If None, all numeric columns are used
            (the original behaviour).
        """
        self.df = pd.read_csv(csv_file_path)
        # filter out data points with Temp<293
        self.df = self.df[self.df['Temp_K'] >= 293]
        self.random_state = random_state

        # Create log(Ionic_Conductivity) target
        self._prepare_log_targets()

        # Target variable
        self.targets = ['log_Ionic_Conductivity']

        # Original feature columns
        if feature_subset is not None:
            missing = [c for c in feature_subset if c not in self.df.columns]
            if missing:
                raise ValueError(f"Features not found in data: {missing}")
            self.original_features = list(feature_subset)
        else:
            exclude_cols = ['electrolyte', 'doi', 'Ea_eV', 'Ionic_Conductivity', 'log_Ionic_Conductivity']
            self.original_features = [col for col in self.df.columns
                                    if col not in exclude_cols and
                                    self.df[col].dtype in ['int64', 'float64', 'bool']]
        
        print(f"Loaded {len(self.df)} samples with {len(self.original_features)} original features")
        print(f"Targets: {self.targets}")
        
        # Initialize Mat2vec
        self.mat2vec_processor = None
        self.mat2vec_model = None
        self.embedding_dim = 200 # FIXED: Changed from 300 to 200 to match paper
        self._initialize_mat2vec()
        
        # Store models and preprocessing objects
        self.trained_models = {}
        self.feature_columns = {}  # Different features for each target
        
    def _prepare_log_targets(self):
        """Create log(Ionic_Conductivity) target with proper handling of zero/negative values"""
        
        print("🔄 Preparing log(Ionic_Conductivity) target...")
        
        if 'Ionic_Conductivity' not in self.df.columns:
            raise ValueError("Ionic_Conductivity column not found in dataset")
        
        # Get ionic conductivity values
        ic_values = self.df['Ionic_Conductivity'].copy()
        
        # Handle zero and negative values for log transformation
        ic_for_log = ic_values.copy()
        ic_for_log = ic_for_log.replace(0, 1e-12)  # Replace exact zeros
        ic_for_log[ic_for_log <= 0] = 1e-12  # Replace negative values
        
        # Create log target
        self.df['log_Ionic_Conductivity'] = np.log10(ic_for_log)
        
        print(f"   ✅ Created log_Ionic_Conductivity with {self.df['log_Ionic_Conductivity'].notna().sum()} valid values")

    def _initialize_mat2vec(self):
        """Initialize Mat2vec processor and model"""
        try:
            self.mat2vec_processor = MaterialsTextProcessor()
            self.mat2vec_model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
            
            # Check embedding dimension
            sample_key = list(self.mat2vec_model.wv.key_to_index.keys())[0]
            actual_dim = len(self.mat2vec_model.wv[sample_key])
            
            # Warn if mismatch, but use our self.embedding_dim for vectors we generate
            if actual_dim != self.embedding_dim:
                 print(f"⚠️ Warning: Pretrained model has {actual_dim} dims, but target is {self.embedding_dim}. Vectors will be padded/truncated or zeroed.")

            print(f"✅ Mat2vec initialized (Targeting {self.embedding_dim}D embeddings)")
        except Exception as e:
            print(f"❌ Mat2vec initialization failed: {e}")
            # Do not raise here, allow fallback to non-mat2vec models

    def generate_mat2vec_embeddings(self, formulas=None):
        """Generate Mat2vec embeddings using element parsing (FIXED LOGIC)"""
        
        if not MAT2VEC_AVAILABLE or self.mat2vec_model is None:
             print("❌ Mat2vec not loaded. Returning zeros.")
             return np.zeros((len(self.df), self.embedding_dim))

        if formulas is None:
            formulas = self.df['electrolyte']
        
        embeddings = []
        successful_count = 0
        
        print(f"🔄 Generating Mat2vec embeddings for {len(formulas)} formulas...")
        
        for formula in formulas:
            try:
                # 1. PARSE: Use custom parser that handles mixtures, fractions, phase prefixes
                elements = parse_mixture_formula(str(formula))

                token_embeddings = []
                weights = []

                # 2. LOOKUP: Iterate through elements, not the full string
                for element, amount in elements.items():
                    if element in self.mat2vec_model.wv:
                        embedding = self.mat2vec_model.wv[element]
                        token_embeddings.append(embedding)
                        weights.append(amount)
                        
                if token_embeddings:
                    # Calculate weighted average based on stoichiometry
                    # Note: We must ensure the pretrained vector matches our desired dim
                    # If pretrained is 200, great. If 100, we might need to handle it.
                    # Assuming pretrained matches or we just use what it gives:
                    formula_embedding = np.average(token_embeddings, axis=0, weights=weights)
                    
                    # Force resize to self.embedding_dim if needed (simple truncation/padding)
                    if len(formula_embedding) != self.embedding_dim:
                         resized = np.zeros(self.embedding_dim)
                         min_dim = min(len(formula_embedding), self.embedding_dim)
                         resized[:min_dim] = formula_embedding[:min_dim]
                         formula_embedding = resized

                    embeddings.append(formula_embedding)
                    successful_count += 1
                else:
                    embeddings.append(np.zeros(self.embedding_dim))
                    
            except Exception as e:
                embeddings.append(np.zeros(self.embedding_dim))
        
        embeddings_array = np.array(embeddings)
        print(f"✅ Generated embeddings: {embeddings_array.shape}, Success: {successful_count}/{len(formulas)}")
        
        return embeddings_array
    
    @staticmethod
    def compute_inverse_frequency_weights(y, n_bins=20):
        """Compute inverse-frequency sample weights based on target distribution.

        Bins the target into `n_bins` equal-width bins, then assigns each sample
        a weight proportional to 1/bin_count.  Weights are normalised so they
        sum to len(y) (i.e. the effective dataset size stays the same).
        """
        bin_edges = np.linspace(y.min(), y.max(), n_bins + 1)
        # np.digitize returns 1-indexed; clip to [1, n_bins]
        bin_indices = np.clip(np.digitize(y, bin_edges), 1, n_bins)

        bin_counts = np.bincount(bin_indices, minlength=n_bins + 1)
        # Avoid division by zero for empty bins
        bin_counts = np.maximum(bin_counts, 1)

        raw_weights = 1.0 / bin_counts[bin_indices]
        # Normalise so weights sum to N
        weights = raw_weights * len(y) / raw_weights.sum()
        return weights

    def train_and_save_best_models(self, save_dir=None, use_sample_weights=False):
        """
        Dynamically train multiple models on multiple feature sets,
        find the best performing one for each target, and save it.
        """
        
        import os
        if save_dir is None:
            save_dir = str(MODELS_DIR)
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("🚀 AUTOMATED MODEL SELECTION & SAVING")
        print("="*80)

        # 1. Generate Mat2vec embeddings once
        mat2vec_embeddings = self.generate_mat2vec_embeddings()
        
        # 2. Prepare Original Data
        original_data = self.df[self.original_features].fillna(self.df[self.original_features].median())
        
        # 3. Define Combinations to Test
        feature_combinations = [
            {'name': 'Original_Only', 'use_mat2vec': False, 'use_original': True},
            {'name': 'Mat2vec_Only', 'use_mat2vec': True, 'use_original': False},
            {'name': 'Original_Mat2vec_Combined', 'use_mat2vec': True, 'use_original': True}
        ]

        # 4. Define Models to Test
        models_to_test = {
            'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=self.random_state),
            'Ridge': Ridge(alpha=0.5),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=self.random_state)
        }
        if XGBOOST_AVAILABLE:
            models_to_test['XGBoost'] = XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=self.random_state)

        # Dictionary to store the winning configuration for metadata
        best_configs_metadata = {}

        # 5. Main Loop: For each target -> Test all combos -> Save Winner
        for target in self.targets:
            if target not in self.df.columns:
                print(f"⚠️ Target {target} not found, skipping.")
                continue

            print(f"\n🎯 Processing Target: {target}")
            print("-" * 40)

            y = self.df[target].copy()
            valid_idx = y.notna()
            y_clean = y[valid_idx]
            
            if len(y_clean) < 10:
                print("❌ Too few samples.")
                continue

            best_score = -np.inf
            best_model_name = None
            best_feature_name = None
            best_pipeline = None
            best_feature_columns = []

            # Loop through feature combinations
            for combo in feature_combinations:
                # Construct X matrix
                matrices = []
                col_names = []
                
                if combo['use_original']:
                    matrices.append(original_data.values)
                    col_names.extend([f"orig_{col}" for col in self.original_features])
                
                if combo['use_mat2vec']:
                    matrices.append(mat2vec_embeddings)
                    col_names.extend([f"mat2vec_{i}" for i in range(mat2vec_embeddings.shape[1])])
                
                if not matrices: continue

                X_full = np.hstack(matrices)
                X_clean = X_full[valid_idx]
                
                # Split: train (80%) / test (20%)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=self.random_state
                )

                # Compute sample weights for training set if requested
                sample_weights = None
                if use_sample_weights:
                    sample_weights = self.compute_inverse_frequency_weights(y_train.values)

                # Loop through models
                for model_name, model_inst in models_to_test.items():
                    try:
                        pipeline = Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('model', model_inst)
                        ])

                        fit_params = {}
                        if sample_weights is not None:
                            fit_params["model__sample_weight"] = sample_weights
                        pipeline.fit(X_train, y_train, **fit_params)
                        y_pred = pipeline.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Print progress
                        print(f"   Testing {combo['name']} + {model_name}: R² = {r2:.3f}")

                        # Check if winner
                        if r2 > best_score:
                            best_score = r2
                            best_model_name = model_name
                            best_feature_name = combo['name']
                            best_pipeline = pipeline
                            best_feature_columns = col_names
                            best_X_test = X_test
                            best_y_test = y_test

                            # Save Test Set for this specific winning combo (optional, overwrites previous best)
                            test_df = pd.DataFrame(X_test, columns=col_names)
                            test_df[target] = y_test.values
                            test_df.to_csv(os.path.join(save_dir, f"test_{target}.csv"), index=False)

                    except Exception as e:
                        print(f"   ❌ Error with {model_name}: {e}")

            # End of loops for this target
            if best_pipeline:
                print(f"\n🏆 WINNER for {target}:")
                print(f"   Model: {best_model_name}")
                print(f"   Features: {best_feature_name}")
                print(f"   R² Score: {best_score:.3f}")
                
                # Save the model
                model_path = os.path.join(save_dir, f"ddse_model_{target}.joblib")
                joblib.dump(best_pipeline, model_path)
                print(f"   Saved model to {model_path}")

                # ── Error interval from residual std on test set ──
                test_preds = best_pipeline.predict(best_X_test)
                residuals = best_y_test.values - test_preds
                residual_std = np.std(residuals, ddof=1)
                residual_mean = np.mean(residuals)

                error_info = {
                    'residual_std': residual_std,
                    'residual_mean': residual_mean,
                    'n_test': len(residuals),
                }
                error_path = os.path.join(save_dir, f"error_stats_{target}.joblib")
                joblib.dump(error_info, error_path)
                print(f"   Saved error stats to {error_path}")
                print(f"   Test residual std: {residual_std:.4f} log10(S/cm)")
                print(f"   Test residual mean: {residual_mean:.4f}")
                print(f"   ~68% interval: +/-{residual_std:.4f}")
                print(f"   ~95% interval: +/-{1.96 * residual_std:.4f}")

                # Store info for class and metadata
                self.feature_columns[target] = best_feature_columns
                best_configs_metadata[target] = {
                    'best_model': best_model_name,
                    'features': best_feature_name,
                    'score': best_score
                }
            else:
                print(f"❌ Could not find a working model for {target}")

        # 6. Save Metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'original_features': self.original_features,
            'embedding_dim': self.embedding_dim,
            'targets': self.targets,
            'random_state': self.random_state,
            'model_configs': best_configs_metadata, # Now dynamic!
            'log_transformation_info': {
                'target': 'log_Ionic_Conductivity',
                'original_target': 'Ionic_Conductivity',
                'zero_replacement': 1e-12,
                'transformation': 'log10'
            },
            'sample_weighting': 'inverse_frequency' if use_sample_weights else 'none'
        }
        
        metadata_path = os.path.join(save_dir, "model_metadata.joblib")
        joblib.dump(metadata, metadata_path)
        print(f"\n✅ Saved metadata to: {metadata_path}")
        
        # Save Mat2vec info
        mat2vec_info = {
            'processor': self.mat2vec_processor,
            'model_path': str(MAT2VEC_PRETRAINED),
            'embedding_dim': self.embedding_dim
        }
        mat2vec_path = os.path.join(save_dir, "mat2vec_info.joblib")
        joblib.dump(mat2vec_info, mat2vec_path)
        print(f"✅ Saved Mat2vec info to: {mat2vec_path}")

# Main execution
def main():
    print("🚀 STARTING AUTOMATED TRAINING RUN")
    print("="*60)
    
    # Initialize trainer
    # Ensure 'ddse_compositional.csv' exists or change path
    try:
        trainer = DDSEModelTrainer(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
        # Train and save the best models dynamically
        trainer.train_and_save_best_models()
    except FileNotFoundError:
        print("❌ Error: 'ddse_compositional.csv' not found. Please check file path.")

if __name__ == "__main__":
    main()