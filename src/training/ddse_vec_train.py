import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_CLEANED, MAT2VEC_PRETRAINED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Mat2vec imports only
try:
    from mat2vec.processing import MaterialsTextProcessor
    from gensim.models import Word2Vec
    MAT2VEC_AVAILABLE = True
    print("✅ Mat2vec available")
except ImportError:
    MAT2VEC_AVAILABLE = False
    print("❌ Mat2vec not available. Install with: pip install mat2vec")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class Mat2vecOnlyTrainer:
    def __init__(self, csv_file_path, random_state=42):
        """Initialize trainer with Mat2vec only - predicting log(Ionic_Conductivity)"""
        self.df = pd.read_csv(csv_file_path)
        self.random_state = random_state
        self.embeddings_data = {}
        self.results = {}
        
        # Create log(Ionic_Conductivity) target
        self._prepare_log_targets()
        
        # Target variable
        self.targets = ['log_Ionic_Conductivity']
        
        # Original feature columns - exclude both original and log targets
        exclude_cols = ['electrolyte', 'doi', 'Ea_eV', 'Ionic_Conductivity', 'log_Ionic_Conductivity']
        self.original_features = [col for col in self.df.columns 
                                if col not in exclude_cols and 
                                self.df[col].dtype in ['int64', 'float64', 'bool']]
        
        print(f"Loaded {len(self.df)} samples with {len(self.original_features)} original features")
        print(f"Targets: {self.targets}")
        
        # Initialize Mat2vec only
        self.mat2vec_processor = None
        self.mat2vec_model = None
        self.embedding_dim = 300  # Standard Mat2vec dimension
        self._initialize_mat2vec()
    
    def _prepare_log_targets(self):
        """Create log(Ionic_Conductivity) target with proper handling of zero/negative values"""
        
        print("🔄 Preparing log(Ionic_Conductivity) target...")
        
        if 'Ionic_Conductivity' not in self.df.columns:
            raise ValueError("Ionic_Conductivity column not found in dataset")
        
        # Get ionic conductivity values
        ic_values = self.df['Ionic_Conductivity'].copy()
        
        # Check for problematic values
        zero_count = (ic_values == 0).sum()
        negative_count = (ic_values < 0).sum()
        missing_count = ic_values.isna().sum()
        
        print(f"   Original IC values: {len(ic_values)} total")
        print(f"   Zero values: {zero_count}")
        print(f"   Negative values: {negative_count}")
        print(f"   Missing values: {missing_count}")
        
        # Handle zero and negative values for log transformation
        # Option 1: Replace zeros with small positive value
        # Option 2: Set zeros/negatives to NaN
        
        # Using Option 1: Replace with small positive value (1e-12)
        ic_for_log = ic_values.copy()
        ic_for_log = ic_for_log.replace(0, 1e-12)  # Replace exact zeros
        ic_for_log[ic_for_log <= 0] = 1e-12  # Replace negative values
        
        # Create log target (base 10)
        self.df['log_Ionic_Conductivity'] = np.log10(ic_for_log)
        
        # Count valid log values
        valid_log_count = self.df['log_Ionic_Conductivity'].notna().sum()
        
        print(f"   ✅ Created log_Ionic_Conductivity with {valid_log_count} valid values")
        print(f"   Log IC range: {self.df['log_Ionic_Conductivity'].min():.3f} to {self.df['log_Ionic_Conductivity'].max():.3f}")
    
    def _initialize_mat2vec(self):
        """Initialize Mat2vec processor and model"""
        
        if not MAT2VEC_AVAILABLE:
            raise ImportError("Mat2vec is required but not available")
        
        try:
            # Initialize processor
            self.mat2vec_processor = MaterialsTextProcessor()
            print("✅ Mat2vec processor initialized")
            
            # Load pretrained model
            try:
                self.mat2vec_model = Word2Vec.load(str(MAT2VEC_PRETRAINED))
                print("✅ Mat2vec pretrained model loaded")
                
                # Get actual embedding dimension from model
                sample_key = list(self.mat2vec_model.wv.key_to_index.keys())[0]
                self.embedding_dim = len(self.mat2vec_model.wv[sample_key])
                print(f"✅ Mat2vec embedding dimension: {self.embedding_dim}D")
                
            except Exception as e:
                print(f"❌ Failed to load Mat2vec pretrained model: {e}")
                print("📥 Please ensure the pretrained model is downloaded from:")
                print("   https://storage.googleapis.com/mat2vec/pretrained_embeddings")
                raise
                
        except Exception as e:
            print(f"❌ Mat2vec initialization failed: {e}")
            raise
    
    def generate_mat2vec_embeddings(self):
        """Generate Mat2vec embeddings with consistent dimensions"""
        
        if not self.mat2vec_processor or not self.mat2vec_model:
            print("❌ Mat2vec not properly initialized")
            return None
        
        print("🔄 Generating Mat2vec embeddings (fixed dimension handling)...")
        
        mat2vec_features = []
        successful_count = 0
        failed_formulas = []
        token_stats = {'found': 0, 'not_found': 0}
        
        for i, formula in enumerate(self.df['electrolyte']):
            try:
                formula_str = str(formula).strip()
                
                # Process formula using Mat2vec
                processed_tokens, replacements = self.mat2vec_processor.process(formula_str)
                
                # Get embeddings for each token
                token_embeddings = []
                for token in processed_tokens:
                    try:
                        # Try different token formats
                        embedding = None
                        
                        # Try lowercase
                        if token.lower() in self.mat2vec_model.wv:
                            embedding = self.mat2vec_model.wv[token.lower()]
                            token_stats['found'] += 1
                        # Try original case
                        elif token in self.mat2vec_model.wv:
                            embedding = self.mat2vec_model.wv[token]
                            token_stats['found'] += 1
                        # Try normalized formula if it's a chemical formula
                        elif hasattr(self.mat2vec_processor, 'normalized_formula'):
                            try:
                                normalized = self.mat2vec_processor.normalized_formula(token)
                                if normalized in self.mat2vec_model.wv:
                                    embedding = self.mat2vec_model.wv[normalized]
                                    token_stats['found'] += 1
                            except:
                                pass
                        
                        if embedding is not None:
                            # Ensure consistent dimension
                            if len(embedding) == self.embedding_dim:
                                token_embeddings.append(embedding)
                            else:
                                print(f"⚠️ Dimension mismatch for token '{token}': {len(embedding)} vs {self.embedding_dim}")
                        else:
                            token_stats['not_found'] += 1
                            
                    except Exception as token_error:
                        token_stats['not_found'] += 1
                        continue
                
                if token_embeddings:
                    # Average token embeddings for formula representation
                    formula_embedding = np.mean(token_embeddings, axis=0)
                    
                    # Double-check dimension consistency
                    if len(formula_embedding) == self.embedding_dim:
                        mat2vec_features.append(formula_embedding)
                        successful_count += 1
                    else:
                        print(f"⚠️ Formula embedding dimension error: {len(formula_embedding)}")
                        mat2vec_features.append(np.zeros(self.embedding_dim))
                        failed_formulas.append(formula_str)
                else:
                    # No valid tokens found, use zero vector
                    mat2vec_features.append(np.zeros(self.embedding_dim))
                    failed_formulas.append(formula_str)
                
                # Progress tracking
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1}/{len(self.df)} formulas...")
                
            except Exception as e:
                mat2vec_features.append(np.zeros(self.embedding_dim))
                failed_formulas.append(str(formula))
                if i < 5:  # Show first few errors
                    print(f"Mat2vec error for '{formula}': {e}")
        
        # Convert to numpy array (should work now with consistent dimensions)
        try:
            mat2vec_array = np.array(mat2vec_features)
            
            print(f"✅ Generated Mat2vec embeddings: {mat2vec_array.shape}")
            print(f"✅ Success rate: {successful_count}/{len(self.df)} ({successful_count/len(self.df)*100:.1f}%)")
            print(f"📊 Token stats: Found={token_stats['found']}, Not found={token_stats['not_found']}")
            
            if len(failed_formulas) > 0:
                print(f"⚠️ Failed formulas: {min(len(failed_formulas), 5)} shown")
                for formula in failed_formulas[:5]:
                    print(f"   - {formula}")
            
            return mat2vec_array
            
        except Exception as array_error:
            print(f"❌ Array creation failed: {array_error}")
            print("🔍 Debugging embedding shapes...")
            
            # Debug: Check shapes of first 10 embeddings
            for i, emb in enumerate(mat2vec_features[:10]):
                print(f"   Embedding {i}: shape={np.array(emb).shape}, type={type(emb)}")
            
            return None
    
    def train_mat2vec_models(self):
        """Train models with Mat2vec embeddings - now for Ea_eV and log(Ionic_Conductivity)"""
        
        print("\n" + "="*80)
        print("🚀 TRAINING WITH MAT2VEC EMBEDDINGS - log(Ionic_Conductivity)")
        print("="*80)
        
        # Generate Mat2vec embeddings
        mat2vec_embeddings = self.generate_mat2vec_embeddings()
        
        if mat2vec_embeddings is None:
            print("❌ Failed to generate Mat2vec embeddings")
            return None
        
        self.embeddings_data['mat2vec'] = mat2vec_embeddings
        
        # Get original features
        original_data = self.df[self.original_features].fillna(self.df[self.original_features].median())
        
        # Test different combinations
        feature_combinations = [
            {'name': 'Original_Only', 'use_mat2vec': False, 'use_original': True},
            {'name': 'Mat2vec_Only', 'use_mat2vec': True, 'use_original': False},
            {'name': 'Original_Mat2vec_Combined', 'use_mat2vec': True, 'use_original': True}
        ]
        
        all_results = []
        
        for combo in feature_combinations:
            print(f"\n🧪 Testing: {combo['name']}")
            
            # Create feature matrix
            feature_matrices = []
            feature_names = []
            
            if combo['use_original']:
                feature_matrices.append(original_data.values)
                feature_names.extend([f"orig_{col}" for col in self.original_features])
            
            if combo['use_mat2vec']:
                feature_matrices.append(mat2vec_embeddings)
                feature_names.extend([f"mat2vec_{i}" for i in range(mat2vec_embeddings.shape[1])])
            
            if not feature_matrices:
                print(f"❌ No features available for {combo['name']}")
                continue
            
            X = np.hstack(feature_matrices)
            print(f"Feature matrix shape: {X.shape}")
            
            # Train for each target
            for target in self.targets:
                if target not in self.df.columns:
                    print(f"⚠️ Target {target} not found in dataframe")
                    continue
                
                print(f"\n🎯 Target: {target}")
                
                y = self.df[target].copy()
                valid_idx = y.notna()
                X_clean = X[valid_idx]
                y_clean = y[valid_idx]
                
                print(f"   Valid samples: {len(X_clean)}")
                
                if target == 'log_Ionic_Conductivity':
                    print(f"   Log IC range: {y_clean.min():.3f} to {y_clean.max():.3f}")
                
                if len(X_clean) < 10:
                    print(f"❌ Too few samples for {target}")
                    continue
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=self.random_state
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train models
                models = {
                    'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=self.random_state),
                    'GradientBoosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=self.random_state),
                    'Ridge': Ridge(alpha=0.5),
                    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=self.random_state)
                }
                
                if XGBOOST_AVAILABLE:
                    models['XGBoost'] = XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=self.random_state)
                
                best_score = -np.inf
                best_model = None
                
                for model_name, model in models.items():
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        result = {
                            'combination': combo['name'],
                            'target': target,
                            'model': model_name,
                            'r2_score': r2,
                            'rmse': rmse,
                            'mae': mae,
                            'n_features': X.shape[1],
                            'n_samples': len(X_test)
                        }
                        
                        all_results.append(result)
                        
                        print(f"  {model_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
                        
                        if r2 > best_score:
                            best_score = r2
                            best_model = model_name
                    
                    except Exception as e:
                        print(f"  ❌ {model_name} failed: {e}")
                
                print(f"🏆 Best: {best_model} (R² = {best_score:.3f})")
        
        # Results summary
        results_df = pd.DataFrame(all_results)
        self.results = results_df
        
        print("\n" + "="*80)
        print("📊 MAT2VEC RESULTS SUMMARY - log(Ionic_Conductivity)")
        print("="*80)
        
        for target in self.targets:
            target_results = results_df[results_df['target'] == target]
            if len(target_results) > 0:
                best_result = target_results.loc[target_results['r2_score'].idxmax()]
                print(f"\n🎯 BEST for {target}:")
                print(f"   {best_result['combination']} + {best_result['model']}")
                print(f"   R² = {best_result['r2_score']:.3f}")
                print(f"   RMSE = {best_result['rmse']:.3f}")
                print(f"   MAE = {best_result['mae']:.3f}")
        
        # Improvement analysis
        print(f"\n📈 MAT2VEC IMPACT ANALYSIS:")
        comparison = results_df.groupby(['combination', 'target'])['r2_score'].mean().unstack()
        if comparison is not None and len(comparison) > 0:
            print(comparison.round(3))
        
        # Calculate improvements
        for target in self.targets:
            target_results = results_df[results_df['target'] == target]
            if len(target_results) > 0:
                original_score = target_results[target_results['combination'] == 'Original_Only']['r2_score'].mean()
                best_mat2vec_score = target_results[target_results['combination'] != 'Original_Only']['r2_score'].max()
                
                if not np.isnan(original_score) and not np.isnan(best_mat2vec_score):
                    improvement = ((best_mat2vec_score - original_score) / original_score) * 100
                    print(f"{target} improvement with Mat2vec: {improvement:+.1f}%")
        
        return results_df
    
    def export_results(self, filename_prefix="ddse_mat2vec_log_targets"):
        """Export Mat2vec results for log targets"""
        
        if hasattr(self, 'results') and len(self.results) > 0:
            self.results.to_csv(f"{filename_prefix}_results.csv", index=False)
            print(f"✅ Exported results to {filename_prefix}_results.csv")
        
        # Export Mat2vec embeddings
        if 'mat2vec' in self.embeddings_data:
            np.save(f"{filename_prefix}_embeddings.npy", self.embeddings_data['mat2vec'])
            print(f"✅ Exported Mat2vec embeddings to {filename_prefix}_embeddings.npy")
        
        # Create summary report
        if hasattr(self, 'results') and len(self.results) > 0:
            summary_data = []
            for target in self.targets:
                target_results = self.results[self.results['target'] == target]
                if len(target_results) > 0:
                    best_result = target_results.loc[target_results['r2_score'].idxmax()]
                    
                    # Find original baseline
                    original_result = target_results[target_results['combination'] == 'Original_Only']
                    original_r2 = original_result['r2_score'].iloc[0] if len(original_result) > 0 else np.nan
                    
                    improvement = ((best_result['r2_score'] - original_r2) / original_r2 * 100) if not np.isnan(original_r2) else np.nan
                    
                    summary_data.append({
                        'Target': target,
                        'Best_Combination': best_result['combination'],
                        'Best_Model': best_result['model'],
                        'Best_R2': best_result['r2_score'],
                        'Original_R2': original_r2,
                        'Improvement_Percent': improvement,
                        'Best_RMSE': best_result['rmse'],
                        'Best_MAE': best_result['mae'],
                        'N_Features': best_result['n_features']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f"{filename_prefix}_summary.csv", index=False)
            print(f"✅ Exported summary to {filename_prefix}_summary.csv")

# Main execution function
def main():
    """Main function for Mat2vec training with log targets"""
    
    print("🚀 STARTING MAT2VEC TRAINING - log(Ionic_Conductivity)")
    print("🎯 Focus: Mat2vec embeddings with log-transformed ionic conductivity")
    print("="*80)
    
    try:
        # Initialize trainer
        trainer = Mat2vecOnlyTrainer(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
        
        # Train models with Mat2vec for log targets
        results_df = trainer.train_mat2vec_models()
        
        if results_df is not None:
            # Export results
            trainer.export_results()
            
            print("\n✅ MAT2VEC LOG TARGET TRAINING COMPLETE!")
            print("📋 Key achievements:")
            print("   • Successfully created log(Ionic_Conductivity) target")
            print("   • Trained models for log(IC)")
            print("   • Fixed dimension consistency issues")
            print("   • Log transformation handling for zero/negative IC values")
            print("   • Comprehensive performance analysis")
            
            # Show log transformation summary
            if hasattr(trainer, 'df') and 'log_Ionic_Conductivity' in trainer.df.columns:
                log_ic = trainer.df['log_Ionic_Conductivity']
                print(f"\n📊 Log Transformation Summary:")
                print(f"   Valid log(IC) values: {log_ic.notna().sum()}")
                print(f"   Log(IC) range: {log_ic.min():.3f} to {log_ic.max():.3f}")
                print(f"   Mean log(IC): {log_ic.mean():.3f}")
        else:
            print("\n❌ Training failed due to embedding generation issues")
        
        return trainer
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the Mat2vec pipeline with log targets
    trainer = main()
