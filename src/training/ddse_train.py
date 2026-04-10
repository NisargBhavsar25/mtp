import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_CLEANED, OUTPUTS_DIR, journal_savefig
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score, KFold)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                           mean_absolute_percentage_error)
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                BayesianRidge)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                            ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

class DDSEModelTrainer:
    def __init__(self, csv_file_path, random_state=42):
        """Initialize the model trainer with DDSE data"""
        self.df = pd.read_csv(csv_file_path)
        self.random_state = random_state
        self.results = {}
        self.best_models = {}
        self.feature_importance = {}
        
        # Define target variables
        self.targets = ['Ionic_Conductivity']
        
        # Define feature columns (exclude targets, identifiers, and non-feature columns)
        exclude_cols = ['electrolyte', 'doi', 'Ea_eV'] + self.targets
        self.feature_cols = [col for col in self.df.columns 
                           if col not in exclude_cols and 
                           self.df[col].dtype in ['int64', 'float64', 'bool']]
        
        print(f"Initialized with {len(self.df)} samples and {len(self.feature_cols)} features")
        print(f"Target variables: {self.targets}")
        print(f"Feature columns: {self.feature_cols}")
    
    def prepare_data(self, target_col, feature_selection_k=15):
        """Prepare data for training with preprocessing and feature selection"""
        
        # Get features and target
        X = self.df[self.feature_cols].copy()
        y = self.df[target_col].copy()
        
        # Remove rows with missing target values
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"\nPreparing data for {target_col}:")
        print(f"Samples after removing missing targets: {len(X)}")
        
        # Handle missing values in features
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Feature selection using SelectKBest
        if len(self.feature_cols) > feature_selection_k:
            selector = SelectKBest(score_func=f_regression, k=feature_selection_k)
            X_selected = selector.fit_transform(X_imputed, y)
            selected_features = np.array(self.feature_cols)[selector.get_support()]
            print(f"Selected top {feature_selection_k} features: {list(selected_features)}")
        else:
            X_selected = X_imputed.values
            selected_features = self.feature_cols
        
        # Train/validation/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=self.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state  # 0.25 * 0.8 = 0.2
        )
        
        print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, 
                selected_features, imputer)
    
    def get_models(self):
        """Define all models to train with their hyperparameter grids"""
        
        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            
            'Lasso': {
                'model': Lasso(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0]
                }
            },
            
            'ElasticNet': {
                'model': ElasticNet(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            
            'RandomForest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            
            'ExtraTrees': {
                'model': ExtraTreesRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            },
            
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'epsilon': [0.01, 0.1, 0.2],
                    'kernel': ['rbf', 'linear']
                }
            },
            
            'KNeighbors': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance']
                }
            },
            
            'MLPRegressor': {
                'model': MLPRegressor(random_state=self.random_state, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'learning_rate_init': [0.001, 0.01],
                    'alpha': [0.0001, 0.001]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': XGBRegressor(random_state=self.random_state, eval_metric='rmse'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': LGBMRegressor(random_state=self.random_state, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [20, 31, 50]
                }
            }
        
        return models
    
    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        """Evaluate a trained model on validation and test sets"""
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        metrics = {}
        
        # Training metrics
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        
        # Validation metrics
        metrics['val_r2'] = r2_score(y_val, y_val_pred)
        metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_val_pred))
        metrics['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        
        # Test metrics
        metrics['test_r2'] = r2_score(y_test, y_test_pred)
        metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        
        # MAPE (if no zero values)
        if not np.any(y_test == 0):
            metrics['test_mape'] = mean_absolute_percentage_error(y_test, y_test_pred)
        
        return metrics, y_test_pred
    
    def train_single_target(self, target_col, use_grid_search=True, cv_folds=5):
        """Train all models for a single target variable"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING MODELS FOR {target_col}")
        print(f"{'='*60}")
        
        # Prepare data
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         selected_features, imputer) = self.prepare_data(target_col)
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Get models
        models = self.get_models()
        
        # Store results for this target
        target_results = []
        
        # Train each model
        for model_name, model_config in models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                base_model = model_config['model']
                param_grid = model_config['params']
                
                if use_grid_search and param_grid:
                    # Hyperparameter tuning with GridSearchCV
                    grid_search = GridSearchCV(
                        base_model, 
                        param_grid, 
                        cv=cv_folds,
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    cv_score = grid_search.best_score_
                    
                    print(f"Best params: {best_params}")
                    print(f"CV R2 score: {cv_score:.3f}")
                    
                else:
                    # Train with default parameters
                    best_model = base_model
                    best_model.fit(X_train_scaled, y_train)
                    best_params = "Default"
                    cv_score = None
                
                # Evaluate model
                metrics, y_pred = self.evaluate_model(
                    best_model, X_train_scaled, X_val_scaled, X_test_scaled,
                    y_train, y_val, y_test
                )
                
                # Add model info to metrics
                metrics['model_name'] = model_name
                metrics['best_params'] = str(best_params)
                metrics['cv_score'] = cv_score
                
                # Calculate overfitting indicator
                metrics['overfitting'] = metrics['train_r2'] - metrics['val_r2']
                
                target_results.append(metrics)
                
                print(f"Test R2: {metrics['test_r2']:.3f}, "
                      f"Test RMSE: {metrics['test_rmse']:.3f}, "
                      f"Test MAE: {metrics['test_mae']:.3f}")
                
                # Store feature importance if available
                if hasattr(best_model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': selected_features,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    self.feature_importance[f"{target_col}_{model_name}"] = importance
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Convert results to DataFrame and sort by test R2
        results_df = pd.DataFrame(target_results)
        results_df = results_df.sort_values('test_r2', ascending=False).reset_index(drop=True)
        
        # Store results
        self.results[target_col] = results_df
        
        # Store best model info
        best_model_info = results_df.iloc[0]
        self.best_models[target_col] = {
            'model_name': best_model_info['model_name'],
            'test_r2': best_model_info['test_r2'],
            'test_rmse': best_model_info['test_rmse'],
            'test_mae': best_model_info['test_mae']
        }
        
        print(f"\n📊 RESULTS SUMMARY FOR {target_col}:")
        print(results_df[['model_name', 'test_r2', 'test_rmse', 'test_mae', 'overfitting']].round(3))
        
        print(f"\n🏆 BEST MODEL FOR {target_col}: {best_model_info['model_name']}")
        print(f"   Test R2: {best_model_info['test_r2']:.3f}")
        print(f"   Test RMSE: {best_model_info['test_rmse']:.3f}")
        print(f"   Test MAE: {best_model_info['test_mae']:.3f}")
        
        return results_df
    
    def train_all_targets(self, use_grid_search=True, cv_folds=5):
        """Train models for all target variables"""
        
        print("🚀 STARTING COMPREHENSIVE MODEL TRAINING")
        print(f"Targets: {self.targets}")
        print(f"Grid Search: {use_grid_search}")
        print(f"CV Folds: {cv_folds}")
        
        # Train for each target
        for target in self.targets:
            if target in self.df.columns:
                self.train_single_target(target, use_grid_search, cv_folds)
            else:
                print(f"Warning: Target {target} not found in data")
        
        # Generate final summary
        self.generate_final_summary()
    
    def generate_final_summary(self):
        """Generate final summary of all results"""
        
        print("\n" + "="*80)
        print("🏆 FINAL MODEL COMPARISON SUMMARY")
        print("="*80)
        
        for target, results_df in self.results.items():
            print(f"\n📊 TOP 5 MODELS FOR {target}:")
            top_5 = results_df.head(5)[['model_name', 'test_r2', 'test_rmse', 'test_mae']]
            print(top_5.to_string(index=False))
        
        print(f"\n🎯 BEST MODEL FOR EACH TARGET:")
        for target, best_info in self.best_models.items():
            print(f"{target}: {best_info['model_name']} (R2: {best_info['test_r2']:.3f})")
        
        # Overall recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Check for consistent best performers
        all_models = []
        for results_df in self.results.values():
            all_models.extend(results_df['model_name'].head(3).tolist())
        
        from collections import Counter
        model_counts = Counter(all_models)
        top_performers = model_counts.most_common(3)
        
        print(f"• Most consistent performers across targets:")
        for model, count in top_performers:
            print(f"  - {model} (appeared in top 3 for {count} targets)")
        
        # Check for overfitting
        print(f"• Models with low overfitting (train_r2 - val_r2 < 0.1):")
        for target, results_df in self.results.items():
            low_overfit = results_df[results_df['overfitting'] < 0.1]['model_name'].head(3).tolist()
            if low_overfit:
                print(f"  {target}: {', '.join(low_overfit)}")
    
    def plot_results(self, save_plots=True):
        """Generate comprehensive result visualizations"""
        
        # Set style
        # Journal style applied globally via src.config
        
        # 1. Model comparison plot
        fig, axes = plt.subplots(1, len(self.targets), figsize=(6*len(self.targets), 8))
        if len(self.targets) == 1:
            axes = [axes]
        
        for i, (target, results_df) in enumerate(self.results.items()):
            # Plot R2 scores
            top_10 = results_df.head(10)
            axes[i].barh(range(len(top_10)), top_10['test_r2'], color='skyblue', alpha=0.7)
            axes[i].set_yticks(range(len(top_10)))
            axes[i].set_yticklabels(top_10['model_name'])
            axes[i].set_xlabel('Test R² Score')
            axes[i].set_title(f'Model Performance - {target}')
            axes[i].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for j, v in enumerate(top_10['test_r2']):
                axes[i].text(v + 0.01, j, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        if save_plots:
            journal_savefig(str(OUTPUTS_DIR / 'results' / 'model_comparison.png'))
        plt.show()
        
        # 2. Overfitting analysis
        fig, axes = plt.subplots(1, len(self.targets), figsize=(6*len(self.targets), 6))
        if len(self.targets) == 1:
            axes = [axes]
        
        for i, (target, results_df) in enumerate(self.results.items()):
            axes[i].scatter(results_df['val_r2'], results_df['test_r2'], 
                          alpha=0.7, s=60, color='coral')
            
            # Add diagonal line (perfect generalization)
            min_val, max_val = axes[i].get_xlim()
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            axes[i].set_xlabel('Validation R²')
            axes[i].set_ylabel('Test R²')
            axes[i].set_title(f'Generalization Analysis - {target}')
            axes[i].grid(alpha=0.3)
            
            # Annotate best models
            for idx, row in results_df.head(3).iterrows():
                axes[i].annotate(row['model_name'], 
                               (row['val_r2'], row['test_r2']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        if save_plots:
            journal_savefig(str(OUTPUTS_DIR / 'results' / 'generalization_analysis.png'))
        plt.show()
        
        # 3. Feature importance plot (for best tree-based model)
        for target in self.targets:
            importance_keys = [key for key in self.feature_importance.keys() 
                             if target in key and any(model in key for model in 
                                                    ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'XGBoost', 'LightGBM'])]
            
            if importance_keys:
                # Get feature importance for best tree-based model
                best_key = importance_keys[0]  # Could be improved to select actual best
                importance_df = self.feature_importance[best_key]
                
                plt.figure(figsize=(10, 8))
                top_features = importance_df.head(15)
                plt.barh(range(len(top_features)), top_features['importance'], color='lightgreen')
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 15 Feature Importances - {target}\n({best_key.split("_")[-1]} Model)')
                plt.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, v in enumerate(top_features['importance']):
                    plt.text(v + 0.001, i, f'{v:.3f}', va='center')
                
                plt.tight_layout()
                if save_plots:
                    journal_savefig(str(OUTPUTS_DIR / 'results' / f'feature_importance_{target}.png'))
                plt.show()
    
    def export_results(self, filename_prefix="ddse_model_results"):
        """Export results to CSV files"""
        
        # Export detailed results for each target
        for target, results_df in self.results.items():
            filename = f"{filename_prefix}_{target}.csv"
            results_df.to_csv(filename, index=False)
            print(f"Exported {target} results to {filename}")
        
        # Export feature importance
        if self.feature_importance:
            with pd.ExcelWriter(f"{filename_prefix}_feature_importance.xlsx") as writer:
                for key, importance_df in self.feature_importance.items():
                    sheet_name = key.replace('/', '_')[:31]  # Excel sheet name limit
                    importance_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Exported feature importance to {filename_prefix}_feature_importance.xlsx")
        
        # Export summary
        summary_data = []
        for target, best_info in self.best_models.items():
            summary_data.append({
                'Target': target,
                'Best_Model': best_info['model_name'],
                'Test_R2': best_info['test_r2'],
                'Test_RMSE': best_info['test_rmse'],
                'Test_MAE': best_info['test_mae']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{filename_prefix}_summary.csv", index=False)
        print(f"Exported summary to {filename_prefix}_summary.csv")

# Usage Example and Main Function
def main():
    """Main function to run the complete training pipeline"""
    
    # Initialize trainer
    trainer = DDSEModelTrainer(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
    
    # Train all models (this will take some time)
    trainer.train_all_targets(use_grid_search=True, cv_folds=5)
    
    # Generate plots
    trainer.plot_results(save_plots=True)
    
    # Export results
    trainer.export_results()
    
    print("\n✅ TRAINING COMPLETE!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
