import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
import os
import warnings

from src.config import MODELS_DIR
from src.features import get_composition as gc

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. MAT2VEC GENERATOR (Uses your parser)
# ---------------------------------------------------------
class Mat2VecGenerator:
    """Generates embeddings using the parsed composition from your file"""
    
    def __init__(self, metadata_path=None):
        self.model = None
        self.dim = 200

        if metadata_path is None:
            metadata_path = str(MODELS_DIR / "mat2vec_info.joblib")
        if os.path.exists(metadata_path):
            try:
                from gensim.models import Word2Vec
                info = joblib.load(metadata_path)
                self.model = Word2Vec.load(info['model_path'])
                self.dim = info.get('embedding_dim', 200)
                print(f"✅ Loaded Mat2Vec model (Dim: {self.dim})")
            except Exception as e:
                print(f"⚠️ Could not load Mat2Vec: {e}")
        else:
            print("⚠️ Mat2Vec metadata not found.")

    def get_embedding(self, formula):
        if not self.model:
            return np.zeros(self.dim)
        
        try:
            # USE YOUR PARSER HERE
            composition = gc.parse_mixture_formula(formula)
            
            vectors = []
            weights = []
            
            for el, amt in composition.items():
                if el in self.model.wv:
                    vectors.append(self.model.wv[el])
                    weights.append(amt)
            
            if vectors:
                # Weighted average of element vectors
                embedding = np.average(vectors, axis=0, weights=weights)
                
                # Safety resize
                if len(embedding) != self.dim:
                    new_vec = np.zeros(self.dim)
                    m = min(len(embedding), self.dim)
                    new_vec[:m] = embedding[:m]
                    return new_vec
                return embedding
            
            return np.zeros(self.dim)
            
        except Exception as e:
            # print(f"Mat2Vec Error for {formula}: {e}")
            return np.zeros(self.dim)

# ---------------------------------------------------------
# 2. MAIN PREDICTOR
# ---------------------------------------------------------
class ConductivityPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = str(MODELS_DIR)
        self.model_dir = model_dir

        # Load Metadata
        try:
            self.metadata = joblib.load(os.path.join(model_dir, "model_metadata.joblib"))
        except FileNotFoundError:
            print("❌ Error: 'saved_models/model_metadata.joblib' not found. Train the model first.")
            sys.exit(1)
            
        self.target = 'log_Ionic_Conductivity'

        # Load Model
        model_path = os.path.join(model_dir, f"ddse_model_{self.target}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model for {self.target} not found!")

        self.model = joblib.load(model_path)
        print(f"Loaded {self.target} model")

        # Load error stats (residual std from test set)
        error_path = os.path.join(model_dir, f"error_stats_{self.target}.joblib")
        self.residual_std = None
        if os.path.exists(error_path):
            error_info = joblib.load(error_path)
            self.residual_std = error_info['residual_std']
            print(f"Loaded error stats (residual std={self.residual_std:.4f})")
        else:
            print("Error stats not found — predictions will have no error intervals")

        # Check configuration
        self.config = self.metadata['model_configs'][self.target]
        self.required_cols = self.metadata['feature_columns'][self.target]
        self.use_mat2vec = 'Mat2vec' in self.config['features']

        print(f"Model uses: {self.config['features']}")

        # Initialize Mat2Vec if needed
        if self.use_mat2vec:
            self.m2v_gen = Mat2VecGenerator(os.path.join(model_dir, "mat2vec_info.joblib"))

    def predict_csv(self, csv_path, output_path="predictions.csv"):
        print(f"\n🚀 Processing: {csv_path}")
        
        # 1. Read Data
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return

        # Identify formula column
        possible_names = ['formula', 'Formula', 'composition', 'Composition', 'Material', 'electrolyte']
        formula_col = next((col for col in possible_names if col in df.columns), None)
        
        if not formula_col:
            print(f"❌ Could not find formula column. Checked: {possible_names}")
            return
        
        print(f"   Formula column: '{formula_col}'")
        
        # 2. Add Temperature if missing (Default to Room Temp)
        if 'Temp_K' not in df.columns:
            print("   ℹ️  'Temp_K' column missing. Assuming 298 K (Room Temperature).")
            df['Temp_K'] = 298.0
        
        # 3. GENERATE PHYSICAL FEATURES (Using your file)
        # This adds columns like 'avg_electronegativity', 'li_fraction' to df
        print("   🧪 Calculating physical descriptors...")
        df_enhanced = gc.enhance_composition_features_fixed(df, formula_col)
        
        # 4. RENAME COLUMNS to match training data ('orig_' prefix)
        # The model was trained on 'orig_avg_electronegativity', but your script gives 'avg_electronegativity'
        rename_map = {col: f"orig_{col}" for col in df_enhanced.columns 
                      if col not in ['electrolyte', 'formula', formula_col, 'Temp_K', 'doi']}
        
        # Manually ensure Temp_K is renamed if the model expects orig_Temp_K
        rename_map['Temp_K'] = 'orig_Temp_K'
        
        df_model_input = df_enhanced.rename(columns=rename_map)
        
        # 5. GENERATE MAT2VEC (if needed)
        if self.use_mat2vec:
            print("   🧠 Generating Mat2Vec embeddings...")
            mat2vec_cols = {}
            for idx, row in df_enhanced.iterrows():
                emb = self.m2v_gen.get_embedding(row[formula_col])
                for i, val in enumerate(emb):
                    col_name = f"mat2vec_{i}"
                    if col_name not in mat2vec_cols:
                        mat2vec_cols[col_name] = []
                    mat2vec_cols[col_name].append(val)
            
            # Add to dataframe
            for col, values in mat2vec_cols.items():
                df_model_input[col] = values

        # 6. ALIGN COLUMNS & PREDICT
        print("   🔮 Predicting conductivity...")
        
        # Create a clean dataframe with ONLY the columns the model expects, in exact order
        # Fill missing features with 0 (safe fallback)
        X = df_model_input.reindex(columns=self.required_cols, fill_value=0)
        
        try:
            log_preds = self.model.predict(X)

            # Add results to original dataframe
            df['Predicted_log_IC'] = log_preds
            df['Predicted_IC_S_cm'] = 10 ** log_preds

            # Add error intervals based on residual std (±1σ and ±2σ)
            if self.residual_std is not None:
                s = self.residual_std
                df['Error_1sigma_log'] = s
                df['Error_2sigma_log'] = 1.96 * s
                df['CI_lower_log_95'] = log_preds - 1.96 * s
                df['CI_upper_log_95'] = log_preds + 1.96 * s
                df['CI_lower_S_cm_95'] = 10 ** df['CI_lower_log_95']
                df['CI_upper_S_cm_95'] = 10 ** df['CI_upper_log_95']

            df.to_csv(output_path, index=False)
            print(f"\nDone! Results saved to: {output_path}")

            preview_cols = [formula_col, 'Predicted_log_IC', 'Predicted_IC_S_cm']
            if self.residual_std is not None:
                preview_cols += ['CI_lower_log_95', 'CI_upper_log_95']
            print(f"   Preview:\n{df[preview_cols].head()}")

        except Exception as e:
            print(f"Prediction failed: {e}")
            print("   Debug: Input shape:", X.shape)
            print("   Debug: Expected cols:", self.required_cols[:5])

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    csv_file = input("Enter path to input CSV file: ").strip()
    
    if os.path.exists(csv_file):
        predictor = ConductivityPredictor()
        predictor.predict_csv(csv_file)
    else:
        print("❌ File not found.")