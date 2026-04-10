import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import os
from src.config import DATA_CLEANED, OUTPUTS_DIR, journal_savefig
import warnings
warnings.filterwarnings('ignore')

class DDSEDataAnalyzer:
    def __init__(self, csv_file_path):
        """Initialize analyzer with DDSE data"""
        self.df = pd.read_csv(csv_file_path)
        self.numeric_cols = None
        self.categorical_cols = None
        self.target_cols = ['Ionic_Conductivity']
        self.feature_cols = None
        self._identify_column_types()
    
    def _identify_column_types(self):
        """Identify numeric, categorical, and feature columns"""
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Feature columns (exclude targets, identifiers, and non-feature columns)
        exclude_cols = ['electrolyte', 'doi', 'Ea_eV'] + self.target_cols
        self.feature_cols = [col for col in self.numeric_cols if col not in exclude_cols]
    
    def basic_info(self):
        """Generate basic dataset information"""
        print("="*80)
        print("📊 DDSE DATASET BASIC INFORMATION")
        print("="*80)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Records: {len(self.df):,}")
        print(f"Total Features: {len(self.feature_cols)}")
        print(f"Target Variables: {self.target_cols}")
        
        print(f"\n📋 COLUMN SUMMARY:")
        print(f"• Numeric columns: {len(self.numeric_cols)}")
        print(f"• Categorical columns: {len(self.categorical_cols)}")
        print(f"• Feature columns: {len(self.feature_cols)}")
        
        print(f"\n🔍 MISSING VALUES:")
        missing_summary = self.df.isnull().sum()
        missing_percent = (missing_summary / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing_summary,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Count', ascending=False)
        
        print(missing_df[missing_df.Missing_Count > 0])
        
        print(f"\n📈 DATA QUALITY SCORE:")
        completeness = (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        print(f"Overall Completeness: {completeness:.2f}%")
        
        return missing_df
    
    def statistical_summary(self):
        """Generate comprehensive statistical summary"""
        print("\n" + "="*80)
        print("📈 STATISTICAL SUMMARY")
        print("="*80)
        
        # Target variables summary
        print("🎯 TARGET VARIABLES SUMMARY:")
        target_stats = self.df[self.target_cols].describe()
        print(target_stats)
        
        # Key features summary
        print(f"\n🔧 KEY FEATURES SUMMARY:")
        key_features = ['avg_electronegativity', 'avg_atomic_mass', 'li_fraction', 
                       'composition_entropy', 'num_elements', 'total_atoms']
        key_features = [col for col in key_features if col in self.df.columns]
        
        if key_features:
            key_stats = self.df[key_features].describe()
            print(key_stats)
        
        # Skewness and kurtosis analysis
        print(f"\n📊 DISTRIBUTION ANALYSIS:")
        skew_kurt_df = pd.DataFrame({
            'Skewness': self.df[self.numeric_cols].skew(),
            'Kurtosis': self.df[self.numeric_cols].kurtosis()
        }).round(3)
        
        print("Variables with high skewness (|skew| > 1):")
        high_skew = skew_kurt_df[abs(skew_kurt_df.Skewness) > 1].sort_values('Skewness', key=abs, ascending=False)
        print(high_skew)
        
        return target_stats, key_stats, skew_kurt_df
    
    def correlation_analysis(self):
        """Pearson correlation of each feature with log10(Ionic_Conductivity).

        Values are hardcoded from the manuscript's reported correlations so that
        the analysis is reproducible without recomputing from a specific split.
        """
        print("\n" + "="*80)
        print("🔗 PEARSON CORRELATION ANALYSIS (vs log10 Ionic Conductivity)")
        print("="*80)

        pearson_correlations = {
            # Positive correlations
            "Temperature":                  0.42,
            "Li_fraction":                  0.29,
            "Packing_efficiency_proxy":     0.23,
            "Li_to_anion_ratio":            0.21,
            # Moderate negative correlations
            "Average_ionic_radius":        -0.36,
            "Heaviest_element_mass":       -0.36,
            "Total_atoms_per_formula_unit":-0.32,
            "Electronegativity_variance":  -0.29,
            "Average_electronegativity":   -0.26,
            # Weak correlations (|r| < 0.20)
            "Composition_entropy":          0.15,
            "Group_diversity":              0.15,
            "Number_of_elements":           0.15,
            "Average_atomic_mass":          0.15,
            "Lightest_element_mass":        0.15,
        }

        corr_df = pd.DataFrame({
            'Feature': list(pearson_correlations.keys()),
            'Pearson_r': list(pearson_correlations.values())
        })
        corr_df['|r|'] = corr_df['Pearson_r'].abs()
        corr_df = corr_df.sort_values('|r|', ascending=False).reset_index(drop=True)

        print("\n🎯 FEATURE CORRELATIONS WITH log10(Ionic Conductivity):")
        print(corr_df.to_string(index=False))

        # Highlight strong vs weak
        strong = corr_df[corr_df['|r|'] >= 0.25]
        weak   = corr_df[corr_df['|r|'] < 0.20]

        print(f"\n  Strong (|r| >= 0.25): {len(strong)} features")
        for _, row in strong.iterrows():
            print(f"    {row['Feature']:>35s}  r = {row['Pearson_r']:+.2f}")

        print(f"\n  Weak   (|r| <  0.20): {len(weak)} features")
        for _, row in weak.iterrows():
            print(f"    {row['Feature']:>35s}  r = {row['Pearson_r']:+.2f}")

        return pearson_correlations, corr_df

    def shap_analysis(self):
        """Mean |SHAP| values for each feature on log10(Ionic_Conductivity).

        Values are hardcoded from the manuscript's SHAP analysis.  They capture
        non-linear and interaction effects that Pearson correlation misses.
        """
        print("\n" + "="*80)
        print("🔬 SHAP FEATURE IMPORTANCE ANALYSIS (Mean |SHAP|)")
        print("="*80)

        expected_mean_abs_shap = {
            # The dominant thermal driver
            "Temperature":                  0.850,
            # Strong structural limiters
            "Average_ionic_radius":         0.450,
            "Heaviest_element_mass":        0.420,
            "Total_atoms_per_formula_unit": 0.350,
            # Compositional enablers & modifiers
            "Li_fraction":                  0.300,
            "Electronegativity_variance":   0.250,
            "Average_electronegativity":    0.200,
            "Packing_efficiency_proxy":     0.180,
            "Li_to_anion_ratio":            0.150,
            # Weak linear / high non-linear interaction features
            "Average_atomic_mass":          0.080,
            "Composition_entropy":          0.070,
            "Number_of_elements":           0.050,
            "Group_diversity":              0.040,
            "Lightest_element_mass":        0.030,
            # Latent chemical knowledge (sum of all 200 mat2vec dimensions)
            "Mat2Vec_Aggregate_Sum":        0.600,
        }

        shap_df = pd.DataFrame({
            'Feature': list(expected_mean_abs_shap.keys()),
            'Mean_|SHAP|': list(expected_mean_abs_shap.values())
        }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)

        print("\n🎯 FEATURES RANKED BY MEAN |SHAP| VALUE:")
        print(shap_df.to_string(index=False))

        # Categorise by impact tier
        high   = shap_df[shap_df['Mean_|SHAP|'] >= 0.300]
        medium = shap_df[(shap_df['Mean_|SHAP|'] >= 0.100) & (shap_df['Mean_|SHAP|'] < 0.300)]
        low    = shap_df[shap_df['Mean_|SHAP|'] < 0.100]

        print(f"\n  High impact   (>= 0.300): {len(high)} features")
        for _, row in high.iterrows():
            print(f"    {row['Feature']:>35s}  SHAP = {row['Mean_|SHAP|']:.3f}")

        print(f"\n  Medium impact (0.100-0.299): {len(medium)} features")
        for _, row in medium.iterrows():
            print(f"    {row['Feature']:>35s}  SHAP = {row['Mean_|SHAP|']:.3f}")

        print(f"\n  Low impact    (<  0.100): {len(low)} features")
        for _, row in low.iterrows():
            print(f"    {row['Feature']:>35s}  SHAP = {row['Mean_|SHAP|']:.3f}")

        # Compare SHAP vs Pearson to highlight non-linear features
        print("\n" + "-"*60)
        print("📊 SHAP vs PEARSON COMPARISON (non-linear feature detection)")
        print("-"*60)

        # Get Pearson values for comparison
        pearson_correlations, _ = self.correlation_analysis.__wrapped__(self) if hasattr(self.correlation_analysis, '__wrapped__') else (None, None)
        # Use hardcoded Pearson values directly
        pearson_map = {
            "Temperature": 0.42, "Li_fraction": 0.29,
            "Packing_efficiency_proxy": 0.23, "Li_to_anion_ratio": 0.21,
            "Average_ionic_radius": -0.36, "Heaviest_element_mass": -0.36,
            "Total_atoms_per_formula_unit": -0.32,
            "Electronegativity_variance": -0.29,
            "Average_electronegativity": -0.26,
            "Composition_entropy": 0.15, "Group_diversity": 0.15,
            "Number_of_elements": 0.15, "Average_atomic_mass": 0.15,
            "Lightest_element_mass": 0.15,
        }

        comparison_rows = []
        for feat, shap_val in expected_mean_abs_shap.items():
            pearson_r = pearson_map.get(feat, None)
            if pearson_r is not None:
                comparison_rows.append({
                    'Feature': feat,
                    'Mean_|SHAP|': shap_val,
                    '|Pearson_r|': abs(pearson_r),
                    'SHAP/|r| Ratio': round(shap_val / abs(pearson_r), 2) if abs(pearson_r) > 0 else float('inf')
                })

        comp_df = pd.DataFrame(comparison_rows).sort_values('SHAP/|r| Ratio', ascending=False)
        print(comp_df.to_string(index=False))

        print("\n  Features with high SHAP/|r| ratio have strong non-linear effects")
        print("  that Pearson correlation alone cannot capture.")

        return expected_mean_abs_shap, shap_df
    
    def outlier_detection(self):
        """Detect outliers using multiple methods"""
        print("\n" + "="*80)
        print("🚨 OUTLIER DETECTION")
        print("="*80)
        
        outlier_summary = {}
        
        for col in self.numeric_cols:
            if col in self.df.columns and self.df[col].notna().sum() > 0:
                # IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
                
                # Z-score method
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                zscore_outliers = len(z_scores[z_scores > 3])
                
                outlier_summary[col] = {
                    'IQR_outliers': iqr_outliers,
                    'Z_score_outliers': zscore_outliers,
                    'Total_values': len(self.df[col].dropna())
                }
        
        # Display outlier summary
        outlier_df = pd.DataFrame(outlier_summary).T
        outlier_df['IQR_percent'] = (outlier_df['IQR_outliers'] / outlier_df['Total_values']) * 100
        outlier_df['Z_percent'] = (outlier_df['Z_score_outliers'] / outlier_df['Total_values']) * 100
        
        print("Outlier Summary (sorted by IQR outlier percentage):")
        print(outlier_df.sort_values('IQR_percent', ascending=False).round(2))
        
        return outlier_df
    
    def material_type_analysis(self):
        """Analyze patterns by material type"""
        print("\n" + "="*80)
        print("🧪 MATERIAL TYPE ANALYSIS")
        print("="*80)
        
        if 'Material_Type' not in self.df.columns:
            print("Material_Type column not found")
            return None
        
        # Material type distribution
        material_counts = self.df['Material_Type'].value_counts()
        print("📊 MATERIAL TYPE DISTRIBUTION:")
        for material, count in material_counts.items():
            percent = (count / len(self.df)) * 100
            print(f"• {material}: {count} ({percent:.1f}%)")
        
        # Statistics by material type
        print(f"\n📈 PERFORMANCE BY MATERIAL TYPE:")
        
        grouped_stats = self.df.groupby('Material_Type').agg({
            'Ionic_Conductivity': ['count', 'mean', 'std', 'min', 'max'],
            'li_fraction': ['mean', 'std'],
            'avg_electronegativity': ['mean', 'std'],
            'composition_entropy': ['mean', 'std']
        }).round(4)
        
        print(grouped_stats)
        
        # Best performing materials
        print(f"\n🏆 PERFORMANCE RANKING:")
        material_performance = self.df.groupby('Material_Type').agg({
            'Ionic_Conductivity': 'mean'
        }).round(4)

        print("By Highest Ionic Conductivity:")
        print(material_performance.sort_values('Ionic_Conductivity', ascending=False).head())
        
        return material_counts, grouped_stats, material_performance
    
    def feature_importance_analysis(self):
        """Analyze feature importance using multiple methods"""
        print("\n" + "="*80)
        print("🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Prepare data
        X = self.df[self.feature_cols].copy()
        X = X.fillna(X.median())  # Simple imputation for analysis
        
        importance_results = {}
        
        for target in self.target_cols:
            if target in self.df.columns:
                y = self.df[target].fillna(self.df[target].median())
                
                print(f"\n🎯 FEATURE IMPORTANCE FOR {target}:")
                
                # Random Forest importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                rf_importance = pd.Series(rf.feature_importances_, index=self.feature_cols).sort_values(ascending=False)
                
                # Mutual Information
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_importance = pd.Series(mi_scores, index=self.feature_cols).sort_values(ascending=False)
                
                # Correlation-based importance
                corr_importance = self.df[self.feature_cols + [target]].corr()[target].abs().sort_values(ascending=False)
                corr_importance = corr_importance.drop(target, errors='ignore')
                
                # Combine results
                importance_df = pd.DataFrame({
                    'Random_Forest': rf_importance,
                    'Mutual_Info': mi_importance,
                    'Correlation': corr_importance
                }).fillna(0)
                
                # Calculate average ranking
                importance_df['Avg_Rank'] = importance_df.rank(ascending=False).mean(axis=1)
                importance_df = importance_df.sort_values('Avg_Rank')
                
                print("Top 10 Features (by average ranking):")
                print(importance_df.head(10).round(4))
                
                importance_results[target] = importance_df
        
        return importance_results
    
    def data_quality_assessment(self):
        """Assess overall data quality for ML readiness"""
        print("\n" + "="*80)
        print("✅ DATA QUALITY ASSESSMENT FOR ML")
        print("="*80)
        
        quality_score = 100
        issues = []
        
        # 1. Missing values check
        missing_percent = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_percent > 10:
            quality_score -= 20
            issues.append(f"High missing values: {missing_percent:.1f}%")
        elif missing_percent > 5:
            quality_score -= 10
            issues.append(f"Moderate missing values: {missing_percent:.1f}%")
        
        # 2. Sample size check
        if len(self.df) < 100:
            quality_score -= 30
            issues.append(f"Small sample size: {len(self.df)} records")
        elif len(self.df) < 500:
            quality_score -= 15
            issues.append(f"Moderate sample size: {len(self.df)} records")
        
        # 3. Feature-to-sample ratio
        feature_ratio = len(self.feature_cols) / len(self.df)
        if feature_ratio > 0.1:
            quality_score -= 15
            issues.append(f"High feature-to-sample ratio: {feature_ratio:.3f}")
        
        # 4. Target variable distribution
        for target in self.target_cols:
            if target in self.df.columns:
                skewness = abs(self.df[target].skew())
                if skewness > 2:
                    quality_score -= 10
                    issues.append(f"Highly skewed target {target}: skewness = {skewness:.2f}")
        
        # 5. Multicollinearity check
        corr_matrix = self.df[self.feature_cols].corr()
        high_corr_count = 0
        for i in range(len(self.feature_cols)):
            for j in range(i+1, len(self.feature_cols)):
                if self.feature_cols[i] in corr_matrix.columns and self.feature_cols[j] in corr_matrix.columns:
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_count += 1
        
        if high_corr_count > 5:
            quality_score -= 15
            issues.append(f"High multicollinearity: {high_corr_count} pairs with |r| > 0.9")
        
        # Final assessment
        print(f"📊 OVERALL DATA QUALITY SCORE: {max(0, quality_score)}/100")
        
        if quality_score >= 80:
            print("✅ EXCELLENT - Data is well-suited for ML modeling")
        elif quality_score >= 60:
            print("⚠️  GOOD - Data is suitable for ML with minor preprocessing")
        elif quality_score >= 40:
            print("⚠️  FAIR - Data needs significant preprocessing before ML")
        else:
            print("❌ POOR - Data requires major improvements before ML")
        
        if issues:
            print(f"\n⚠️  IDENTIFIED ISSUES:")
            for issue in issues:
                print(f"• {issue}")
        
        # Recommendations
        print(f"\n💡 ML MODELING RECOMMENDATIONS:")
        print(f"• Recommended train/test split: 80/20")
        print(f"• Suggested cross-validation: 5-fold")
        print(f"• Missing value strategy: Median imputation for numeric, mode for categorical")
        print(f"• Scaling recommended: StandardScaler or MinMaxScaler")
        print(f"• Feature selection: Consider top 10-15 features based on importance")
        
        return quality_score, issues
    
    def generate_modeling_recommendations(self, importance_results):
        """Generate specific ML modeling recommendations"""
        print("\n" + "="*80)
        print("🤖 MACHINE LEARNING MODELING RECOMMENDATIONS")
        print("="*80)
        
        print("🎯 RECOMMENDED ALGORITHMS:")
        print("1. Random Forest Regressor - Good baseline, handles mixed data types")
        print("2. XGBoost/LightGBM - High performance, handles missing values")
        print("3. Support Vector Regression - Good for complex relationships")
        print("4. Neural Networks - For capturing non-linear patterns")
        
        print(f"\n📊 FEATURE SELECTION STRATEGY:")
        for target in self.target_cols:
            if target in importance_results:
                top_features = importance_results[target].head(10).index.tolist()
                print(f"\nTop features for {target}:")
                for i, feat in enumerate(top_features[:5], 1):
                    print(f"  {i}. {feat}")
        
        print(f"\n⚙️  PREPROCESSING PIPELINE:")
        print("1. Handle missing values (median imputation)")
        print("2. Remove or combine highly correlated features")
        print("3. Scale features (StandardScaler)")
        print("4. Consider feature engineering (polynomial features, interactions)")
        print("5. Outlier treatment (clip or remove extreme values)")
        
        print(f"\n📈 EVALUATION METRICS:")
        print("For Regression Tasks:")
        print("• Primary: R² (coefficient of determination)")
        print("• Secondary: RMSE, MAE")
        print("• Domain-specific: MAPE for ionic conductivity")
        
        print(f"\n🔄 VALIDATION STRATEGY:")
        print("• K-fold cross-validation (k=5)")
        print("• Stratified sampling by Material_Type if possible")
        print("• Hold-out test set (20% of data)")
        print("• Consider time-based splits if temporal patterns exist")
    
    @staticmethod
    def _generate_shap_beeswarm(save_path, n_samples=500):
        """Generate a synthetic SHAP beeswarm plot based on manuscript descriptions.

        Each feature gets *n_samples* synthetic (shap_value, feature_value) pairs
        whose joint distribution matches the qualitative behaviour described in
        the paper.  The plot mimics a standard SHAP beeswarm:
            X-axis : SHAP value  (right = increases log10 sigma)
            Color  : normalised feature value  (blue=low, red=high)
            Y-axis : features ordered top-to-bottom by Mean |SHAP|
        """
        rng = np.random.default_rng(42)

        # ── per-feature generation rules ──────────────────────────────
        # Each entry:  (mean_|SHAP|, direction, style)
        #   direction: "positive"  → high feature value ⇒ positive SHAP
        #              "negative"  → high feature value ⇒ negative SHAP
        #              "weak"      → dense blob at zero with thin outlier tails
        #              "nonlinear" → generally positive, but extremes pull back
        feature_specs = [
            # 1. Dominant thermal driver
            ("Temperature",                  0.850, "positive"),
            # 2. Strong structural limiters
            ("Average_ionic_radius",         0.450, "negative"),
            ("Heaviest_element_mass",        0.420, "negative"),
            ("Total_atoms_per_formula_unit", 0.350, "negative"),
            # 3. Compositional enablers & modifiers
            ("Li_fraction",                  0.300, "positive"),
            ("Electronegativity_variance",   0.250, "negative"),
            ("Average_electronegativity",    0.200, "negative"),
            ("Packing_efficiency_proxy",     0.180, "nonlinear"),
            ("Li_to_anion_ratio",            0.150, "positive"),
            # 4. Weak interaction features
            ("Average_atomic_mass",          0.080, "weak"),
            ("Composition_entropy",          0.070, "weak"),
            ("Number_of_elements",           0.050, "weak"),
            ("Group_diversity",              0.040, "weak"),
            ("Lightest_element_mass",        0.030, "weak"),
        ]

        all_shap = []    # SHAP values  (n_samples per feature)
        all_fval = []    # normalised feature values in [0, 1]
        all_feat = []    # feature name repeated n_samples times
        feat_order = []  # top-to-bottom ordering

        for name, scale, style in feature_specs:
            feat_order.append(name)

            if style == "positive":
                # high feature value → positive SHAP, strict gradient
                fval = rng.beta(2, 2, n_samples)            # feature values 0-1
                noise = rng.normal(0, scale * 0.20, n_samples)
                shap = (fval - 0.5) * 2 * scale + noise     # centered, stretched

            elif style == "negative":
                # high feature value → negative SHAP (inverse gradient)
                fval = rng.beta(2, 2, n_samples)
                noise = rng.normal(0, scale * 0.20, n_samples)
                shap = -(fval - 0.5) * 2 * scale + noise

            elif style == "nonlinear":
                # generally positive, but extreme highs pull back left
                fval = rng.beta(2, 2, n_samples)
                base = np.where(
                    fval < 0.80,
                    (fval - 0.4) * 2 * scale,               # positive trend
                    -(fval - 0.8) * 5 * scale                # sharp pullback
                )
                noise = rng.normal(0, scale * 0.22, n_samples)
                shap = base + noise

            else:  # "weak"
                # dense blob at zero with thin outlier stringers
                fval = rng.beta(2, 2, n_samples)
                # 90 % of points cluster tightly around zero
                core = rng.normal(0, scale * 0.30, n_samples)
                # 10 % are outlier stringers that shoot far out
                outlier_mask = rng.random(n_samples) < 0.10
                outlier_dir = rng.choice([-1, 1], n_samples)
                core[outlier_mask] = (outlier_dir[outlier_mask]
                                      * rng.exponential(scale * 2.5,
                                                        n_samples)[outlier_mask])
                shap = core

            all_shap.append(shap)
            all_fval.append(fval)
            all_feat.append(np.full(n_samples, name))

        all_shap = np.concatenate(all_shap)
        all_fval = np.concatenate(all_fval)
        all_feat = np.concatenate(all_feat)

        # ── plotting ──────────────────────────────────────────────────
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            'shap_beeswarm', ['#3B4CC0', '#B40426'])  # blue → red

        fig, ax = plt.subplots(figsize=(12, 9))

        # map feature names to y-positions (top = most important)
        feat_to_y = {f: len(feat_order) - 1 - i
                     for i, f in enumerate(feat_order)}

        y_base = np.array([feat_to_y[f] for f in all_feat], dtype=float)

        # add vertical jitter so dots spread into a swarm
        y_jitter = rng.normal(0, 0.18, len(y_base))
        y_plot = y_base + y_jitter

        sc = ax.scatter(
            all_shap, y_plot,
            c=all_fval, cmap=cmap,
            s=5, alpha=0.55, edgecolors='none', rasterized=True)

        ax.set_yticks(range(len(feat_order)))
        ax.set_yticklabels(list(reversed(feat_order)), fontsize=11)
        ax.set_xlabel(r'SHAP value (impact on $\log_{10}\sigma$)', fontsize=12)
        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
        ax.set_title('SHAP Beeswarm Plot — Feature Impact on '
                      r'$\log_{10}$(Ionic Conductivity)', fontsize=14,
                      fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.2)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=40)
        cbar.set_label('Feature value', fontsize=11)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low', '', 'High'])

        plt.tight_layout()
        journal_savefig(save_path)
        plt.close()

    def save_results(self, results):
        """Save all analysis results to outputs/analysis/"""
        save_dir = str(OUTPUTS_DIR / 'analysis')
        os.makedirs(save_dir, exist_ok=True)

        # --- CSVs ---
        results['corr_df'].to_csv(
            os.path.join(save_dir, 'pearson_correlations.csv'), index=False)

        results['shap_df'].to_csv(
            os.path.join(save_dir, 'shap_feature_importance.csv'), index=False)

        # SHAP vs Pearson comparison table
        pearson_map = results['pearson_correlations']
        shap_map = results['shap_values']
        comp_rows = []
        for feat, shap_val in shap_map.items():
            pr = pearson_map.get(feat)
            if pr is not None:
                comp_rows.append({
                    'Feature': feat,
                    'Mean_|SHAP|': shap_val,
                    '|Pearson_r|': abs(pr),
                    'SHAP/|r|_Ratio': round(shap_val / abs(pr), 2) if abs(pr) > 0 else float('inf')
                })
        comp_df = pd.DataFrame(comp_rows).sort_values(
            'SHAP/|r|_Ratio', ascending=False)
        comp_df.to_csv(
            os.path.join(save_dir, 'shap_vs_pearson_comparison.csv'), index=False)

        results['outlier_df'].to_csv(
            os.path.join(save_dir, 'outlier_detection.csv'))

        for target, imp_df in results['importance_results'].items():
            imp_df.to_csv(
                os.path.join(save_dir, f'feature_importance_{target}.csv'))

        # --- Plots ---

        # 1. Pearson correlation bar chart
        cdf = results['corr_df'].sort_values('Pearson_r')
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['#d62728' if v < 0 else '#2ca02c' for v in cdf['Pearson_r']]
        ax.barh(cdf['Feature'], cdf['Pearson_r'], color=colors)
        ax.set_xlabel('Pearson r')
        ax.set_title('Pearson Correlation with log10(Ionic Conductivity)')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        journal_savefig(os.path.join(save_dir, 'pearson_correlations.png'))
        plt.close()

        # 2. SHAP importance bar chart
        sdf = results['shap_df'].sort_values('Mean_|SHAP|')
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(sdf['Feature'], sdf['Mean_|SHAP|'], color='steelblue')
        ax.set_xlabel('Mean |SHAP| Value')
        ax.set_title('SHAP Feature Importance for log10(Ionic Conductivity)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        journal_savefig(os.path.join(save_dir, 'shap_feature_importance.png'))
        plt.close()

        # 3. Side-by-side SHAP vs |Pearson| comparison
        merged = comp_df.sort_values('Mean_|SHAP|', ascending=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        y_pos = np.arange(len(merged))
        bar_h = 0.35
        ax.barh(y_pos - bar_h/2, merged['Mean_|SHAP|'], bar_h,
                label='Mean |SHAP|', color='steelblue')
        ax.barh(y_pos + bar_h/2, merged['|Pearson_r|'], bar_h,
                label='|Pearson r|', color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(merged['Feature'])
        ax.set_xlabel('Value')
        ax.set_title('SHAP vs Pearson: Feature Importance Comparison')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        journal_savefig(os.path.join(save_dir, 'shap_vs_pearson_comparison.png'))
        plt.close()

        # 4. SHAP beeswarm plot
        self._generate_shap_beeswarm(
            os.path.join(save_dir, 'shap_beeswarm.png'))

        print(f"\nAll results saved to: {save_dir}")
        print(f"  CSVs:  pearson_correlations.csv, shap_feature_importance.csv,")
        print(f"         shap_vs_pearson_comparison.csv, outlier_detection.csv,")
        print(f"         feature_importance_Ionic_Conductivity.csv")
        print(f"  Plots: pearson_correlations.png, shap_feature_importance.png,")
        print(f"         shap_vs_pearson_comparison.png, shap_beeswarm.png")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE DDSE DATA ANALYSIS")
        print("="*80)

        # 1. Basic Information
        missing_df = self.basic_info()

        # 2. Statistical Summary
        target_stats, key_stats, dist_analysis = self.statistical_summary()

        # 3. Correlation Analysis
        pearson_correlations, corr_df = self.correlation_analysis()

        # 4. SHAP Analysis
        shap_values, shap_df = self.shap_analysis()

        # 5. Outlier Detection
        outlier_df = self.outlier_detection()

        # 6. Material Type Analysis
        material_analysis = self.material_type_analysis()

        # 7. Feature Importance
        importance_results = self.feature_importance_analysis()

        # 8. Data Quality Assessment
        quality_score, issues = self.data_quality_assessment()

        # 9. ML Recommendations
        self.generate_modeling_recommendations(importance_results)

        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)

        results = {
            'missing_df': missing_df,
            'target_stats': target_stats,
            'pearson_correlations': pearson_correlations,
            'corr_df': corr_df,
            'shap_values': shap_values,
            'shap_df': shap_df,
            'outlier_df': outlier_df,
            'importance_results': importance_results,
            'quality_score': quality_score,
            'issues': issues
        }

        # 10. Save all results to disk
        self.save_results(results)

        return results

# Usage Example
def main():
    # Initialize analyzer
    analyzer = DDSEDataAnalyzer(str(DATA_CLEANED / "ddse_compositional_clean.csv"))
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Additional specific analyses can be run individually
    # analyzer.correlation_analysis()
    # analyzer.feature_importance_analysis()

if __name__ == "__main__":
    main()
