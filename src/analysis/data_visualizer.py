import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Rectangle
from src.config import DATA_CLEANED, OUTPUTS_DIR, journal_savefig
import warnings
warnings.filterwarnings('ignore')

# Create visualizations directory
VIZ_DIR = str(OUTPUTS_DIR / 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

# Load the data
df = pd.read_csv(str(DATA_CLEANED / 'ddse_compositional_clean.csv'))

sns.set_palette("husl")

# 1. Ionic Conductivity vs Activation Energy by Material Type
plt.figure(figsize=(12, 8))
material_types = df['Material_Type'].value_counts().head(10).index
df_top = df[df['Material_Type'].isin(material_types)]

scatter = plt.scatter(df_top['Ea_eV'], np.log10(df_top['Ionic_Conductivity']), 
                     c=pd.Categorical(df_top['Material_Type']).codes, 
                     alpha=0.6, s=50, cmap='tab20')
plt.xlabel('Activation Energy (eV)', fontsize=12)
plt.ylabel('Log₁₀(Ionic Conductivity) [S/cm]', fontsize=12)
plt.title('Ionic Conductivity vs Activation Energy by Material Type', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Material Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'conductivity_vs_activation_energy.png'))
plt.close()

# 2. Distribution of Ionic Conductivity by Material Type
plt.figure(figsize=(15, 8))
top_materials = df['Material_Type'].value_counts().head(8).index
df_subset = df[df['Material_Type'].isin(top_materials)]
sns.boxplot(data=df_subset, x='Material_Type', y='Ionic_Conductivity', log_scale=(False, True))
plt.xticks(rotation=45, ha='right')
plt.ylabel('Ionic Conductivity [S/cm] (log scale)', fontsize=12)
plt.xlabel('Material Type', fontsize=12)
plt.title('Distribution of Ionic Conductivity by Material Type', fontsize=14, fontweight='bold')
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'conductivity_distribution_by_material.png'))
plt.close()

# 3. Correlation Heatmap
numerical_cols = ['Temp_K', 'Ea_eV', 'Ionic_Conductivity', 'avg_electronegativity', 
                 'avg_atomic_mass', 'avg_ionic_radius', 'num_elements', 'li_fraction',
                 'composition_entropy', 'electronegativity_variance', 'li_to_anion_ratio']
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.1, fmt='.2f')
plt.title('Correlation Matrix of Electrolyte Properties', fontsize=14, fontweight='bold')
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'))
plt.close()

# 4. Temperature vs Ionic Conductivity
plt.figure(figsize=(10, 8))
plt.scatter(df['Temp_K'], np.log10(df['Ionic_Conductivity']), 
           alpha=0.5, s=30, c='steelblue')
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel('Log₁₀(Ionic Conductivity) [S/cm]', fontsize=12)
plt.title('Ionic Conductivity vs Temperature', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'conductivity_vs_temperature.png'))
plt.close()

# 5. Composition Entropy vs Ionic Conductivity
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['composition_entropy'], np.log10(df['Ionic_Conductivity']), 
                     c=df['num_elements'], alpha=0.6, s=40, cmap='viridis')
plt.xlabel('Composition Entropy', fontsize=12)
plt.ylabel('Log₁₀(Ionic Conductivity) [S/cm]', fontsize=12)
plt.title('Ionic Conductivity vs Composition Entropy', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Number of Elements')
plt.grid(True, alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'conductivity_vs_entropy.png'))
plt.close()

# 6. Material Type Distribution
plt.figure(figsize=(12, 8))
material_counts = df['Material_Type'].value_counts().head(15)
bars = plt.bar(range(len(material_counts)), material_counts.values, color='skyblue', edgecolor='navy')
plt.xlabel('Material Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Material Types in Dataset', fontsize=14, fontweight='bold')
plt.xticks(range(len(material_counts)), material_counts.index, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'material_type_distribution.png'))
plt.close()

# 7. Li Fraction vs Ionic Conductivity
plt.figure(figsize=(10, 8))
plt.scatter(df['li_fraction'], np.log10(df['Ionic_Conductivity']), 
           alpha=0.5, s=30, c='coral')
plt.xlabel('Li Fraction', fontsize=12)
plt.ylabel('Log₁₀(Ionic Conductivity) [S/cm]', fontsize=12)
plt.title('Ionic Conductivity vs Lithium Fraction', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'conductivity_vs_li_fraction.png'))
plt.close()

# 8. Multi-panel figure showing key relationships
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Panel 1: Conductivity vs Activation Energy
axes[0,0].scatter(df['Ea_eV'], np.log10(df['Ionic_Conductivity']), 
                 alpha=0.5, s=20, c='blue')
axes[0,0].set_xlabel('Activation Energy (eV)')
axes[0,0].set_ylabel('Log₁₀(Ionic Conductivity)')
axes[0,0].set_title('Conductivity vs Activation Energy')
axes[0,0].grid(True, alpha=0.3)

# Panel 2: Average electronegativity vs Conductivity
axes[0,1].scatter(df['avg_electronegativity'], np.log10(df['Ionic_Conductivity']), 
                 alpha=0.5, s=20, c='red')
axes[0,1].set_xlabel('Average Electronegativity')
axes[0,1].set_ylabel('Log₁₀(Ionic Conductivity)')
axes[0,1].set_title('Conductivity vs Electronegativity')
axes[0,1].grid(True, alpha=0.3)

# Panel 3: Number of elements vs Conductivity
axes[1,0].scatter(df['num_elements'], np.log10(df['Ionic_Conductivity']), 
                 alpha=0.5, s=20, c='green')
axes[1,0].set_xlabel('Number of Elements')
axes[1,0].set_ylabel('Log₁₀(Ionic Conductivity)')
axes[1,0].set_title('Conductivity vs Number of Elements')
axes[1,0].grid(True, alpha=0.3)

# Panel 4: Packing efficiency vs Conductivity
axes[1,1].scatter(df['packing_efficiency_proxy'], np.log10(df['Ionic_Conductivity']), 
                 alpha=0.5, s=20, c='purple')
axes[1,1].set_xlabel('Packing Efficiency Proxy')
axes[1,1].set_ylabel('Log₁₀(Ionic Conductivity)')
axes[1,1].set_title('Conductivity vs Packing Efficiency')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'multi_panel_analysis.png'))
plt.close()

# 9. High-conductivity materials analysis
high_conductivity = df[df['Ionic_Conductivity'] > 0.001]
plt.figure(figsize=(12, 8))
if len(high_conductivity) > 0:
    material_high = high_conductivity['Material_Type'].value_counts().head(10)
    bars = plt.bar(range(len(material_high)), material_high.values, 
                   color='lightgreen', edgecolor='darkgreen')
    plt.xlabel('Material Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Material Types with High Ionic Conductivity (>0.001 S/cm)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(material_high)), material_high.index, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'high_conductivity_materials.png'))
plt.close()

# 10. Activation Energy Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Ea_eV'], bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Activation Energy (eV)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Activation Energies', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'activation_energy_distribution.png'))
plt.close()

print("All visualizations have been saved to the 'visualizations' folder!")
print(f"Total number of data points: {len(df)}")
print(f"Number of unique material types: {df['Material_Type'].nunique()}")
print(f"Ionic conductivity range: {df['Ionic_Conductivity'].min():.2e} to {df['Ionic_Conductivity'].max():.2e} S/cm")

# 11. Ionic Conductivity Distribution
plt.figure(figsize=(10, 6))
plt.hist(np.log10(df['Ionic_Conductivity']), bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.xlabel('Log₁₀(Ionic Conductivity) [S/cm]', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Ionic Conductivities', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'ionic_conductivity_distribution.png'))
plt.close()

# 12. Plot cumulative frequency plot of log Ionic Conductivity with bin size of 1
plt.figure(figsize=(10, 6))
counts, bin_edges = np.histogram(np.log10(df['Ionic_Conductivity']), bins=50)
cumulative = np.cumsum(counts)
plt.step(bin_edges[1:], cumulative, where='post', color='purple', alpha=0.7)
plt.xlabel('Log₁₀(Ionic Conductivity) [S/cm]', fontsize=12)
plt.ylabel('Cumulative Frequency', fontsize=12)
plt.title('Cumulative Distribution of Ionic Conductivities', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add special markers at log(IC) = -2, -3, -4, -5, -6, -7
special_logs = [-2, -3, -4, -5, -6, -7]
for log_ic in special_logs:
    idx = np.searchsorted(bin_edges, log_ic, side='right') - 1
    if 0 <= idx < len(cumulative):
        plt.plot(log_ic, cumulative[idx], marker='o', color='red')
        plt.text(
            log_ic, cumulative[idx],
            f'{log_ic}\n({cumulative[idx]})',
            color='red', fontsize=10, ha='center', va='bottom'
        )

plt.tight_layout()
journal_savefig(os.path.join(VIZ_DIR, 'ionic_conductivity_cumulative_distribution.png'))
plt.close()