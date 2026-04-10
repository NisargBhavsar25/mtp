import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
import os
from src.config import DATA_CLEANED, OUTPUTS_DIR, journal_savefig

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    Filters out zero values in y_true to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.inf
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def regression_analysis_multiple(datasets_info, output_dir="combined_regression_plots"):
    """
    Performs regression analysis on multiple datasets and generates overlaid plots.

    Args:
        datasets_info (list of dicts): A list where each dict contains info for one dataset.
                                      Keys: 'file', 'target_col', 'result_col', 'name'.
        output_dir (str): The directory where plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    all_data = []
    # Use a color palette for distinction
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets_info)))

    # --- Data Loading and Metrics Calculation ---
    for i, info in enumerate(datasets_info):
        try:
            df = pd.read_csv(info['file'])
        except FileNotFoundError:
            print(f"Error: The file '{info['file']}' was not found. Skipping.")
            continue

        if not all(col in df.columns for col in [info['target_col'], info['result_col']]):
            print(f"Error: Columns not found in '{info['file']}'. Skipping.")
            continue

        y_true = df[info['target_col']].values
        y_pred = df[info['result_col']].values
        residuals = y_pred - y_true
        
        all_data.append({
            "name": info['name'],
            "y_true": y_true,
            "y_pred": y_pred,
            "residuals": residuals,
            "color": colors[i],
            "metrics": {
                "R²": r2_score(y_true, y_pred),
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAPE": mean_absolute_percentage_error(y_true, y_pred)
            }
        })

    # --- Plot 1: Overlaid Residual Distributions (KDE) ---
    plt.figure(figsize=(12, 7))
    for data in all_data:
        sns.kdeplot(data['residuals'], label=data['name'], color=data['color'], fill=True, alpha=0.3, linewidth=2)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Overlaid Residual Distributions', fontsize=16, pad=20)
    plt.xlabel('Residuals (Predicted - Actual log₁₀(Ionic Conductivity))', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Dataset')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    journal_savefig(os.path.join(output_dir, '1_overlaid_residual_distributions.png'))
    plt.close()

    # --- Plot 2: Overlaid Parity Plot ---
    plt.figure(figsize=(10, 10))
    # Determine plot limits to encompass all data
    min_val_all = min([d['y_true'].min() for d in all_data] + [d['y_pred'].min() for d in all_data])
    max_val_all = max([d['y_true'].max() for d in all_data] + [d['y_pred'].max() for d in all_data])
    
    plt.plot([min_val_all, max_val_all], [min_val_all, max_val_all], 'k--', lw=2, label='Ideal Fit')

    for data in all_data:
        plt.scatter(data['y_true'], data['y_pred'], label=data['name'], color=data['color'], alpha=0.6, edgecolors='w', linewidth=0.5)

    plt.title('Parity Plot: Predicted vs. Actual', fontsize=16, pad=20)
    plt.xlabel('Actual log₁₀(Ionic Conductivity)', fontsize=12)
    plt.ylabel('Predicted log₁₀(Ionic Conductivity)', fontsize=12)
    plt.legend(title='Dataset')
    plt.axis('equal')
    plt.xlim(min_val_all, max_val_all)
    plt.ylim(min_val_all, max_val_all)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    journal_savefig(os.path.join(output_dir, '2_overlaid_parity_plot.png'))
    plt.close()

    # --- Plot 3: Overlaid Residuals vs. Predicted Values ---
    plt.figure(figsize=(12, 7))
    for data in all_data:
        plt.scatter(data['y_pred'], data['residuals'], label=data['name'], color=data['color'], alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals vs. Predicted Values', fontsize=16, pad=20)
    plt.xlabel('Predicted log₁₀(Ionic Conductivity)', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.legend(title='Dataset')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    journal_savefig(os.path.join(output_dir, '3_overlaid_residuals_vs_predicted.png'))
    plt.close()

    # --- Plot 4: Q-Q Plots (Subplots) ---
    fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 5), sharey=True)
    if len(all_data) == 1: axes = [axes] # Ensure axes is always iterable
        
    for i, data in enumerate(all_data):
        stats.probplot(data['residuals'], dist="norm", plot=axes[i])
        axes[i].set_title(data['name'], fontsize=14)
        axes[i].get_lines()[0].set_markerfacecolor(data['color'])
        axes[i].get_lines()[0].set_markeredgecolor(data['color'])
    fig.suptitle('Q-Q Plots of Residuals (Normality Check)', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    journal_savefig(os.path.join(output_dir, '4_qq_plots.png'))
    plt.close()

    # --- Print Metrics ---
    metrics_df = pd.DataFrame([{
        "Dataset": d['name'],
        "R²": f"{d['metrics']['R²']:.4f}",
        "MAE": f"{d['metrics']['MAE']:.4f}",
        "RMSE": f"{d['metrics']['RMSE']:.4f}",
        "MAPE (%)": f"{d['metrics']['MAPE']:.2f}"
    } for d in all_data])

    # add a row for overall metrics
    overall_y_true = np.concatenate([d['y_true'] for d in all_data])
    overall_y_pred = np.concatenate([d['y_pred'] for d in all_data])
    overall_metrics = {
        "Dataset": "Overall",
        "R²": f"{r2_score(overall_y_true, overall_y_pred):.4f}",
        "MAE": f"{mean_absolute_error(overall_y_true, overall_y_pred):.4f}",    
        "RMSE": f"{np.sqrt(mean_squared_error(overall_y_true, overall_y_pred)):.4f}",
        "MAPE (%)": f"{mean_absolute_percentage_error(overall_y_true, overall_y_pred):.2f}"
    }
    metrics_df = pd.concat([metrics_df, pd.DataFrame([overall_metrics])], ignore_index=True)
    

    print("\n--- Comparative Regression Metrics ---")
    print(metrics_df.to_string(index=False))
    # Save metrics to a CSV file for easy access
    metrics_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    
    print(f"\nAnalysis complete. All plots and metrics summary saved in '{output_dir}'.")


if __name__ == '__main__':
    # --- USER INPUT AREA ---
    # Please modify the list below with your file details.
    # You can add more dictionaries to the list to process more than 3 files.
    
    # NOTE: After retraining with cleaned DDSE data, re-run predictions on
    # these validation sets and update the result_col names if needed.
    datasets = [
        {
            "file": str(DATA_CLEANED / "LLZO_clean.csv"),
            "target_col": "log10_target",
            "result_col": "log10_predict",
            "name": "LLZO-focused Dataset"
        },
        {
            "file": str(DATA_CLEANED / "Sendek_clean.csv"),
            "target_col": "log10_target",
            "result_col": "log10_predict",
            "name": "Sendek Dataset"
        }
    ]

    # Run the analysis
    regression_analysis_multiple(datasets, output_dir=str(OUTPUTS_DIR / "combined_regression_plots"))
