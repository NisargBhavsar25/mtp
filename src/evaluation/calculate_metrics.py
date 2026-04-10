import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import DATA_CLEANED

def calculate_metrics(y_true, y_pred):
    """Calculate all regression metrics."""
    n = len(y_true)

    # Number of predictors (assuming 1 for simple regression)
    p = 1

    # Mean True Value (MTV)
    mtv = np.mean(y_true)

    # Mean Predicted Value (MPV)
    mpv = np.mean(y_pred)

    # Residuals/Errors
    errors = y_pred - y_true

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(errors))

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(errors**2))

    # Mean Bias Error (MBE)
    mbe = np.mean(errors)

    # Standard Deviation of errors (STD)
    std = np.std(errors, ddof=1)

    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)

    # Adjusted R-squared
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    return {
        'N': n,
        'R_adj^2': r2_adj,
        'MTV': mtv,
        'MPV': mpv,
        'MAE': mae,
        'RMSE': rmse,
        'MBE': mbe,
        'STD': std
    }

def main():
    # Get all CSV files in the current directory
    csv_files = list(DATA_CLEANED.glob('*.csv'))

    results = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Check if required columns exist
            if 'log10_target' in df.columns and 'log10_predict' in df.columns:
                y_true = df['log10_target'].values
                y_pred = df['log10_predict'].values

                # Remove any NaN values
                mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                y_true = y_true[mask]
                y_pred = y_pred[mask]

                if len(y_true) > 0:
                    metrics = calculate_metrics(y_true, y_pred)
                    metrics['Data set'] = csv_file.stem
                    results.append(metrics)
                    print(f"Processed: {csv_file.name}")
                else:
                    print(f"Skipped (no valid data): {csv_file.name}")
            else:
                print(f"Skipped (missing columns): {csv_file.name}")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

    if results:
        # Create DataFrame with results
        results_df = pd.DataFrame(results)

        # Reorder columns
        cols = ['Data set', 'N', 'R_adj^2', 'MTV', 'MPV', 'MAE', 'RMSE', 'MBE', 'STD']
        results_df = results_df[cols]

        # Print results
        print("\n" + "="*100)
        print("METRICS SUMMARY")
        print("="*100)
        print(results_df.to_string(index=False))

        # Save to CSV
        output_file = DATA_CLEANED / 'metrics_summary.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Also print in a formatted table
        print("\n" + "="*100)
        print("FORMATTED TABLE")
        print("="*100)
        print(f"{'Data set':<40} {'N':>6} {'R_adj^2':>10} {'MTV':>10} {'MPV':>10} {'MAE':>10} {'RMSE':>10} {'MBE':>10} {'STD':>10}")
        print("-"*100)
        for _, row in results_df.iterrows():
            print(f"{row['Data set']:<40} {row['N']:>6} {row['R_adj^2']:>10.4f} {row['MTV']:>10.4f} {row['MPV']:>10.4f} {row['MAE']:>10.4f} {row['RMSE']:>10.4f} {row['MBE']:>10.4f} {row['STD']:>10.4f}")
    else:
        print("No valid CSV files found with required columns.")

if __name__ == "__main__":
    main()
