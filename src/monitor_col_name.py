import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def detect_drift(train_path="train/train.csv", new_data_path="data/new_data.csv"):
    """
    Simple drift detection by comparing feature distributions
    """
    try:
        # Load training (reference) data
        train_df = pd.read_csv(train_path)
        
        # Load new data that triggered the retraining
        new_df = pd.read_csv(new_data_path)
    except FileNotFoundError:
        # No new data to check
        with open("drift_status.txt", "w") as f:
            f.write("false")
        return
    
    # Get common columns
    common_cols = [col for col in train_df.columns if col in new_df.columns]
    
    if len(common_cols) == 0:
        print("No common columns to compare!")
        with open("drift_status.txt", "w") as f:
            f.write("false")
        return
    
    # Check drift for each numeric column
    drift_scores = []
    for col in common_cols:
        if train_df[col].dtype in ['float64', 'int64']:
            # Compare means (simple drift metric)
            train_mean = train_df[col].mean()
            new_mean = new_df[col].mean()
            
            # If means differ by more than 20%, flag as drift
            if abs(train_mean - new_mean) / (abs(train_mean) + 0.001) > 0.2:
                drift_scores.append(1)
                print(f"⚠️ Drift detected in column: {col}")
                print(f"   Train mean: {train_mean:.3f}, New mean: {new_mean:.3f}")
            else:
                drift_scores.append(0)dvc
    
    # Determine if significant drift occurred
    drift_percentage = sum(drift_scores) / len(drift_scores) if drift_scores else 0
    drift_detected = drift_percentage > 0.3  # If >30% of columns have drift
    
    # Save result for GitHub Actions
    with open("drift_status.txt", "w") as f:
        f.write(str(drift_detected).lower())
    
    if drift_detected:
        print("✅ DRIFT DETECTED! Notification will be sent.")
    else:
        print("No significant drift detected.")
    
    return drift_detected

if __name__ == "__main__":
    detect_drift()