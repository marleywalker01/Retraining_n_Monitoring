"""
Preprocess new data for model prediction and retraining
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from utils import ArtifactManager, DataQualityMonitor, Notifier, setup_logging

# Setup logging
logger = setup_logging('preprocessing')

def preprocess_new_data():
    """Preprocess new data using saved artifacts from training"""
    
    logger.info("="*70)
    logger.info("🔄 PREPROCESSING NEW DATA FOR RETRAINING")
    logger.info("="*70)
    
    artifact_mgr = ArtifactManager()
    
    # Check if new data exists
    new_data_path = 'data/new_data.csv'
    if not os.path.exists(new_data_path):
        logger.warning("No new data found at data/new_data.csv")
        return False
    
    # Load new data
    logger.info(f"Loading new data from {new_data_path}...")
    df = pd.read_csv(new_data_path)
    logger.info(f"✅ Loaded {len(df)} rows with {df.shape[1]} columns")
    
    # Data quality check
    quality_report = DataQualityMonitor.generate_report(df, "new_data")
    artifact_mgr.save_metrics(quality_report, 'data_quality_new.json')
    
    if quality_report['missing_values']['total_missing'] > 0:
        logger.warning(f"Found {quality_report['missing_values']['total_missing']} missing values")
        
        # Handle missing values
        for col, count in quality_report['missing_values']['missing_by_column'].items():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
                logger.info(f"  Filled {col} with median")
            else:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', inplace=True)
                logger.info(f"  Filled {col} with mode")
    
    # Load feature columns
    logger.info("Loading feature columns from training...")
    feature_columns = artifact_mgr.load_preprocessing('feature_columns.json')
    logger.info(f"✅ Loaded {len(feature_columns)} feature columns")
    
    # Extract target
    logger.info("Extracting target variable...")
    if 'target' in df.columns:
        y_new = df['target'].values
        X_df = df.drop('target', axis=1)
    elif 'y' in df.columns:
        y_new = df['y'].values
        X_df = df.drop('y', axis=1)
    else:
        y_new = df.iloc[:, -1].values
        X_df = df.iloc[:, :-1]
    
    # Match features
    available_features = [col for col in feature_columns if col in X_df.columns]
    missing_features = [col for col in feature_columns if col not in X_df.columns]
    
    logger.info(f"Available features: {len(available_features)}/{len(feature_columns)}")
    
    if missing_features:
        logger.warning(f"Missing {len(missing_features)} features, filling with 0")
        for col in missing_features:
            X_df[col] = 0
    
    # Ensure correct order
    X_new = X_df[feature_columns].values
    
    # Handle NaN
    if np.isnan(X_new).any():
        logger.warning(f"Found NaN in features, filling with 0")
        X_new = np.nan_to_num(X_new, nan=0.0)
    
    # Apply scaler
    logger.info("Applying scaler...")
    if os.path.exists(f'{artifact_mgr.base_path}/preprocessing/scaler.pkl'):
        scaler = artifact_mgr.load_preprocessing('scaler.pkl')
        X_new_scaled = scaler.transform(X_new)
        logger.info("✅ Scaler applied")
    else:
        X_new_scaled = X_new
        logger.warning("No scaler found, using raw values")
    
    # Reshape for CNN
    logger.info("Reshaping for model...")
    import tensorflow as tf
    model = artifact_mgr.load_model()
    expected_shape = model.input_shape
    X_new_cnn = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
    logger.info(f"✅ Reshaped to {X_new_cnn.shape}")
    
    # Save preprocessed data
    logger.info("Saving preprocessed data...")
    artifact_mgr.save_data(X_new_cnn, 'X_new')
    artifact_mgr.save_data(y_new, 'y_new')
    logger.info(f"✅ Saved to artifacts/data/")
    
    # Combine with training data for retraining
    logger.info("Preparing combined dataset for retraining...")
    if os.path.exists('train/train.csv'):
        train_df = pd.read_csv('train/train.csv')
        logger.info(f"Original training data: {len(train_df)} rows")
        
        combined_df = pd.concat([train_df, df], ignore_index=True)
        combined_df = combined_df.dropna()
        logger.info(f"Combined dataset: {len(combined_df)} rows")
        
        combined_df.to_csv('train/train_combined.csv', index=False)
        
        # Backup original and replace
        import shutil
        shutil.copy('train/train.csv', 'train/train_backup.csv')
        shutil.copy('train/train_combined.csv', 'train/train.csv')
        logger.info("✅ Updated train/train.csv with new data")
    
    logger.info("="*70)
    logger.info("✅ New data preprocessing complete!")
    
    return True

if __name__ == "__main__":
    success = preprocess_new_data()
    sys.exit(0 if success else 1)