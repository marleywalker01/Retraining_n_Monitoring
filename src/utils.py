"""
Shared utilities for all scripts
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Setup centralized logging
def setup_logging(name, log_file='logs/monitoring.log'):
    """Setup logging with file and console handlers"""
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Centralized artifact management
class ArtifactManager:
    """Manage all artifacts in centralized location"""
    
    def __init__(self, base_path='artifacts'):
        self.base_path = base_path
        self._create_dirs()
    
    def _create_dirs(self):
        """Create artifact subdirectories"""
        dirs = ['models', 'data', 'preprocessing', 'metrics', 'metadata']
        for d in dirs:
            os.makedirs(f'{self.base_path}/{d}', exist_ok=True)
    
    def save_model(self, model, name='model.keras'):
        """Save model to artifacts/models/"""
        path = f'{self.base_path}/models/{name}'
        model.save(path)
        return path
    
    def load_model(self, name='model.keras'):
        """Load model from artifacts/models/"""
        import tensorflow as tf
        path = f'{self.base_path}/models/{name}'
        return tf.keras.models.load_model(path, compile=False)
    
    def save_data(self, data, name):
        """Save numpy array to artifacts/data/"""
        path = f'{self.base_path}/data/{name}'
        np.save(path, data)
        return path
    
    def load_data(self, name):
        """Load numpy array from artifacts/data/"""
        path = f'{self.base_path}/data/{name}.npy'
        return np.load(path)
    
    def save_preprocessing(self, obj, name):
        """Save preprocessing object (scaler, encoder, etc.)"""
        path = f'{self.base_path}/preprocessing/{name}'
        joblib.dump(obj, path)
        return path
    
    def load_preprocessing(self, name):
        """Load preprocessing object"""
        path = f'{self.base_path}/preprocessing/{name}'
        return joblib.load(path)
    
    def save_metrics(self, metrics, name):
        """Save metrics as JSON"""
        path = f'{self.base_path}/metrics/{name}'
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return path
    
    def load_metrics(self, name):
        """Load metrics from JSON"""
        path = f'{self.base_path}/metrics/{name}'
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_metadata(self, key, value):
        """Save metadata key-value pair"""
        path = f'{self.base_path}/metadata/{key}.txt'
        with open(path, 'w') as f:
            f.write(str(value))
    
    def load_metadata(self, key):
        """Load metadata value"""
        path = f'{self.base_path}/metadata/{key}.txt'
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read().strip()
        return None

# Data quality monitoring
class DataQualityMonitor:
    """Monitor data quality metrics"""
    
    @staticmethod
    def check_missing_values(df):
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        return {
            'total_missing': int(missing.sum()),
            'missing_by_column': missing[missing > 0].to_dict(),
            'missing_percentage': missing_pct[missing_pct > 0].to_dict()
        }
    
    @staticmethod
    def check_data_types(df):
        """Check data types"""
        return df.dtypes.astype(str).to_dict()
    
    @staticmethod
    def check_statistics(df):
        """Calculate basic statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return df[numeric_cols].describe().to_dict()
        return {}
    
    @staticmethod
    def generate_report(df, name):
        """Generate complete data quality report"""
        return {
            'dataset': name,
            'shape': df.shape,
            'missing_values': DataQualityMonitor.check_missing_values(df),
            'data_types': DataQualityMonitor.check_data_types(df),
            'statistics': DataQualityMonitor.check_statistics(df)
        }

# Performance monitoring
class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, task='regression'):
        """Calculate performance metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if task == 'regression':
            return {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        else:
            y_pred_class = np.round(y_pred)
            return {
                'accuracy': float(accuracy_score(y_true, y_pred_class)),
                'precision': float(precision_score(y_true, y_pred_class, average='weighted')),
                'recall': float(recall_score(y_true, y_pred_class, average='weighted')),
                'f1': float(f1_score(y_true, y_pred_class, average='weighted'))
            }

# Notification system
class Notifier:
    """Send notifications for alerts"""
    
    @staticmethod
    def send_alert(alert_type, message, details=None):
        """Send alert via email/Slack (configure as needed)"""
        
        # Create alert log
        os.makedirs('monitoring/alerts', exist_ok=True)
        alert_file = f'monitoring/alerts/{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'details': details
        }
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        # Print to console
        print(f"\n⚠️ ALERT [{alert_type}]: {message}")
        
        # TODO: Add email/Slack integration
        # Example Slack webhook (uncomment and add your webhook URL)
        """
        import requests
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if webhook_url:
            requests.post(webhook_url, json={'text': f'*{alert_type}*: {message}'})
        """
        
        return alert_file