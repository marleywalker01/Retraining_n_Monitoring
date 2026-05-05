"""
Comprehensive monitoring: drift, performance, data quality
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from utils import ArtifactManager, PerformanceMonitor, DataQualityMonitor, Notifier, setup_logging

# Setup logging
logger = setup_logging('monitoring')

def load_all_data():
    """Load all datasets from centralized artifacts"""
    artifact_mgr = ArtifactManager()
    
    logger.info("Loading data from artifacts...")
    
    # Load test data
    X_test = artifact_mgr.load_data('X_test')
    y_test = artifact_mgr.load_data('y_test')
    logger.info(f"✅ Test data: {X_test.shape}")
    
    # Load training data if available
    X_train = None
    y_train = None
    if os.path.exists(f'{artifact_mgr.base_path}/data/X_train.npy'):
        X_train = artifact_mgr.load_data('X_train')
        y_train = artifact_mgr.load_data('y_train')
        logger.info(f"✅ Training data: {X_train.shape}")
    
    # Load new data if available
    X_new = None
    y_new = None
    if os.path.exists(f'{artifact_mgr.base_path}/data/X_new.npy'):
        X_new = artifact_mgr.load_data('X_new')
        y_new = artifact_mgr.load_data('y_new')
        logger.info(f"✅ New data: {X_new.shape}")
    else:
        X_new = X_test.copy()
        y_new = y_test.copy()
        logger.warning("No new data, using test data as reference")
    
    # Load model
    model = artifact_mgr.load_model()
    logger.info("✅ Model loaded")
    
    return model, X_train, y_train, X_test, y_test, X_new, y_new

def detect_data_drift(reference_data, current_data, threshold=0.1):
    """Detect data drift between reference and current datasets"""
    from scipy.stats import ks_2samp
    
    drift_results = {}
    drifted_features = []
    
    for i in range(reference_data.shape[1]):
        # Kolmogorov-Smirnov test for each feature
        stat, p_value = ks_2samp(reference_data[:, i], current_data[:, i])
        
        if p_value < threshold:
            drifted_features.append(i)
            drift_results[f'feature_{i}'] = {
                'ks_statistic': float(stat),
                'p_value': float(p_value),
                'drift_detected': True
            }
        else:
            drift_results[f'feature_{i}'] = {
                'ks_statistic': float(stat),
                'p_value': float(p_value),
                'drift_detected': False
            }
    
    drift_percentage = len(drifted_features) / reference_data.shape[1]
    
    return {
        'drift_detected': drift_percentage > 0.3,  # 30% features drifted
        'drift_percentage': drift_percentage,
        'drifted_features': len(drifted_features),
        'total_features': reference_data.shape[1],
        'feature_details': drift_results
    }

def generate_monitoring_dashboard(metrics, drift_report, quality_report):
    """Generate HTML dashboard for monitoring"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Monitoring Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ background: #f0f0f0; padding: 15px; margin: 10px; border-radius: 5px; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .critical {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>ML Monitoring Dashboard</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="metric">
            <h2>📊 Model Performance</h2>
            <table>
    """
    
    for metric, value in metrics.items():
        color = 'good' if value > 0.7 else ('warning' if value > 0.5 else 'critical')
        html += f"<tr><td>{metric}</td><td class='{color}'>{value:.4f}</td></tr>"
    
    html += """
            </table>
        </div>
        
        <div class="metric">
            <h2>📈 Data Drift Report</h2>
    """
    
    if drift_report['drift_detected']:
        html += f"<p class='critical'>⚠️ DRIFT DETECTED: {drift_report['drift_percentage']*100:.1f}% of features have drifted</p>"
    else:
        html += f"<p class='good'>✅ No significant drift detected</p>"
    
    html += f"""
            <p>Drifted features: {drift_report['drifted_features']}/{drift_report['total_features']}</p>
        </div>
        
        <div class="metric">
            <h2>✅ Data Quality Report</h2>
            <p>Missing values: {quality_report['missing_values']['total_missing']}</p>
            <p>Dataset shape: {quality_report['shape']}</p>
        </div>
    </body>
    </html>
    """
    
    # Save dashboard
    os.makedirs('reports', exist_ok=True)
    with open('reports/monitoring_dashboard.html', 'w') as f:
        f.write(html)
    
    logger.info("✅ Monitoring dashboard saved to reports/monitoring_dashboard.html")

def main():
    """Main monitoring function"""
    
    logger.info("="*70)
    logger.info("🔍 STARTING COMPREHENSIVE MONITORING")
    logger.info("="*70)
    
    artifact_mgr = ArtifactManager()
    
    # Load all data
    model, X_train, y_train, X_test, y_test, X_new, y_new = load_all_data()
    
    # Get predictions
    logger.info("Getting model predictions...")
    y_test_pred = model.predict(X_test, verbose=0).flatten()
    y_new_pred = model.predict(X_new, verbose=0).flatten()
    
    # Determine task type
    is_regression = len(np.unique(y_test)) > 10
    
    # Calculate performance metrics
    logger.info("Calculating performance metrics...")
    test_metrics = PerformanceMonitor.calculate_metrics(y_test, y_test_pred, 
                                                          'regression' if is_regression else 'classification')
    new_metrics = PerformanceMonitor.calculate_metrics(y_new, y_new_pred,
                                                         'regression' if is_regression else 'classification')
    
    logger.info(f"Test performance: {test_metrics}")
    logger.info(f"New data performance: {new_metrics}")
    
    # Save metrics
    artifact_mgr.save_metrics(test_metrics, 'test_metrics.json')
    artifact_mgr.save_metrics(new_metrics, 'new_metrics.json')
    
    # Detect data drift
    logger.info("Detecting data drift...")
    if X_train is not None:
        drift_report = detect_data_drift(X_train.reshape(X_train.shape[0], -1), 
                                         X_test.reshape(X_test.shape[0], -1))
    else:
        drift_report = detect_data_drift(X_test.reshape(X_test.shape[0], -1),
                                         X_new.reshape(X_new.shape[0], -1))
    
    logger.info(f"Drift report: {drift_report['drift_percentage']*100:.1f}% features drifted")
    
    # Save drift report
    with open('reports/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=2)
    
    # Check data quality
    logger.info("Checking data quality...")
    quality_report = DataQualityMonitor.generate_report(pd.DataFrame(X_test.reshape(X_test.shape[0], -1)), "test_data")
    
    # Calculate performance change
    if is_regression:
        perf_change = (new_metrics['r2'] - test_metrics['r2']) / (abs(test_metrics['r2']) + 1e-6)
    else:
        perf_change = (new_metrics['accuracy'] - test_metrics['accuracy']) / (abs(test_metrics['accuracy']) + 1e-6)
    
    # Determine if retraining is needed
    retrain_needed = False
    alerts = []
    
    # Check performance degradation
    if perf_change < -0.15:
        retrain_needed = True
        alerts.append(f"Performance degraded by {abs(perf_change)*100:.1f}%")
    
    # Check data drift
    if drift_report['drift_detected']:
        retrain_needed = True
        alerts.append(f"Data drift detected: {drift_report['drift_percentage']*100:.1f}% features drifted")
    
    # Check data quality
    if quality_report['missing_values']['total_missing'] > 0:
        alerts.append(f"Found {quality_report['missing_values']['total_missing']} missing values")
    
    # Send alerts if needed
    if alerts:
        alert_message = " | ".join(alerts)
        logger.warning(f"⚠️ ALERTS: {alert_message}")
        Notifier.send_alert("monitoring_alert", alert_message, {
            'perf_change': float(perf_change),
            'drift_percentage': drift_report['drift_percentage'],
            'timestamp': datetime.now().isoformat()
        })
        
        if retrain_needed:
            with open('retrain_needed.txt', 'w') as f:
                f.write('true')
    else:
        logger.info("✅ No issues detected")
        with open('retrain_needed.txt', 'w') as f:
            f.write('false')
    
    # Generate dashboard
    generate_monitoring_dashboard(new_metrics, drift_report, quality_report)
    
    # Save monitoring summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_metrics': test_metrics,
        'new_metrics': new_metrics,
        'performance_change': float(perf_change),
        'drift_report': drift_report,
        'quality_report': quality_report,
        'retrain_needed': retrain_needed,
        'alerts': alerts
    }
    
    artifact_mgr.save_metrics(summary, 'monitoring_summary.json')
    
    logger.info("="*70)
    logger.info(f"✅ Monitoring complete. Retrain needed: {retrain_needed}")
    logger.info("="*70)
    
    return retrain_needed

if __name__ == "__main__":
    try:
        retrain_needed = main()
        sys.exit(1 if retrain_needed else 0)
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(0)