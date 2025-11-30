"""
Flask Web Application for Dual-Model IDS System

This application provides a web interface for:
1. Uploading CSV files with network traffic data
2. Preprocessing and feature engineering
3. Binary and multiclass classification
4. Confidence-based routing and decision making
5. Real-time visualization of results
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path
import json
from io import BytesIO
import logging
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Import IDS system components
from ids_confidence_routing_system import (
    DualModelIDS, BatchIDS, ConfidenceBasedRouter,
    ConfidenceLevel, ActionPriority
)

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
ALLOWED_EXTENSIONS = {'csv'}

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['RESULTS_FOLDER']).mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class IDSConfig:
    """IDS System Configuration"""
    
    # Model paths (update these with your actual model paths)
    BINARY_MODEL_PATH = "data/models/binary_best_model.pkl"
    MULTICLASS_MODEL_PATH = "data/models/multiclass_best_model.pkl"
    SCALER_PATH = "data/models/scaler.pkl"
    PCA_PATH = "data/models/pca.pkl"
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    
    # Processing parameters
    BATCH_SIZE = 100  # Process samples in batches
    MAX_SAMPLES = 10000  # Maximum samples per upload
    
    # Feature columns (adjust based on your dataset)
    LABEL_COLUMNS = ['label1', 'label2', 'label3', 'label4', 'label_full']
    
    @staticmethod
    def get_feature_columns(df):
        """Extract feature columns from DataFrame"""
        return [col for col in df.columns 
                if not any(label in col for label in IDSConfig.LABEL_COLUMNS)]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_ids_system():
    """Load trained IDS models"""
    try:
        ids = DualModelIDS(
            binary_model_path=IDSConfig.BINARY_MODEL_PATH,
            multiclass_model_path=IDSConfig.MULTICLASS_MODEL_PATH,
            scaler_path=IDSConfig.SCALER_PATH,
            pca_model_path=IDSConfig.PCA_PATH
        )
        return ids, None
    except Exception as e:
        error_msg = f"Error loading IDS models: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def preprocess_csv(file_path):
    """Load and validate CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate required columns
        if 'label1' not in df.columns:
            return None, "Missing 'label1' column"
        
        if 'label2' not in df.columns:
            return None, "Missing 'label2' column"
        
        # Limit samples
        if len(df) > IDSConfig.MAX_SAMPLES:
            logger.warning(f"Limiting samples from {len(df)} to {IDSConfig.MAX_SAMPLES}")
            df = df.head(IDSConfig.MAX_SAMPLES)
        
        return df, None
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"


def preprocess_user_features(df):
    """
    Apply the same preprocessing pipeline as used during model training.
    
    Steps:
    1. Extract features (remove label columns)
    2. Handle categorical features (one-hot encoding for low-cardinality)
    3. Remove high-correlated features (correlation > 0.95)
    4. Apply StandardScaler normalization
    5. Apply variance threshold filtering
    6. Features will be PCA-transformed inside the model
    
    Returns: preprocessed features array, or error message
    """
    try:
        # Step 1: Extract features (all columns except labels)
        feature_cols = IDSConfig.get_feature_columns(df)
        if len(feature_cols) == 0:
            return None, "No feature columns found"
        
        X = df[feature_cols].copy()
        logger.info(f"Extracted {len(feature_cols)} features from {len(X)} samples")
        
        # Step 2: Check for missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values, filling with median")
            X = X.fillna(X.median())
        
        # Step 3: Handle categorical features (one-hot encoding for low-cardinality)
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            if unique_count <= 20:  # Low-cardinality: one-hot encode
                logger.info(f"One-hot encoding column '{col}' ({unique_count} unique values)")
                encoded = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = X.drop(columns=[col])
                X = pd.concat([X, encoded], axis=1)
            else:  # High-cardinality: drop column
                logger.warning(f"Dropping high-cardinality column '{col}' ({unique_count} unique values)")
                X = X.drop(columns=[col])
        
        # Step 4: Remove non-numeric columns if any remain
        X = X.select_dtypes(include=[np.number])
        logger.info(f"Features after categorical handling: {X.shape[1]}")
        
        # Step 5: Remove highly correlated features (correlation > 0.95)
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find columns with correlation > 0.95
        drop_cols = []
        for column in upper.columns:
            if any(upper[column] > 0.95):
                drop_cols.append(column)
        
        if len(drop_cols) > 0:
            logger.info(f"Dropping {len(drop_cols)} highly correlated features")
            X = X.drop(columns=drop_cols)
        
        logger.info(f"Features after correlation filtering: {X.shape[1]}")
        
        # Step 6: Apply variance threshold (remove near-zero variance features)
        var_threshold = VarianceThreshold(threshold=0.01)
        X_var_filtered = var_threshold.fit_transform(X)
        X = pd.DataFrame(X_var_filtered, columns=[f"f_{i}" for i in range(X_var_filtered.shape[1])])
        logger.info(f"Features after variance filtering: {X.shape[1]}")
        
        return X.values, None
        
    except Exception as e:
        error_msg = f"Error preprocessing features: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def get_feature_matrix(df):
    """Extract feature matrix from DataFrame (legacy)"""
    feature_cols = IDSConfig.get_feature_columns(df)
    if len(feature_cols) == 0:
        return None, "No feature columns found"
    
    X = df[feature_cols].values
    return X, None


# ============================================================================
# FLASK ROUTES - STATIC PAGES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    """CSV upload page"""
    return render_template('upload.html')


@app.route('/results')
def results_page():
    """Results visualization page"""
    return render_template('results.html')


@app.route('/architecture')
def architecture_page():
    """System architecture documentation page"""
    return render_template('architecture.html')


@app.route('/guide')
def guide_page():
    """Deployment guide page"""
    return render_template('guide.html')


# ============================================================================
# FLASK ROUTES - API ENDPOINTS
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload and process CSV file"""
    
    try:
        # Check file presence
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Only CSV files allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {safe_filename}")
        
        # Preprocess CSV
        df, error = preprocess_csv(file_path)
        if error:
            return jsonify({'status': 'error', 'message': error}), 400
        
        # Get initial statistics
        stats = {
            'total_samples': len(df),
            'total_features': len(IDSConfig.get_feature_columns(df)),
            'label1_distribution': df['label1'].value_counts().to_dict(),
            'label2_samples': df['label2'].nunique()
        }
        
        logger.info(f"CSV preprocessed: {stats}")
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded and preprocessed',
            'file_id': safe_filename,
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/process/<file_id>', methods=['POST'])
def api_process(file_id):
    """Process uploaded file with IDS system"""
    
    try:
        # Load CSV
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        df, error = preprocess_csv(file_path)
        if error:
            return jsonify({'status': 'error', 'message': error}), 400
        
        # Load IDS system
        ids, error = load_ids_system()
        if error:
            return jsonify({'status': 'error', 'message': error}), 500
        
        logger.info(f"Processing file {file_id} with {len(df)} samples")
        
        # APPLY FULL PREPROCESSING PIPELINE (matching training phase)
        X, error = preprocess_user_features(df)
        if error:
            return jsonify({'status': 'error', 'message': error}), 400
        
        logger.info(f"Features preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Process samples
        results = []
        predictions = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for idx, features in enumerate(X):
            sample_id = f"sample_{idx}"
            decision = ids.detect_and_route(sample_id, features)
            
            result = {
                'sample_id': sample_id,
                'index': idx,
                'confidence_level': decision.confidence_level.value,
                'classification': decision.classification,
                'attack_type': decision.attack_type,
                'confidence_score': float(decision.confidence_score),
                'priority': decision.priority.value,
                'actions_count': len(decision.actions)
            }
            
            results.append(result)
            predictions.append(decision.confidence_level.value)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Generate statistics
        stats = {
            'total_processed': len(results),
            'high_confidence': len(results_df[results_df['confidence_level'] == 'HIGH']),
            'medium_confidence': len(results_df[results_df['confidence_level'] == 'MEDIUM']),
            'low_confidence': len(results_df[results_df['confidence_level'] == 'LOW']),
            'average_confidence': float(results_df['confidence_score'].mean()),
            'max_confidence': float(results_df['confidence_score'].max()),
            'min_confidence': float(results_df['confidence_score'].min()),
            'attacks_detected': len(results_df[results_df['confidence_level'].isin(['HIGH', 'MEDIUM'])]),
            'attack_distribution': results_df[results_df['confidence_level'] == 'HIGH']['attack_type'].value_counts().to_dict(),
            'priority_distribution': results_df['priority'].value_counts().to_dict()
        }
        
        # Save results
        result_id = f"results_{timestamp}"
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_id)
        Path(result_dir).mkdir(exist_ok=True)
        
        # Save CSV results
        results_csv = os.path.join(result_dir, 'predictions.csv')
        results_df.to_csv(results_csv, index=False)
        
        # Save statistics
        stats_json = os.path.join(result_dir, 'statistics.json')
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing complete for {file_id}: {stats}")
        
        return jsonify({
            'status': 'success',
            'message': 'Processing complete',
            'result_id': result_id,
            'statistics': stats,
            'predictions': results
        }), 200
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/results/<result_id>', methods=['GET'])
def api_get_results(result_id):
    """Get processing results"""
    
    try:
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_id)
        
        # Load statistics
        stats_file = os.path.join(result_dir, 'statistics.json')
        if not os.path.exists(stats_file):
            return jsonify({'status': 'error', 'message': 'Results not found'}), 404
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        # Load predictions
        predictions_file = os.path.join(result_dir, 'predictions.csv')
        predictions_df = pd.read_csv(predictions_file)
        
        return jsonify({
            'status': 'success',
            'statistics': stats,
            'predictions': predictions_df.head(100).to_dict('records')
        }), 200
        
    except Exception as e:
        logger.error(f"Results retrieval error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/download/<result_id>', methods=['GET'])
def api_download_results(result_id):
    """Download results as CSV"""
    
    try:
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], result_id)
        predictions_file = os.path.join(result_dir, 'predictions.csv')
        
        if not os.path.exists(predictions_file):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        return send_file(
            predictions_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{result_id}_predictions.csv'
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/system-status', methods=['GET'])
def api_system_status():
    """Get system status and configuration"""
    
    try:
        # Check model files
        models_available = {
            'binary': os.path.exists(IDSConfig.BINARY_MODEL_PATH),
            'multiclass': os.path.exists(IDSConfig.MULTICLASS_MODEL_PATH),
            'scaler': os.path.exists(IDSConfig.SCALER_PATH),
            'pca': os.path.exists(IDSConfig.PCA_PATH)
        }
        
        all_models_available = all(models_available.values())
        
        # Get upload folder stats
        upload_count = len(os.listdir(app.config['UPLOAD_FOLDER']))
        results_count = len(os.listdir(app.config['RESULTS_FOLDER']))
        
        return jsonify({
            'status': 'success',
            'system_status': 'ready' if all_models_available else 'missing_models',
            'models': models_available,
            'configuration': {
                'high_confidence_threshold': IDSConfig.HIGH_CONFIDENCE_THRESHOLD,
                'medium_confidence_threshold': IDSConfig.MEDIUM_CONFIDENCE_THRESHOLD,
                'batch_size': IDSConfig.BATCH_SIZE,
                'max_samples': IDSConfig.MAX_SAMPLES
            },
            'statistics': {
                'uploaded_files': upload_count,
                'processed_results': results_count
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'status': 'error', 'message': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(error)}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Flask IDS Web Application")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
