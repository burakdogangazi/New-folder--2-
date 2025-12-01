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
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

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

# These are the EXACT 40 features used during model training
# From: data/features/combined_engineered_features.csv
TRAINING_FEATURES = [
    'log_data-ranges_avg',
    'log_data-ranges_std_deviation',
    'log_interval-messages',
    'log_messages_count',
    'network_fragmentation-score',
    'network_fragmented-packets',
    'network_header-length_avg',
    'network_header-length_std_deviation',
    'network_interval-packets',
    'network_ip-flags_avg',
    'network_ip-flags_std_deviation',
    'network_ip-length_avg',
    'network_ip-length_max',
    'network_ip-length_min',
    'network_ip-length_std_deviation',
    'network_ips_all_count',
    'network_ips_src_count',
    'network_mss_avg',
    'network_mss_std_deviation',
    'network_packet-size_min',
    'network_packets_all_count',
    'network_packets_src_count',
    'network_payload-length_std_deviation',
    'network_ports_all_count',
    'network_ports_dst_count',
    'network_tcp-flags-ack_count',
    'network_tcp-flags-fin_count',
    'network_tcp-flags-syn_count',
    'network_tcp-flags_avg',
    'network_tcp-flags_std_deviation',
    'network_time-delta_avg',
    'network_time-delta_max',
    'network_time-delta_min',
    'network_time-delta_std_deviation',
    'network_ttl_avg',
    'network_ttl_std_deviation',
    'network_window-size_avg',
    'network_window-size_max',
    'network_window-size_min',
    'network_window-size_std_deviation',
]


class IDSConfig:
    """IDS System Configuration"""
    
    # Model paths
    BINARY_MODEL_PATH = "models/binary_model.pkl"
    MULTICLASS_MODEL_PATH = "models/multiclass_model.pkl"
    # Optional preprocessing models
    SCALER_PATH = None  # Not required if features already scaled
    PCA_PATH = None     # Not required if using raw features
    
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


def step_1_clean_dataframe(df):
    """
    STEP 1: Clean DataFrame (Notebook 01 logic)
    
    Replicates the cleaning steps from notebook 01:
    1. Remove duplicate rows
    2. Drop columns with >51% missing values
    3. Fill remaining missing values (numeric: median, categorical: mode)
    4. Remove near-constant features
    5. Handle infinity values
    """
    logger.info("[STEP 1] Cleaning DataFrame...")
    initial_rows, initial_cols = df.shape
    
    # Protected label columns
    protected_cols = ["label1", "label2", "label3", "label4", "label_full"]
    
    # 1. Remove duplicates
    before_dups = len(df)
    df = df.drop_duplicates()
    dups_removed = before_dups - len(df)
    logger.info(f"  → Removed {dups_removed} duplicate rows")
    
    # 2. Missing value analysis - drop columns with >51% missing
    missing_ratio = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing_ratio[
        (missing_ratio > 0.51) & (~missing_ratio.index.isin(protected_cols))
    ]
    
    if len(high_missing) > 0:
        df = df.drop(columns=high_missing.index.tolist())
        logger.info(f"  → Dropped {len(high_missing)} columns with >51% missing values")
    
    # 3. Fill remaining missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Fill numeric with median
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical with mode
    for col in cat_cols:
        if df[col].isnull().any():
            if len(df[col].mode()) > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 4. Remove near-constant features
    nunique_ratio = df.nunique() / len(df)
    low_var_cols = nunique_ratio[
        (nunique_ratio < 0.0001) & (~nunique_ratio.index.isin(protected_cols))
    ].index.tolist()
    
    if len(low_var_cols) > 0:
        df = df.drop(columns=low_var_cols)
        logger.info(f"  → Dropped {len(low_var_cols)} near-constant features")
    
    # 5. Handle infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    final_rows, final_cols = df.shape
    logger.info(f"  → Cleaned: {final_rows:,} rows × {final_cols} cols "
                f"(removed {initial_cols - final_cols} columns)")
    
    return df


def step_2_select_and_scale(df):
    """
    STEP 2: Select Training Features + Scale
    
    The models were trained on EXACTLY 40 specific features from 
    'data/features/combined_engineered_features.csv'
    
    This function:
    1. Selects ONLY those 40 features (if they exist)
    2. Fills any missing with 0 (shouldn't happen if data is complete)
    3. Applies StandardScaler normalization
    4. Separates labels and returns features
    """
    from sklearn.preprocessing import StandardScaler
    
    logger.info("[STEP 2] Selecting Training Features + Scaling...")
    
    # Protected label columns
    protected_cols = ["label1", "label2", "label3", "label4", "label_full"]
    
    # Separate labels
    label_cols = [col for col in protected_cols if col in df.columns]
    labels_df = df[label_cols].copy() if label_cols else None
    
    logger.info(f"  → Separated {len(label_cols)} label columns")
    
    # Select ONLY the training features
    available_features = [col for col in TRAINING_FEATURES if col in df.columns]
    missing_features = [col for col in TRAINING_FEATURES if col not in df.columns]
    
    if len(missing_features) > 0:
        logger.warning(f"  ⚠️  Missing {len(missing_features)} training features:")
        for col in missing_features[:5]:
            logger.warning(f"     - {col}")
        if len(missing_features) > 5:
            logger.warning(f"     ... and {len(missing_features)-5} more")
    
    # Select features
    features_df = df[available_features].copy()
    
    # Fill any missing values with 0
    if features_df.isnull().any().any():
        missing_count = features_df.isnull().sum().sum()
        logger.warning(f"  ⚠️  Found {missing_count} missing values - filling with 0")
        features_df = features_df.fillna(0)
    
    logger.info(f"  → Selected {features_df.shape[1]} training features")
    
    # Apply StandardScaler normalization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
    logger.info(f"  → Applied StandardScaler: mean≈{features_df.values.mean():.6f}, std≈{features_df.values.std():.6f}")
    
    # Return features only (labels handled separately)
    logger.info(f"  → Final features: {features_df.shape[1]} columns")
    
    return features_df, labels_df


def preprocess_user_features(df):
    """
    FULL PIPELINE: Clean + Select Training Features + Scale
    
    This matches the EXACT pipeline used during model training:
    1. STEP 1: Data Cleaning (duplicate removal, missing value handling)
    2. STEP 2: Select the exact 40 training features + StandardScaler
    
    Returns: features array (40 features), labels DataFrame, or error message
    """
    try:
        logger.info("=" * 70)
        logger.info("STARTING FULL PREPROCESSING PIPELINE")
        logger.info("=" * 70)
        
        # STEP 1: Clean
        df_cleaned = step_1_clean_dataframe(df)
        
        # STEP 2: Select training features + scale
        features_df, labels_df = step_2_select_and_scale(df_cleaned)
        
        logger.info("=" * 70)
        total_cols = features_df.shape[1] + (labels_df.shape[1] if labels_df is not None else 0)
        logger.info(f"PIPELINE COMPLETE: {features_df.shape[0]} rows × {total_cols} columns")
        logger.info("=" * 70)
        
        logger.info(f"Features shape: {features_df.shape}")
        if labels_df is not None:
            logger.info(f"Labels shape: {labels_df.shape}")
        
        # Verify we have the correct number of features
        expected_features = len(TRAINING_FEATURES)
        if features_df.shape[1] != expected_features:
            logger.warning(f"⚠️  Expected {expected_features} features but got {features_df.shape[1]}!")
        else:
            logger.info(f"✓ Feature count matches training data: {features_df.shape[1]} features")
        
        return features_df.values, labels_df, None
        
    except Exception as e:
        error_msg = f"Error in preprocessing pipeline: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return None, None, error_msg


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
        # Load CSV - file_id is just the filename, construct full path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        logger.info(f"Looking for file at: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'status': 'error', 'message': f'File not found: {file_path}'}), 404
        
        df, error = preprocess_csv(file_path)
        if error:
            return jsonify({'status': 'error', 'message': error}), 400
        
        # Load IDS system
        ids, error = load_ids_system()
        if error:
            return jsonify({'status': 'error', 'message': error}), 500
        
        logger.info(f"Processing file {file_id} with {len(df)} samples")
        
        # APPLY FULL PREPROCESSING PIPELINE (matching training phase)
        X, y, error = preprocess_user_features(df)
        if error:
            return jsonify({'status': 'error', 'message': error}), 400
        
        logger.info(f"Features preprocessed: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Process samples
        results = []
        predictions = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"Starting prediction loop for {X.shape[0]} samples...")
        
        for idx, features in enumerate(X):
            sample_id = f"sample_{idx}"
            try:
                decision = ids.detect_and_route(sample_id, features)
                
                if decision is None:
                    logger.warning(f"  Skipping sample {idx}: decision is None")
                    continue
                
                # Safely extract all fields with fallbacks
                try:
                    conf_level_str = decision.confidence_level.value if (decision.confidence_level and hasattr(decision.confidence_level, 'value')) else 'UNKNOWN'
                except:
                    conf_level_str = 'UNKNOWN'
                
                try:
                    priority_str = decision.priority.value if (decision.priority and hasattr(decision.priority, 'value')) else 'UNKNOWN'
                except:
                    priority_str = 'UNKNOWN'
                
                try:
                    conf_score_val = float(decision.confidence_score) if decision.confidence_score is not None else 0.0
                except:
                    conf_score_val = 0.0
                
                result = {
                    'sample_id': str(sample_id) if sample_id else 'unknown',
                    'index': int(idx) if idx is not None else -1,
                    'confidence_level': conf_level_str if conf_level_str else 'UNKNOWN',
                    'classification': str(decision.classification) if decision.classification is not None else 'UNKNOWN',
                    'attack_type': str(decision.attack_type) if decision.attack_type is not None else 'UNKNOWN',
                    'confidence_score': conf_score_val if conf_score_val is not None else 0.0,
                    'priority': priority_str if priority_str else 'UNKNOWN',
                    'actions_count': int(len(decision.actions)) if decision.actions else 0
                }
                
                results.append(result)
                predictions.append(result['confidence_level'])
            except Exception as e:
                import traceback
                logger.error(f"  Error processing sample {idx}: {str(e)}")
                logger.error(f"  Traceback: {traceback.format_exc()}")
                continue
        
        logger.info(f"Successfully processed {len(results)} samples")
        
        # Create results DataFrame
        if len(results) == 0:
            return jsonify({'status': 'error', 'message': 'No predictions generated'}), 500
        
        results_df = pd.DataFrame(results)
        
        # Generate statistics with safe dict access
        try:
            high_conf_df = results_df[results_df['confidence_level'] == 'HIGH']
            attack_dist = high_conf_df['attack_type'].value_counts().to_dict() if len(high_conf_df) > 0 else {}
            
            priority_dist = results_df['priority'].value_counts().to_dict() if len(results_df) > 0 else {}
            
            stats = {
                'total_processed': len(results),
                'high_confidence': len(results_df[results_df['confidence_level'] == 'HIGH']),
                'medium_confidence': len(results_df[results_df['confidence_level'] == 'MEDIUM']),
                'low_confidence': len(results_df[results_df['confidence_level'] == 'LOW']),
                'average_confidence': float(results_df['confidence_score'].mean()) if len(results_df) > 0 else 0.0,
                'max_confidence': float(results_df['confidence_score'].max()) if len(results_df) > 0 else 0.0,
                'min_confidence': float(results_df['confidence_score'].min()) if len(results_df) > 0 else 0.0,
                'attacks_detected': len(results_df[results_df['confidence_level'].isin(['HIGH', 'MEDIUM'])]),
                'attack_distribution': attack_dist,
                'priority_distribution': priority_dist
            }
        except Exception as e:
            logger.error(f"Error generating statistics: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            stats = {
                'total_processed': len(results),
                'error': str(e)
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
