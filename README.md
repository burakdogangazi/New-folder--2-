# IDS Web System - Burak Doğan - Dual Model Network Attack Detection

A Flask web application that implements a dual-model machine learning system for network intrusion detection with confidence-based routing and enterprise integration capabilities.

## 🎯 Features

- **Dual-Stage Classification Pipeline**
  - Stage 1: Binary classification (Benign vs. Attack)
  - Stage 2: Multiclass classification (Attack type identification)

- **Confidence-Based Decision Routing**
  - HIGH (>85%): Immediate blocking + SIEM alerts + SOC notification
  - MEDIUM (60-85%): Rate limiting + Analyst queue + Threat enrichment
  - LOW (<60%): Routine logging + Retraining dataset

- **Modern Web Interface**
  - Bootstrap 5 responsive design
  - Real-time upload progress
  - Interactive data visualization (Chart.js)
  - CSV download capability

- **Machine Learning Models**
  - 6 ensemble models per stage (LogReg, KNN, Naive Bayes, DT, RF, SVM)
  - PCA feature reduction (95% variance retention)
  - StandardScaler normalization

## 📋 Requirements

- Python 3.10+
- Flask 3.0+
- scikit-learn 1.7+
- pandas 2.3+
- numpy 2.2+
- joblib 1.5+

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Yapay Zeka Proje"
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place trained models in the `data/models/` directory:
```
data/models/
├── binary_best_model.pkl
├── multiclass_best_model.pkl
├── scaler.pkl
└── pca.pkl
```

## 🎮 Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload a CSV file with network traffic data:
   - Features should be numeric
   - Required columns: `label1`, `label2`
   - Maximum file size: 100MB
   - Maximum samples: 10,000 per upload

4. Process the file and view results with:
   - Confidence distribution charts
   - Priority distribution analysis
   - Attack type breakdown
   - Detailed predictions table
   - CSV export capability

## 📁 Project Structure

```
Yapay Zeka Proje/
├── app.py                          # Flask application
├── ids_confidence_routing_system.py # Core IDS logic
├── requirements.txt                # Dependencies
├── templates/
│   ├── index.html                  # Dashboard
│   ├── upload.html                 # CSV upload page
│   ├── results.html                # Results visualization
│   └── architecture.html           # System documentation
├── uploads/                        # Temporary uploaded files
├── results/                        # Processing results
└── data/
    ├── models/                     # Trained ML models
    └── features/                   # Feature engineering data
```

## 🔧 Configuration

Edit `app.py` `IDSConfig` class to customize:

```python
class IDSConfig:
    # Model paths
    BINARY_MODEL_PATH = "data/models/binary_best_model.pkl"
    MULTICLASS_MODEL_PATH = "data/models/multiclass_best_model.pkl"
    SCALER_PATH = "data/models/scaler.pkl"
    PCA_PATH = "data/models/pca.pkl"
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85      # > 85%
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60    # 60-85%
    
    # Processing limits
    BATCH_SIZE = 100
    MAX_SAMPLES = 10000
    
    # Label columns in CSV
    LABEL_COLUMNS = ['label1', 'label2', 'label3', 'label4', 'label_full']
```

## 📊 API Endpoints

### Upload & Process
- `POST /api/upload` - Upload CSV file
- `POST /api/process/<file_id>` - Process uploaded file

### Results & Status
- `GET /api/results/<result_id>` - Retrieve processing results
- `GET /api/download/<result_id>` - Download CSV results
- `GET /api/system-status` - Check system status and models

### Pages
- `GET /` - Dashboard
- `GET /upload` - Upload page
- `GET /results` - Results page
- `GET /architecture` - Architecture documentation

## 🎨 User Interface

### Dashboard
- System overview
- Quick access to features
- System status checker
- Confidence level information

### Upload Page
- Drag-and-drop file upload
- Upload progress bar
- File statistics display
- Pre-processing validation

### Results Page
- Statistics cards (HIGH/MEDIUM/LOW distribution)
- Confidence distribution donut chart
- Priority distribution bar chart
- Attack type distribution
- Predictions table with sorting
- CSV export button

### Architecture Page
- System pipeline overview
- Confidence-based routing explanation
- ML models documentation
- Integration points
- Technology stack

## 🔐 Security Considerations

1. **File Upload Validation**
   - Only CSV files allowed
   - File size limited to 100MB
   - Filename sanitization

2. **Model Security**
   - Models should be stored securely
   - Access control recommended
   - Consider model signing for production

3. **Data Privacy**
   - Uploaded files stored temporarily
   - Results stored in isolated directories
   - Consider adding cleanup jobs

## 📈 Performance

- **Upload Processing**: ~1-5 seconds for typical CSV
- **Batch Processing**: ~100-500 samples/second (depends on hardware)
- **Memory Usage**: ~500MB-1GB for typical operations
- **Scalability**: Can be deployed with Gunicorn/uWSGI

## 🔄 Workflow

1. User uploads CSV file via web interface
2. File is validated and preprocessed
3. Binary classifier determines Attack/Benign
4. If Attack: Multiclass classifier identifies attack type
5. Confidence scores determine routing decision
6. Results visualized with charts and tables
7. User can download predictions as CSV

## 🚨 Confidence Routing Decisions

### HIGH Confidence (>85%)
- ✓ Immediate blocking at perimeter
- ✓ SIEM alert generation
- ✓ P1 incident ticket creation
- ✓ SOC team notification

### MEDIUM Confidence (60-85%)
- ⚠ Rate limiting applied
- ⚠ Analyst review queue
- ⚠ Sandbox analysis initiated
- ⚠ Threat intelligence enrichment

### LOW Confidence (<60%)
- ℹ Routine logging
- ℹ Statistics sampling
- ℹ Retraining dataset collection
- ℹ No immediate action

## 🛠️ Troubleshooting

### Models not found
```
Check data/models/ directory contains all required .pkl files
```

### Upload fails
```
Ensure CSV has label1 and label2 columns
Check file size < 100MB
```

### Slow processing
```
Reduce MAX_SAMPLES in IDSConfig
Consider hardware upgrade
Use production server (Gunicorn)
```

## 📞 Support

For issues or questions:
1. Check the Architecture page documentation
2. Review system logs in console output
3. Verify model file paths in IDSConfig

## 📄 License

[Add your license here]

## 👥 Authors

Developed for Dual Model Network Intrusion Detection System

## 🔗 References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn ML Library](https://scikit-learn.org/)
- [Bootstrap 5 Framework](https://getbootstrap.com/)
- [Chart.js Visualization](https://www.chartjs.org/)
