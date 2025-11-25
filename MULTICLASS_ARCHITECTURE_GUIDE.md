# Multiclass Attack Classification - Architecture Guide

## 📋 Overview

Yeni multiclass classification modeli, **binary classification mimarisini tam olarak kullanarak** oluşturulmuştur. Tek fark, hedef değişkenin ve problem türünün değişmesidir.

---

## 🔄 Mimari Karşılaştırması

### Binary Classification (03_model_training_binary_classification_and_comparison.ipynb)
```
Target: label1 (Attack vs Benign - 2 sınıf)
Tüm veri dahil (Attack + Benign)
Metrics: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
ROC Curve: Tek bir curve (binary)
```

### Multiclass Classification (04_multiclass_attack_classification.ipynb)
```
Target: label2 (Attack Sub-Types - N sınıf)
Yalnızca Attack trafiği (Benign filtre edilmiş)
Metrics: Accuracy, Precision (Macro), Recall (Macro), F1 (Macro+Weighted)
ROC Curve: N×N Confusion Matrix
```

---

## 🏗️ Tutarlılıklar (Mimariden Korunan)

### 1. **Veri Yapısı ve Path'ler**
```python
# Binary
FEATURE_PATH = os.path.join("data", "features", "combined_engineered_features.csv")
OUTPUT: binary_classification/results_TIMESTAMP/

# Multiclass
FEATURE_PATH = os.path.join("data", "features", "combined_engineered_features.csv")
OUTPUT: multiclass_classification/results_TIMESTAMP/
```
✅ **Tutarlı:** Aynı input dataset, aynı başlatma mantığı

### 2. **Model Tanımları**
```python
model_defs = {
    "log_reg": {...},
    "knn": {...},
    "nb": {...},
    "dt": {...},
    "rf": {...},
    "svm": {...}
}

MODELS_TO_RUN = ["log_reg", "knn", "nb", "dt", "rf", "svm"]
```
✅ **Tutarlı:** 6 model, aynı parameter grids (multiclass için adapt edilmiş)

### 3. **RandomizedSearchCV Konfigürasyonu**
```python
RandomizedSearchCV(
    n_iter=10,        # Aynı
    cv_folds=3,       # Aynı
    scoring="f1_macro" # Multiclass: F1 macro
)
```
✅ **Tutarlı:** Aynı hesaplama maliyeti ve hız

### 4. **Eğitim Döngüsü**
```python
for key in MODELS_TO_RUN:
    result_row, best_model, y_pred, metrics = train_and_evaluate_model(...)
    results_list.append(result_row)
    models_dict[key] = best_model
    y_preds_dict[key] = y_pred
    metrics_dict[key] = metrics
```
✅ **Tutarlı:** Aynı loop yapısı ve ektiler

### 5. **Raporlama ve Kayıt Formatı**
```
results_TIMESTAMP/
├── 00_EXECUTION_SUMMARY.txt
├── 01_metrics_summary_all_models.csv
├── 02_best_model_report.txt
├── 03_models_comparison_metrics.png
├── 04_f1_score_ranking.png
├── 05_metrics_heatmap.png
└── {model_key}/
    ├── {model_key}_best_model.pkl
    └── {model_key}_confusion_matrix.png
```
✅ **Tutarlı:** Aynı klasör yapısı ve dosya formatları

---

## 🔧 Multiclass-Spesifik Uyarlamalar

### 1. **Veri Preprocessing**
```python
# FARK: Yalnızca attack trafiği
df_attacks = df[df["label1"].str.lower() == "attack"].copy()

# Target: Label 2 (Attack sub-types)
y = df_attacks["label2"].copy()
```

### 2. **Metrik Hesaplamaları**
```python
def compute_metrics_multiclass(y_true, y_pred):
    # Macro-averaged: Tüm sınıflara eşit ağırlık
    precision_macro = precision_score(..., average='macro')
    recall_macro = recall_score(..., average='macro')
    f1_macro = f1_score(..., average='macro')
    
    # Weighted: Sınıf desteğine göre ağırlıklı
    f1_weighted = f1_score(..., average='weighted')
```

### 3. **Model Parametreleri**
```python
# Logistic Regression: multinomial
LogisticRegression(multi_class='multinomial')

# SVM: One-vs-Rest
SVC(decision_function_shape='ovr')

# Diğerleri: Doğal multiclass desteği
```

### 4. **Confusion Matrix Visualizasyonu**
```python
# Binary: 2×2 matrix
# Multiclass: N×N matrix (N = sınıf sayısı)

plt.figure(figsize=(10, 8))  # Boyut dinamik
sns.heatmap(cm, annot=True, xticklabels=class_labels, 
            yticklabels=class_labels)
```

---

## 📊 Çıktı Dosyaları

### Aynı Formatlar (Binary ile Aynı)
| Dosya | Açıklama | Binary | Multiclass |
|-------|----------|--------|-----------|
| 00_EXECUTION_SUMMARY.txt | Çalıştırma özeti | ✅ | ✅ |
| 01_metrics_summary_all_models.csv | Tüm model metrikleri | ✅ | ✅ |
| 02_best_model_report.txt | Best model raporu | ✅ | ✅ |
| 03_models_comparison_metrics.png | 6 metrik karşılaştırması | ✅ | ✅ |
| 04_f1_score_ranking.png | F1 sıralaması | ✅ | ✅ |
| 05_metrics_heatmap.png | Heatmap | ✅ | ✅ |
| {model}/confusion_matrix.png | Confusion matrix | ✅ | ✅ |
| {model}/{model}_best_model.pkl | Kaydedilmiş model | ✅ | ✅ |

### İçerik Farkları
- **Binary:** 2×2 confusion matrices, binary metrics (specificity)
- **Multiclass:** N×N confusion matrices, multiclass metrics (macro/weighted)

---

## 🚀 Kullanım

### Adım 1: Feature Engineering
```python
# 02_feature_engineering.ipynb
# Çıktı: data/features/combined_engineered_features.csv (PCA ile boyut azaltılmış)
```

### Adım 2: Binary Classification (Opsiyonel)
```python
# 03_model_training_binary_classification_and_comparison.ipynb
# Attack vs Benign sınıflandırması
# Çıktı: binary_classification/results_TIMESTAMP/
```

### Adım 3: Multiclass Classification
```python
# 04_multiclass_attack_classification.ipynb
# Saldırı tiplerini sınıflandırma (benign hariç)
# Çıktı: multiclass_classification/results_TIMESTAMP/
```

---

## 💡 Mimari Avantajları

1. **Kod Tekrarı Minimize:** Aynı eğitim döngüsü, aynı RandomizedSearchCV
2. **Tutarlı Metrikleme:** Aynı rapor formatı ve yapısı
3. **Ölçeklenebilirlik:** N sınıfa kadar otomatik uyarlanabilir
4. **Karşılaştırılabilirlik:** Binary ve Multiclass sonuçları direkt karşılaştırılabilir
5. **Bakım Kolaylığı:** Değişiklikler bir yerde yapılması yeterli

---

## 📈 Beklenen Çıktılar

### Multiclass Classification Çalıştırıldığında

```
multiclass_classification/results_20251126_143022/
├── 00_EXECUTION_SUMMARY.txt
│   └── 6 model eğitimi özeti, best model: XYZ F1=0.8543
├── 01_metrics_summary_all_models.csv
│   └── Tüm 6 model için: accuracy, precision, recall, f1 (macro/weighted)
├── 02_best_model_report.txt
│   └── Best model detaylı rapor + classification report (N sınıf için)
├── 03_models_comparison_metrics.png
│   └── 6 subplot (accuracy, precision_macro, recall_macro, etc.)
├── 04_f1_score_ranking.png
│   └── Modellerin F1 (macro) sıralaması
├── 05_metrics_heatmap.png
│   └── N×6 heatmap (her model × her metrik)
├── log_reg/
│   ├── log_reg_best_model.pkl
│   └── log_reg_confusion_matrix.png (N×N)
├── knn/
│   └── ...
├── nb/
│   └── ...
├── dt/
│   └── ...
├── rf/
│   └── ...
└── svm/
    └── ...
```

---

## 🔍 Karşılaştırma: Binary vs Multiclass

### Yapısı
```
✅ Veri Loading:        İDENTİK
✅ Train/Test Split:    İDENTİK (stratify)
✅ Model Definitions:   ADAPT EDILMIŞ (multiclass-compatible)
✅ RandomizedSearchCV:  İDENTİK (cv_folds=3, n_iter=10)
✅ Training Loop:       İDENTİK
✅ Reporting:           İDENTİK (formatı)
⚠️ Metrics:             ADAPTED (macro vs binary)
⚠️ Visualizations:      ADAPTED (N×N vs 2×2)
```

### Hesaplama Süresi
- Binary: ~X dakika (6 model × 3 CV × 10 n_iter)
- Multiclass: ~X dakika (aynı RandomizedSearchCV parametreleri)

---

## 📝 Notlar

1. **PCA Uyumluluğu:** Multiclass model, binary model ile aynı PCA-transformed features kullanır
2. **Sınıf Sayısı:** Label2'deki sınıf sayısı dinamiktir, otomatik handle edilir
3. **Benign Filtrasyonu:** Multiclass model sadece attack trafiğini görür
4. **Model Kaydı:** Pickle format, binary ve multiclass aynı
5. **Hyperparameter Tuning:** RandomizedSearchCV, multiclass scoring için otomatik adapt

---

**✨ Sonuç:** Multiclass model, binary classification mimarisinin tam bir uzantısıdır. Aynı yapı, aynı kalite, fakat farklı problem tanımı.
