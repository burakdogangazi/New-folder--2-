# 🚀 Quick Start Guide - Binary & Multiclass Classification

## 📁 Proje Yapısı

```
Yapay Zeka Proje/
├── data/
│   ├── attack/            # Raw attack samples
│   ├── benign/            # Raw benign samples
│   ├── merged/            # Combined & cleaned
│   │   └── combined_cleaned.csv
│   ├── features/          # Feature engineering output
│   │   └── combined_engineered_features.csv (with PCA)
│   └── models/            # Saved models directory
├── 01_data_merging_and_cleaning.ipynb
├── 02_feature_engineering.ipynb (PCA included)
├── 03_model_training_binary_classification_and_comparison.ipynb
├── 04_multiclass_attack_classification.ipynb (NEW!)
├── binary_classification/
│   └── results_TIMESTAMP/  (Binary model results)
├── multiclass_classification/
│   └── results_TIMESTAMP/  (Multiclass model results - NEW!)
└── DOCUMENTATION.md
```

---

## ⚡ Hızlı Başlangıç (5 Dakika)

### 1️⃣ Data Preparation (İlk Kez)
```python
# 01_data_merging_and_cleaning.ipynb
# ↓
# Çıktı: data/merged/combined_cleaned.csv
```

### 2️⃣ Feature Engineering (İlk Kez)
```python
# 02_feature_engineering.ipynb
# - Correlation filtering
# - Categorical encoding
# - StandardScaler
# - PCA (95% variance)
# ↓
# Çıktı: data/features/combined_engineered_features.csv
```

### 3️⃣ Binary Classification (Attack vs Benign)
```python
# 03_model_training_binary_classification_and_comparison.ipynb
# - 6 model (LogReg, KNN, NB, DT, RF, SVM)
# - RandomizedSearchCV (10 iterations, 3-fold CV)
# - Tüm metrikler ve görseller
# ↓
# Çıktı: binary_classification/results_YYYYMMDD_HHMMSS/
#   ├── 00_EXECUTION_SUMMARY.txt
#   ├── 01_metrics_summary_all_models.csv
#   ├── 02_best_model_report.txt
#   ├── 03_models_comparison_metrics.png
#   ├── 04_roc_curves_comparison.png
#   ├── 05_f1_score_ranking.png
#   ├── 06_metrics_heatmap.png
#   └── {model_name}/
#       ├── {model}_best_model.pkl
#       └── {model}_confusion_matrix.png
```

### 4️⃣ Multiclass Classification (Attack Sub-Types) - YENİ!
```python
# 04_multiclass_attack_classification.ipynb
# - Yalnızca Attack trafiği (Benign filtre)
# - Label2'ye göre sınıflandırma (DDoS, Injection, vb.)
# - 6 model (Binary ile aynı)
# - RandomizedSearchCV (aynı config)
# - Multiclass-optimized metrics
# ↓
# Çıktı: multiclass_classification/results_YYYYMMDD_HHMMSS/
#   ├── 00_EXECUTION_SUMMARY.txt
#   ├── 01_metrics_summary_all_models.csv
#   ├── 02_best_model_report.txt
#   ├── 03_models_comparison_metrics.png
#   ├── 04_f1_score_ranking.png
#   ├── 05_metrics_heatmap.png
#   └── {model_name}/
#       ├── {model}_best_model.pkl
#       └── {model}_confusion_matrix.png (N×N)
```

---

## 📊 Model Detayları

### Binary Classification (03_...)
| Aspekt | Bilgi |
|--------|-------|
| **Hedef** | label1 (Attack vs Benign) |
| **Sınıf Sayısı** | 2 |
| **Veri** | Tüm dataset (Attack + Benign) |
| **Train/Test** | 80/20 stratified split |
| **Scoring** | F1 (binary) |
| **Metrics** | Accuracy, Precision, Recall, Specificity, F1, ROC-AUC |
| **Confusion Matrix** | 2×2 |
| **ROC Curve** | Single curve |

### Multiclass Classification (04_...)
| Aspekt | Bilgi |
|--------|-------|
| **Hedef** | label2 (Attack Sub-Types) |
| **Sınıf Sayısı** | N (dinamik - dataset'e bağlı) |
| **Veri** | Yalnızca Attack trafiği |
| **Train/Test** | 80/20 stratified split |
| **Scoring** | F1 (macro-averaged) |
| **Metrics** | Accuracy, Precision (Macro), Recall (Macro), F1 (Macro+Weighted) |
| **Confusion Matrix** | N×N |
| **ROC Curve** | Multiple or Confusion Matrix |

---

## 🎯 Her Notebook'ta Ne Oluyor?

### 01_data_merging_and_cleaning.ipynb
```
✓ Attack samples (10 dosya) → merge
✓ Benign samples (10 dosya) → merge
✓ Null check & cleaning
✓ Label columns preserve
→ combined_cleaned.csv
```

### 02_feature_engineering.ipynb
```
✓ Load combined_cleaned.csv
✓ Correlation analysis & filtering
✓ Categorical encoding (get_dummies)
✓ StandardScaler normalization
✓ PCA (95% variance retention)
✓ Feature reduction: 100+ → ~40-50 features
→ combined_engineered_features.csv
```

### 03_model_training_binary_classification_and_comparison.ipynb
```
✓ Load combined_engineered_features.csv
✓ Filter: All data (Attack+Benign)
✓ Target: label1 (binary)
✓ Train/test split (80/20)
✓ Train 6 models with RandomizedSearchCV
✓ Compare metrics
✓ Save best model & reports
→ binary_classification/results_TIMESTAMP/
```

### 04_multiclass_attack_classification.ipynb (NEW)
```
✓ Load combined_engineered_features.csv
✓ Filter: Attack trafiği only (Benign excluded)
✓ Target: label2 (multiclass)
✓ Train/test split (80/20)
✓ Train 6 models with RandomizedSearchCV
✓ Compare multiclass metrics (macro/weighted)
✓ Save best model & reports
→ multiclass_classification/results_TIMESTAMP/
```

---

## 🔄 Mimarik Tutarlılık

### İdentik Yapılar
```python
# Her ikisinde de:
✅ 6 aynı model
✅ RandomizedSearchCV (cv_folds=3, n_iter=10)
✅ StratifiedKFold validation
✅ Train/test split (80/20)
✅ Aynı reporting formatı
✅ Aynı visualizations
```

### Farklılıklar
```python
# Binary:
- Target: label1 → 2 sınıf
- Scoring: f1 (binary)
- Veri: Tüm dataset

# Multiclass:
- Target: label2 → N sınıf
- Scoring: f1_macro (multiclass)
- Veri: Yalnızca attack
```

---

## 📈 Beklenen Sonuçlar

### Binary Model Çalıştırıldığında
```
✓ Training Time: ~10-15 dakika (6 model × 3 CV × 10 iter)
✓ Best Model: (örn. Random Forest)
✓ Test Accuracy: ~95%
✓ Files: ~20 MB (models + visualizations)
```

### Multiclass Model Çalıştırıldığında
```
✓ Training Time: ~10-15 dakika (aynı RandomizedSearchCV)
✓ Best Model: (örn. Random Forest)
✓ Test Accuracy: ~93-96% (sınıf sayısına bağlı)
✓ Files: ~20 MB (models + visualizations)
```

---

## 💾 Çıktı Dosyaları

### Her Modelde (Binary & Multiclass)
```
results_TIMESTAMP/
├── 00_EXECUTION_SUMMARY.txt
│   └── Çalıştırma özeti, best model, file listesi
├── 01_metrics_summary_all_models.csv
│   └── Tüm 6 model için tüm metrikler (spreadsheet)
├── 02_best_model_report.txt
│   └── Best model detaylı rapor + classification report
├── 03_models_comparison_metrics.png
│   └── 6 subplot bar chart (accuracy, precision, recall, f1, etc.)
├── 04_f1_score_ranking.png
│   └── Horizontal bar chart F1 sıralaması
├── 05_metrics_heatmap.png
│   └── Heatmap tüm modeller × tüm metrikler
├── log_reg/
│   ├── log_reg_best_model.pkl (pickle)
│   └── log_reg_confusion_matrix.png
├── knn/
│   ├── knn_best_model.pkl
│   └── knn_confusion_matrix.png
├── nb/
├── dt/
├── rf/
└── svm/
    └── ...
```

---

## 🎯 Karşılaştırma: Binary vs Multiclass

### Binary Görevi
```
Input: Tüm trafik (Attack + Benign)
Question: "Bu trafik saldırı mı, yoksa normal mi?"
Output: Attack veya Benign
Benefit: İç ve dış tehditleri ayırt etme
```

### Multiclass Görevi
```
Input: Yalnızca saldırı trafiği
Question: "Bu saldırı hangi türde? (DDoS, Injection, vb.)"
Output: Attack türü
Benefit: Saldırı türüne göre müdahale (IDS alarm setleri)
```

---

## 🚨 Yaygın Hatalar

### ❌ Hata 1: Feature engineering atlayıp çalıştırma
```python
# Yanlış:
03_model_training_binary_classification_and_comparison.ipynb çalıştır

# Doğru:
01_data_merging_and_cleaning.ipynb → 
02_feature_engineering.ipynb → 
03_model_training_binary_classification_and_comparison.ipynb
```

### ❌ Hata 2: Raw data ile başlama
```python
# Yanlış:
combined_cleaned.csv olmadan

# Doğru:
01_data_merging_and_cleaning.ipynb ile başla
```

### ❌ Hata 3: Features yeniden hesaplamadan model değiştirme
```python
# Yanlış:
Eğer 02_feature_engineering.ipynb değiştirildi ise, 
combined_engineered_features.csv yeniden generate etmeden devam etme

# Doğru:
02_feature_engineering.ipynb çalıştır → 
sonra 03 ve 04 çalıştır
```

---

## 📋 Checklist

### İlk Kez Kurulum
- [ ] `data/attack/` ve `data/benign/` klasörleri var
- [ ] CSV dosyalarında label1, label2 sütunları var
- [ ] `01_data_merging_and_cleaning.ipynb` çalıştırıldı
- [ ] `combined_cleaned.csv` oluştu
- [ ] `02_feature_engineering.ipynb` çalıştırıldı
- [ ] `combined_engineered_features.csv` oluştu

### Binary Model Çalıştırması
- [ ] `combined_engineered_features.csv` mevcut
- [ ] `03_model_training_binary_classification_and_comparison.ipynb` çalıştırıldı
- [ ] `binary_classification/results_TIMESTAMP/` klasörü oluştu
- [ ] Best model raporu okundu

### Multiclass Model Çalıştırması
- [ ] `combined_engineered_features.csv` mevcut
- [ ] `04_multiclass_attack_classification.ipynb` çalıştırıldı
- [ ] `multiclass_classification/results_TIMESTAMP/` klasörü oluştu
- [ ] Best model raporu okundu
- [ ] Label2 sınıfları kontrol edildi

---

## 🔗 Kaynak Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `MULTICLASS_ARCHITECTURE_GUIDE.md` | Detaylı mimari karşılaştırması |
| `LABEL2_CLASSIFICATION_GUIDE.md` | Label2 sınıfları ve multiclass detayları |
| `CHANGES_SUMMARY.md` | PCA ve RandomizedSearchCV değişiklikleri |

---

## ✅ Özet

```
┌─────────────────────────────────────────────────────────────┐
│ PIPELINE                                                    │
├─────────────────────────────────────────────────────────────┤
│ 1. Data Merging & Cleaning                                 │
│    ↓                                                        │
│ 2. Feature Engineering (with PCA)                          │
│    ↓                                                        │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 3A. Binary Classification (Attack vs Benign)         │   │
│ │     └→ binary_classification/results_TIMESTAMP/      │   │
│ │                                                       │   │
│ │ 3B. Multiclass Classification (Attack Sub-Types)  ← │   │
│ │     └→ multiclass_classification/results_TIMESTAMP/  │   │
│ └──────────────────────────────────────────────────────┘   │
│    ↓                                                        │
│ 4. Model Comparison & Analysis                            │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

**🎉 Hazırsınız! Başlayın: `02_feature_engineering.ipynb` çalıştırarak**
