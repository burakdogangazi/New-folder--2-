# IDS Sistemi için Makale Metodoloji Rehberi

## 1. Makale Yapısı (Standart Akademik Format)

### 1.1 Abstract (Özet)
- **IDS Problemi:** Ağ trafiğinde anomalileri ve saldırıları tespit etme zorlukları
- **Çözüm:** İki seviyeli sınıflandırma yaklaşımı (binary + multiclass)
- **Katkı:** Dual-model stratejisinin etkinliği ve hızı
- **Sonuç:** Modellerin doğruluk oranları ve pratik uygulanabilirliği

### 1.2 Introduction (Giriş)
- Siber güvenlik tehditleri ve IDS'nin önemi
- Mevcut zorluklar: yüksek false positive, gerçek zamanlı işlem hızı
- Makale hedefi: Dual-classification yaklaşımın avantajları

### 1.3 Related Work (İlgili Çalışmalar)
- Önceki IDS çalışmaları: single binary vs multiclass sınıflandırma
- PCA ve feature engineering uygulamaları
- GridSearchCV vs RandomizedSearchCV karşılaştırmaları

### 1.4 Methodology (Metodoloji)
- **Veri seti:** CIC-IIoT dataset açıklaması
- **Feature Engineering:** PCA ile boyutluluk azaltması
- **Model Mimarisi:** Binary ve multiclass dual approach

### 1.5 Experiments (Deneyler)
- Model eğitim süreci
- Hiperparametre ayarlaması
- Performans metrikleri

### 1.6 Results (Sonuçlar)
- Grafik ve tablo sunumu
- Model karşılaştırması

### 1.7 Discussion (Tartışma)
- Bulguların analizi
- Avantajlar ve sınırlamalar
- Gerçek dünya uygulaması

### 1.8 Operational Deployment (Operasyonel Dağıtım)
- Confidence-based routing logic
- Real-time decision making
- Integration with SIEM systems

### 1.9 Conclusion (Sonuç)
- Özet ve gelecek çalışmalar

---

## 2. Dual-Model Stratejisi (Makale Merkezi)

### 2.1 İki Seviyeli Sınıflandırma Yaklaşımı

```
                    Ağ Trafiği (Raw Data)
                           |
                    Feature Engineering
                    (PCA: 95% variance)
                           |
                    ___________________
                   |                   |
            Level 1: BINARY            |
         (Attack vs Benign)            |
                   |                   |
            ┌──────┴──────┐            |
            |              |           |
         Benign         Attack         |
        (Alert End)      (Proceed) ────┘
                           |
                    Level 2: MULTICLASS
                  (Attack Type Detection)
                           |
                    ┌──────┬──────┬──────┐
                    |      |      |      |
                  DDoS  Injection Scan  ...
                (Alert w/ Type)
```

### 2.2 Neden İki Model?

| Yön | Binary Model | Multiclass Model | Avantaj |
|-----|-------------|-----------------|---------|
| **Amaç** | Attack/Benign ayrımı | Saldırı tipi belirleme | Hızlı deteksiyon + Detaylı analiz |
| **Veri** | Tüm trafik | Sadece attack trafiği | Multiclass overfitting önlemesi |
| **Performans** | Yüksek doğruluk | Yüksek recall | Her seviyede optimize edilmiş |
| **Uyg. Hızı** | Gerçek zamanlı | Tetiklemeli | Etkili işleme |
| **False Positive** | Düşük (benign filtreleme) | Daha düşük (tip belirleme) | İkinci seviye sınıflandırma |

---

## 3. Deneysel Tasarım (Makale İçin)

### 3.1 Veri Seti Açıklaması

```
CIC-IIoT Dataset
├── Benign Traffic: [X benign samples]
├── Attack Traffic: [Y attack samples]
└── Attack Types: [N different types via Label2]
    ├── DDoS
    ├── Injection
    ├── Scanning
    └── ...
```

**Tablo 1: Dataset Özellikleri**
| Metrik | Değer |
|--------|-------|
| Toplam Samples | [X+Y] |
| Benign Samples | X |
| Attack Samples | Y |
| Attack Sınıfları | N |
| Feature Sayısı (Original) | Z |
| Feature Sayısı (PCA) | ~Z×0.4-0.5 |
| Test/Train Split | 80/20 |

### 3.2 Feature Engineering Süreci

```
Step 1: Veri Yükleme
- Raw traffic features
- Label encoding (label1, label2)

Step 2: Ön İşleme
- Eksik veri temizliği
- Anomali tespiti
- Standardizasyon (StandardScaler)

Step 3: Boyutluluk Azaltması (PCA)
- Açıklanan varyans: 95%
- Component sayısı: [M] (M << Z)
- Bilgi kaybı: ~5%

Çıktı: combined_engineered_features.csv
```

**Tablo 2: PCA Etkileri**
| Metrik | Öncesi | Sonrası |
|--------|--------|---------|
| Feature Sayısı | [Z] | [M] |
| Eğitim Süresi | [T1] sec | [T2] sec (~60% daha hızlı) |
| Model Doğruluğu | [A1]% | [A2]% (~98-99% korundu) |

---

## 4. Model Mimarisi Detayları

### 4.1 Binary Classification Model (Level 1)

```
Input: PCA Features (M dimensions)
       |
       v
6 Classifiers:
├── Logistic Regression (Baseline)
├── K-Nearest Neighbors
├── Gaussian Naive Bayes
├── Decision Tree
├── Random Forest (Ensemble)
└── Support Vector Machine
       |
       v
Hyperparameter Tuning: RandomizedSearchCV
- n_iter: 10
- cv_folds: 3
- scoring: f1 (balanced)
       |
       v
Output: Binary Prediction (Attack/Benign)
```

**Tablo 3: Binary Model Parametreleri**
| Model | Parametreler | n_combinations |
|-------|-------------|-----------------|
| Logistic Regression | C, penalty, solver | 16 |
| KNN | n_neighbors, weights, metric | 30 |
| Naive Bayes | var_smoothing | 5 |
| Decision Tree | criterion, depth, features | 288 |
| Random Forest | estimators, depth, features | 432 |
| SVM | C, kernel, gamma | 32 |
| **RandomizedSearchCV (10 iter)** | ~150 test | ~150 |

### 4.2 Multiclass Classification Model (Level 2)

```
Input: PCA Features (M dimensions) + Attack Label (label2)
       (Filtered: label1 == "Attack" only)
       |
       v
6 Classifiers (Same as Binary)
       |
       v
Hyperparameter Tuning: RandomizedSearchCV
- n_iter: 10
- cv_folds: 3
- scoring: f1_macro (multiclass)
       |
       v
Output: Attack Type Prediction (DDoS/Injection/...)
```

**Tablo 4: Multiclass vs Binary Farkları**
| Özellik | Binary | Multiclass |
|---------|--------|-----------|
| Hedef Sınıfları | 2 | N (≥3) |
| Veri Seti | Tümü | Attack-only |
| Metrikler | Accuracy, Precision, Recall, F1, ROC-AUC | Accuracy, Precision (macro), Recall (macro), F1 (macro/weighted) |
| Optimal Model | Genellikle RF/SVM | Değişken |

---

## 5. Performans Metrikleri (Makale Sonuçları)

### 5.1 Binary Model Metrikleri

**Tablo 5: Binary Model Performans Karşılaştırması**
```
Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC
================================================================================
Logistic Regression| [A1]     | [P1]      | [R1]   | [F1]     | [AUC1]
KNN               | [A2]     | [P2]      | [R2]   | [F2]     | [AUC2]
Naive Bayes       | [A3]     | [P3]      | [R3]   | [F3]     | [AUC3]
Decision Tree     | [A4]     | [P4]      | [R4]   | [F4]     | [AUC4]
Random Forest     | [A5]     | [P5]      | [R5]   | [F5]     | [AUC5] ← Best?
SVM               | [A6]     | [P6]      | [R6]   | [F6]     | [AUC6]
```

### 5.2 Multiclass Model Metrikleri

**Tablo 6: Multiclass Model Performans Karşılaştırması**
```
Model              | Accuracy | Precision | Recall  | F1-Macro | F1-Weighted
                   |          | (macro)   | (macro) |          |
================================================================================
Logistic Regression| [M1]     | [MP1]     | [MR1]   | [MF1]    | [MW1]
KNN               | [M2]     | [MP2]     | [MR2]   | [MF2]    | [MW2]
Naive Bayes       | [M3]     | [MP3]     | [MR3]   | [MF3]    | [MW3]
Decision Tree     | [M4]     | [MP4]     | [MR4]   | [MF4]    | [MW4]
Random Forest     | [M5]     | [MP5]     | [MR5]   | [MF5]    | [MW5]
SVM               | [M6]     | [MP6]     | [MR6]   | [MF6]    | [MW6]
```

### 5.3 Görselleştirmeler (Makalada Kullanılacak Figürler)

**Figure 1: Model Performans Karşılaştırması (Binary)**
- Bar chart: Accuracy, Precision, Recall, F1, ROC-AUC
- Best model vurgulu

**Figure 2: Model Performans Karşılaştırması (Multiclass)**
- Bar chart: Accuracy, Precision (macro), Recall (macro), F1 (macro)
- Best model vurgulu

**Figure 3: Confusion Matrix (Best Binary Model)**
- Heatmap: Attack/Benign classification accuracy

**Figure 4: Confusion Matrix (Best Multiclass Model)**
- Heatmap: Attack types classification accuracy
- Satırlar/sütunlar: DDoS, Injection, Scan, ...

**Figure 5: ROC Curves (Binary Model - Best)**
- Tüm classifiers'ın ROC eğrileri overlaid
- AUC değerleri gösterilir

**Figure 6: Eğitim Süresi Karşılaştırması**
- GridSearchCV (eski) vs RandomizedSearchCV (yeni)
- PCA etkisi gösterilir

**Figure 7: Feature Importance (Random Forest)**
- Top-10 önemli features (PCA component'leri)

**Figure 8: Sistem Mimarisi Diyagramı**
- İki seviyeli pipeline gösterimi

---

## 6. Makale için Yazılacak Ana Kısımlar

### 6.1 Giriş Bölümü (İçerik)

```
- IDS'nin tarihçesi ve önemi
- Mevcut sorunlar:
  * High false positive rate (yanlış alarm)
  * Real-time processing gereksinimi
  * Class imbalance (benign >> attack)
  
- Dual-model yaklaşımın farkı:
  * Level 1: Hızlı binary filtreleme
  * Level 2: Detaylı attack karakterizasyonu
  
- Makale katkıları:
  * İki seviyeli sınıflandırma metodolojisi
  * PCA ile boyutluluk azaltması (~60% hızlanma)
  * 6 model karşılaştırması
  * Gerçek IDS implementasyonu rehberi
```

### 6.2 Metodoloji Bölümü (Tablo/Figür)

```
3.1 Dataset Description
  - Tablo 1: Dataset özellikleri
  - Tablo 2: Attack distribution
  
3.2 Feature Engineering
  - Algorithm 1: PCA pseudocode
  - Tablo 3: PCA performans etkisi
  - Figure 6: Feature reduction visualization
  
3.3 Binary Classification Model
  - Algorithm 2: Training pipeline
  - Tablo 4: Hiperparameter grid
  - Figure 8: Model architecture
  
3.4 Multiclass Classification Model
  - Algorithm 3: Attack type classification
  - Tablo 5: Multiclass metrics definitions
  
3.5 Experimental Setup
  - Train/Test split: 80/20
  - CV: Stratified K-Fold (k=3)
  - Optimization: RandomizedSearchCV (n_iter=10)
  - Random state: 42 (reproducibility)
```

### 6.3 Sonuçlar Bölümü (Veri + Grafik)

```
4.1 Binary Classification Results
  - Tablo 6: Model karşılaştırması
  - Figure 1: Performance metrics bar chart
  - Figure 3: Best model confusion matrix
  - Figure 5: ROC curves overlay
  
4.2 Multiclass Classification Results
  - Tablo 7: Multiclass model karşılaştırması
  - Figure 2: Multiclass performance metrics
  - Figure 4: Multiclass confusion matrix
  
4.3 Computational Performance
  - Tablo 8: Training time comparison
  - Figure 6: Speed improvement graph
  - PCA impact analysis
```

### 6.4 Tartışma Bölümü (Analiz)

```
5.1 Model Selection
- Neden Best Model seçildi?
- Binary vs Multiclass trade-offs
- Real-world applicability

5.2 Feature Engineering Impact
- PCA varyans analizi
- Bilgi kaybı vs hesaplama hızı dengesi
- Feature importance ranking

5.3 Limitations
- Dataset sınırları (CIC-IIoT specific)
- Class imbalance etkileri
- Generalization soruları

5.4 Practical Deployment
- Real-time processing requirements
- Resource constraints
- Model update strategy
```

---

## 7. Makalada Kullanılacak Kodlar (Appendix)

### Appendix A: Feature Engineering Code
```python
# PCA implementation code
# StandardScaler code
```

### Appendix B: Model Training Code
```python
# RandomizedSearchCV implementation
# Binary model training loop
# Multiclass model training loop
```

### Appendix C: Evaluation Metrics Code
```python
# Metric computation functions
# Visualization code
```

---

## 8. Makale Yazım Sırası (Tavsiye Edilen)

1. **Metodoloji** → En ayrıntılı olmalı (kod, tablo, figür hepsi)
2. **Sonuçlar** → Tablolar ve grafikler (veri + görsel)
3. **Tartışma** → Bulguların analizi (yorum + insight)
4. **Giriş** → Bağlam ve motivasyon (sonra yaz)
5. **İlgili Çalışmalar** → Literatür review
6. **Sonuç** → Özet + gelecek çalışmalar
7. **Abstract** → Tüm makaleden sonra yaz

---

## 9. Makalada Raportlandırılacak Çıktılar

Aşağıdaki dosyalar doğrudan makaleye girebilir:

```
binary_classification/results_TIMESTAMP/
├── 01_metrics_summary_all_models.csv  → Tablo 6'ya
├── 02_best_model_report.txt           → Sonuçlar bölümüne
├── 03_models_comparison_metrics.png   → Figure 1'e
├── 04_roc_curves_comparison.png       → Figure 5'e
└── [model_name]/
    ├── [model]_confusion_matrix.png   → Figure 3'e
    └── [model]_best_model.pkl         → Reproducibility

multiclass_classification/results_TIMESTAMP/
├── 01_metrics_summary_all_models.csv  → Tablo 7'ye
├── 02_best_model_report.txt           → Sonuçlar bölümüne
├── 03_models_comparison_metrics.png   → Figure 2'ye
└── [model_name]/
    ├── [model]_confusion_matrix.png   → Figure 4'e
    └── [model]_best_model.pkl         → Reproducibility
```

---

## 10. Makale Başlığı Önerileri

1. "Dual-Level Machine Learning Framework for Network Intrusion Detection: Binary and Multiclass Attack Classification"

2. "Real-Time IDS with PCA-Optimized Dual-Classification: A Machine Learning Approach for IoT Networks"

3. "Efficient Attack Detection and Classification in CIC-IIoT Dataset Using Two-Stage Supervised Learning"

4. "Hybrid Binary-Multiclass Classification for Network Intrusion Detection: Feature Engineering and Model Comparison"

---

## 11. Kağıt Yazarken Önemli Noktalar

✓ **Reproducibility:** Random state, seed'ler, veri split oranları kaydet  
✓ **Clarity:** Metodoloji her detayda anlaşılır olmalı  
✓ **Tables:** Tüm sayısal sonuçlar tabloda göster  
✓ **Figures:** Grafikler high-resolution (300 dpi)  
✓ **Statistics:** Standart sapma, confidence intervals ekle  
✓ **Baseline:** Önceki çalışmalarla karşılaştır  
✓ **Limitations:** Açıkça seçenekler ve kısıtlamalar belirt  
✓ **Code:** GitHub'a public repo koy (referans için)  

---

**Bu rehber kullanarak tam akademik makale yazabilirsin!**
