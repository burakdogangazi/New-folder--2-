# Label 2 (Attack Sub-Types) - Sınıf Tanımları

## 📌 Label 2 Nedir?

`Label2` sütunu, **attack trafiğinin hangi türde** olduğunu belirtir. Örneğin:

- Normal trafik (Benign) → Label1 = "Benign", Label2 = ? (tanımsız)
- DDoS Saldırısı → Label1 = "Attack", Label2 = "DDoS"
- Injection Saldırısı → Label1 = "Attack", Label2 = "Injection"
- vb.

---

## 🎯 Multiclass Classification Kapsamı

**04_multiclass_attack_classification.ipynb** modeli:

```python
# Adım 1: Yalnızca attack trafiği filtre et
df_attacks = df[df["label1"].str.lower() == "attack"].copy()

# Adım 2: Attack trafiğini Label2'ye göre sınıflandır
# Örneğin: DDoS, Injection, Reconnaissance, ...
y = df_attacks["label2"].copy()

# Sonuç: N-class classification (N = Label2 içindeki benzersiz sınıf sayısı)
```

---

## 📊 Beklenen Attack Sınıfları

CIC-IIoT dataset'inde tipik Label2 sınıfları:

| Sınıf | Açıklama | Örnek |
|-------|----------|-------|
| **DoS/DDoS** | Denial of Service | TCP Flood, UDP Flood, ICMP Flood |
| **Injection** | Veri Injection | SQL Injection, Command Injection |
| **Reconnaissance** | Bilgi Toplama | Port Scan, Network Scan |
| **Backdoor** | Arka Kapı | Remote Access, Unauthorized Access |
| **Man-in-the-Middle (MITM)** | Ortaya Gizli Yerleşme | ARP Spoofing, DNS Spoofing |
| **Trojan** | Truva Atı | Malware, Botnet |
| **Worm** | Solucan | Self-propagating malware |
| **Spyware** | Casusluk Yazılımı | Info Stealing, Keylogger |
| **Ransomware** | Fidye Yazılımı | File Encryption, Data Locking |

---

## 🔍 Kodda Nasıl Kullanılır?

### Notebook'ta Otomatik Tespit
```python
# Label2 benzersiz değerleri ve dağılımı
print(df_attacks["label2"].value_counts())
print(f"Number of attack classes: {df_attacks['label2'].nunique()}")

# Çıktı örneği:
# Label2
# DDoS          150000
# Injection      75000
# Reconnaissance 50000
# Backdoor       25000
# Total:       300,000 records
```

### Confusion Matrix Etiketleri
```python
class_labels = sorted(y_train.unique())
# Örnek: ['Backdoor', 'DDoS', 'Injection', 'Reconnaissance']

# N×N confusion matrix oluşturulur
# Rows: True Labels
# Columns: Predicted Labels
```

### Classification Report
```python
classification_report(y_test, y_pred, 
                     target_names=class_labels,
                     digits=4)

# Çıktı:
#              precision    recall  f1-score   support
#     Backdoor       0.95      0.92      0.93       500
#         DDoS       0.98      0.99      0.98      3000
#    Injection       0.92      0.94      0.93      1500
# Reconnaissance    0.88      0.85      0.86       800
#      accuracy                           0.95      5800
#     macro avg      0.93      0.92      0.92      5800
#  weighted avg      0.95      0.95      0.95      5800
```

---

## 📈 Metriklerin Anlamı (Multiclass)

### Macro-Averaged (Tavsiye Edilen)
```python
precision_macro = (0.95 + 0.98 + 0.92 + 0.88) / 4 = 0.9325
recall_macro = (0.92 + 0.99 + 0.94 + 0.85) / 4 = 0.925
f1_macro = (0.93 + 0.98 + 0.93 + 0.86) / 4 = 0.925
```
**Kullanım:** Sınıflar arasında dengesiz dağılım varsa uygun

### Weighted-Averaged
```python
f1_weighted = (0.93*500 + 0.98*3000 + 0.93*1500 + 0.86*800) / 5800
           = weighted average (büyük sınıflara daha fazla ağırlık)
```
**Kullanım:** Sınıf dengesizliğini hesaba katmak için

---

## 🎓 Örnek Senaryo

### Dataset Hazırlanıyor
```python
# combined_engineered_features.csv dosyasında:
# - 100,000 Benign record
# - 50,000 DDoS record (Label2="DDoS")
# - 30,000 Injection record (Label2="Injection")
# - 20,000 Reconnaissance record (Label2="Reconnaissance")

# Multiclass model yalnızca attack trafiği görür:
# - 50,000 DDoS
# - 30,000 Injection
# - 20,000 Reconnaissance
# TOTAL: 100,000 attack records
```

### Train/Test Split
```python
train_size = 0.8  # 80,000 records
test_size = 0.2   # 20,000 records

# Stratified split ensures class distribution is preserved
train_distribution = {
    "DDoS": 40,000,
    "Injection": 24,000,
    "Reconnaissance": 16,000
}

test_distribution = {
    "DDoS": 10,000,
    "Injection": 6,000,
    "Reconnaissance": 4,000
}
```

### Model Eğitimi
```python
for key in ["log_reg", "knn", "nb", "dt", "rf", "svm"]:
    RandomizedSearchCV(
        estimator=model,
        n_iter=10,
        cv=StratifiedKFold(n_splits=3),
        scoring="f1_macro"  # Tüm 3 sınıfa eşit ağırlık
    ).fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_test, y_pred)  # 3×3
    }
```

---

## 🚀 Çalıştırma

```python
# 04_multiclass_attack_classification.ipynb çalıştırıldığında:

# 1. Label2 sınıfları otomatik tespit edilir
print(f"Attack classes: {['DDoS', 'Injection', 'Reconnaissance']}")

# 2. 6 model eğitilir (3-fold CV, 10 iterations)
# 3. En iyi model seçilir (highest F1-macro)
# 4. Sonuçlar kaydedilir:
#    - multiclass_classification/results_TIMESTAMP/
#    - Her model için 3×3 confusion matrix
#    - Best model raporu N sınıfın detaylı analizi ile
```

---

## 📋 Çıktı Örneği

### metrics_summary_all_models.csv
```csv
model,cv_f1_macro,test_accuracy,test_precision_macro,test_recall_macro,test_f1_macro,test_f1_weighted
log_reg,0.9213,0.9456,0.9287,0.9145,0.9213,0.9451
knn,0.8956,0.9234,0.9034,0.8876,0.8954,0.9232
nb,0.8123,0.8567,0.8234,0.8012,0.8123,0.8564
dt,0.9034,0.9345,0.9087,0.8987,0.9034,0.9343
rf,0.9456,0.9634,0.9523,0.9398,0.9456,0.9633  ← BEST
svm,0.9145,0.9345,0.9234,0.9056,0.9145,0.9343
```

### best_model_report.txt
```
========================================================================
MULTICLASS ATTACK CLASSIFICATION - BEST MODEL REPORT
========================================================================

Run Timestamp: 20251126_143022
Best Model: RF
Attack Classes: Backdoor, DDoS, Injection, Reconnaissance
Total Classes: 4

========================================================================
MODEL PERFORMANCE METRICS
========================================================================

Accuracy (overall):         0.9634
Precision (macro):          0.9523
Recall (macro):             0.9398
F1-Score (macro):           0.9456
F1-Score (weighted):        0.9633

Confusion Matrix Shape: (4, 4)

========================================================================
CLASSIFICATION REPORT
========================================================================

             precision    recall  f1-score   support
     Backdoor       0.96      0.94      0.95       500
         DDoS       0.97      0.98      0.97      3000
    Injection       0.95      0.94      0.94      1500
Reconnaissance      0.92      0.90      0.91       800
      accuracy                           0.96      5800
     macro avg      0.95      0.94      0.95      5800
  weighted avg      0.96      0.96      0.96      5800
```

---

## 💡 İpuçları

1. **Label2 Sınıf Sayısı:** Dinamik ve dataset'e bağlı
2. **Benign Filtrasyonu:** Label1="Benign" kayıtlar tamamen hariç
3. **Macro vs Weighted:** 
   - Makro: Tüm sınıflara eşit önem
   - Weighted: Sınıf desteğine göre önem
4. **Confusion Matrix:** Label2 sınıf sayısı × sınıf sayısı boyutunda
5. **Hyperparameter Tuning:** Multiclass scoring otomatik adjust edilir

---

**✨ Sonuç:** Multiclass model, attack trafiğinin **içindeki** desenleri ve **farklı saldırı tiplerini** ayırt eder.
