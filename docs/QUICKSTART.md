# Flask IDS Web Application - Quick Start Guide

## 🚀 Başlangıç Rehberi

Flask web uygulamanız başarıyla oluşturuldu! İşte adım adım başlama kılavuzu:

## 📦 Kurulum Adımları

### 1. Gerekli Paketleri Yükle

```powershell
pip install -r requirements.txt
```

Temel paketler:
- Flask 3.0.0 (Web framework)
- pandas 2.3.3 (Data processing)
- scikit-learn 1.7.2 (Machine learning)
- joblib 1.5.2 (Model serialization)

### 2. Modelleri Hazırla

Model dosyalarının aşağıda bulunduğundan emin ol:
```
data/models/
├── binary_best_model.pkl         # Binary classifier
├── multiclass_best_model.pkl     # Multiclass classifier
├── scaler.pkl                    # StandardScaler
└── pca.pkl                       # PCA transformer
```

### 3. Uygulamayı Başlat

```powershell
python app.py
```

Çıkti:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
```

## 🌐 Tarayıcıda Açma

Tarayıcınızı açıp git:
```
http://localhost:5000
```

## 📝 Dosya Yapısı

```
Yapay Zeka Proje/
│
├── app.py                              # Ana Flask uygulaması (370 satır)
│   ├── Flask configuration
│   ├── IDSConfig class
│   ├── Utility functions
│   ├── API endpoints
│   └── Error handlers
│
├── ids_confidence_routing_system.py   # IDS logic (kullanımda)
│
├── requirements.txt                    # Gerekli paketler
├── README.md                           # Detaylı dokümantasyon
│
├── templates/                          # HTML şablonları (Bootstrap 5)
│   ├── index.html                      # Dashboard
│   ├── upload.html                     # CSV yükle
│   ├── results.html                    # Sonuçlar & Grafikler
│   └── architecture.html               # Sistem mimarisi
│
├── uploads/                            # Yüklenen CSV dosyaları (geçici)
│
└── results/                            # İşleme sonuçları
    └── results_YYYYMMDD_HHMMSS/
        ├── predictions.csv
        └── statistics.json
```

## 🎨 Web Arayüzü Özellikleri

### 1. Dashboard (/)
- 📊 Sistem özeti
- 🔗 Hızlı erişim linkler
- ⚙️ Sistem durum kontrolü
- 📚 Güven seviyeleri bilgisi

### 2. CSV Yükle (/upload)
- 📁 Drag-and-drop dosya yükleme
- 📈 İlerleme çubuğu
- 📋 Veri istatistikleri
- ✅ Ön işleme validasyonu

### 3. Sonuçlar (/results)
- 📊 4 istatistik kartı (HIGH/MEDIUM/LOW/Toplam)
- 📉 Güven dağılımı pasta grafiği
- 📊 Öncelik dağılımı çubuk grafiği
- 🎯 Saldırı tipi dağılımı
- 📋 İlk 100 tahmin tablosu
- 💾 CSV indirme

### 4. Mimari (/architecture)
- 🏗️ Sistem pipeline açıklaması
- 📊 Güven-tabanlı yönlendirme detayları
- 🤖 ML modelleri bilgisi
- 🔌 Entegrasyon noktaları
- 💻 Teknoloji yığını

## 🔧 API Endpoints

### Upload & Processing
```
POST   /api/upload              # CSV yükle
POST   /api/process/<file_id>   # İşleme başla
```

### Results
```
GET    /api/results/<result_id>     # Sonuçları getir
GET    /api/download/<result_id>    # CSV indir
GET    /api/system-status           # Sistem durumu
```

### Pages
```
GET    /                           # Dashboard
GET    /upload                     # Upload sayfası
GET    /results                    # Sonuçlar sayfası
GET    /architecture               # Mimari sayfası
```

## 📊 Veri Akışı

```
1. CSV Dosya Yükle
       ↓
2. Dosya Doğrulama & Ön İşleme
       ↓
3. Stage 1: Binary Classification
   (Attack vs Benign)
       ↓
4. Stage 2: Multiclass Classification
   (Attack türü)
       ↓
5. Güven Hesaplaması
       ↓
6. Routing Kararı
   HIGH (>85%) → Engelleme + Alert
   MEDIUM (60-85%) → Rate Limit + Queue
   LOW (<60%) → Logging
       ↓
7. Sonuçları Kaydet & Görüntüle
```

## 🎯 Test CSV Örneği

`data/benign/benign_samples_1sec.csv` dosyasını kullanarak test et:
1. Dashboard → "Upload CSV"
2. CSV dosyasını seç
3. "Start Processing" tıkla
4. Results sayfasında grafikler ve tablolar göreceksin

## ⚙️ Yapılandırma (app.py)

```python
class IDSConfig:
    # Model yolları
    BINARY_MODEL_PATH = "data/models/binary_best_model.pkl"
    MULTICLASS_MODEL_PATH = "data/models/multiclass_best_model.pkl"
    
    # Güven eşikleri
    HIGH_CONFIDENCE_THRESHOLD = 0.85      # > 85%
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60    # 60-85%
    
    # İşleme parametreleri
    MAX_SAMPLES = 10000                   # Maksimum örnek
    BATCH_SIZE = 100
```

## 🐛 Sorun Giderme

### Problem: Models not found
**Çözüm:** `data/models/` klasöründe `.pkl` dosyaları kontrol et

### Problem: Upload starts but doesn't complete
**Çözüm:** CSV dosyasında `label1` ve `label2` sütunları var mı kontrol et

### Problem: Processing is slow
**Çözüm:** `IDSConfig.MAX_SAMPLES` değerini azalt

### Problem: Port 5000 already in use
**Çözüm:** `app.py` içinde portu değiştir:
```python
app.run(port=5001)
```

## 📱 Responsive Design

- ✅ Desktop (1200px+)
- ✅ Tablet (768px - 1199px)
- ✅ Mobile (< 768px)

Bootstrap 5 ile tam responsive

## 🌟 Özellikler Özeti

| Özellik | Durum | Notlar |
|---------|-------|--------|
| CSV Upload | ✅ Aktif | Drag-drop & select |
| Data Preprocessing | ✅ Aktif | Otomatik validation |
| Binary Classification | ✅ Aktif | Attack/Benign |
| Multiclass Classification | ✅ Aktif | Attack type |
| Confidence Routing | ✅ Aktif | 3 seviye |
| Charts & Graphs | ✅ Aktif | Chart.js |
| Results Export | ✅ Aktif | CSV format |
| Responsive UI | ✅ Aktif | Bootstrap 5 |
| Modern Design | ✅ Aktif | Gradients & shadows |

## 💡 İpuçları

1. **Büyük Dosyalar:** 10,000+ örnek için `MAX_SAMPLES` azalt
2. **Production:** `debug=False` yap ve Gunicorn kullan
3. **Models:** Modellerini güncel tutp periyodik olarak retrain et
4. **Logging:** `logging` seviyesini ayarla (INFO, DEBUG, etc)

## 📚 Daha Fazla Bilgi

- README.md için detaylı dokümantasyon
- Architecture sayfasını web arayüzünde ziyaret et
- app.py içindeki docstrings'i oku

## ✅ Kontrol Listesi

- [ ] Requirements.txt'ten paketler yüklendi
- [ ] Model dosyaları data/models/de var
- [ ] uploads/ ve results/ klasörleri oluşturuldu
- [ ] app.py Flask uygulaması başladı
- [ ] http://localhost:5000 tarayıcıda açıldı
- [ ] CSV dosya yüklemesi başarılı
- [ ] İşleme tamamlandı ve sonuçlar gösterildi

## 🎉 Tebrikler!

Dual Model IDS Web Sistemi başarıyla kuruldu ve hazır!
Network trafiği verinizi yükleyip saldırı tespitini başlat!
