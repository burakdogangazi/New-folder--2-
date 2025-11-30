# IDS Web System - Deployment Guide

## 🚀 Deployment Rehberi

### 1. Development Mode

**Basit başlangıç (önerilen öğrenme/test için):**

```powershell
# Virtual environment oluştur
python -m venv venv
venv\Scripts\activate

# Paketleri yükle
pip install -r requirements.txt

# Uygulamayı başlat
python app.py
```

Erişim: `http://localhost:5000`

---

### 2. Production Mode

#### A. Gunicorn ile (Linux/Mac)

```bash
# Gunicorn yükle
pip install gunicorn

# Uygulamayı başlat (4 worker)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### B. Production Ayarları (app.py)

```python
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,              # Production: False
        threaded=True,
        use_reloader=False        # Production: False
    )
```

#### C. Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

### 3. Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories
RUN mkdir -p uploads results data/models

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  ids-web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
      - ./data/models:/app/data/models
    environment:
      - FLASK_ENV=production
    restart: always
```

**Docker ile çalıştır:**

```bash
docker-compose up -d
```

---

### 4. Sistemd Service (Linux)

**/etc/systemd/system/ids-web.service:**

```ini
[Unit]
Description=IDS Web System
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/ids-web
Environment="PATH=/opt/ids-web/venv/bin"
ExecStart=/opt/ids-web/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Başlat ve etkinleştir:**

```bash
sudo systemctl daemon-reload
sudo systemctl start ids-web
sudo systemctl enable ids-web
sudo systemctl status ids-web
```

---

### 5. SSL/TLS Configuration

#### Let's Encrypt ile HTTPS

```bash
# Certbot yükle
sudo apt-get install certbot python3-certbot-nginx

# Sertifika al
sudo certbot certonly --standalone -d your-domain.com

# Nginx konfigürasyonunu güncelle
sudo nano /etc/nginx/sites-available/ids-web
```

**Nginx SSL Config:**

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

---

### 6. Performance Tuning

#### A. Gunicorn Worker Configuration

```bash
# CPU-bound tasks için
workers = (2 * cpu_count) + 1

# Örnek: 4 CPU'lu server
gunicorn -w 9 -b 0.0.0.0:5000 app:app
```

#### B. Flask Configuration

```python
app.config['JSON_SORT_KEYS'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 yıl
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

#### C. File System Cleanup

```bash
# Cron job: Eski dosyaları sil (7 günden eski)
0 2 * * * find /opt/ids-web/uploads -type f -mtime +7 -delete
0 2 * * * find /opt/ids-web/results -type d -mtime +30 -delete
```

---

### 7. Monitoring & Logging

#### A. Application Logging

```python
# app.py içinde
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ids-web.log'),
        logging.StreamHandler()
    ]
)
```

#### B. Systemd Logging

```bash
# Logs kontrol et
sudo journalctl -u ids-web -n 100

# Real-time monitoring
sudo journalctl -u ids-web -f
```

#### C. Prometheus Monitoring (İsteğe bağlı)

```bash
pip install prometheus-flask-exporter

# app.py içinde
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

---

### 8. Backup Strategy

#### A. Otomatik Backup (Cron)

```bash
# results/ ve uploads/ klasörlerini yedekle
0 3 * * * tar -czf /backup/ids-web-$(date +%Y%m%d).tar.gz /opt/ids-web/results/
```

#### B. Database Backup (varsa)

```bash
# Model dosyalarını güvenli yerde tut
cp -r data/models /backup/models-$(date +%Y%m%d)/
```

---

### 9. Security Best Practices

1. **Environment Variables**

```bash
# .env dosyası
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
DEBUG=False
```

2. **File Permissions**

```bash
# Uploads ve results klasörleri
chmod 750 uploads results
chown www-data:www-data uploads results
```

3. **Firewall Rules**

```bash
# UFW ile port 5000 kısıtla
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 5000/tcp  # Sadece localhost'dan
```

4. **SQL Injection/XSS Prevention**

```python
# Flask session configuration
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
```

---

### 10. Troubleshooting

#### Problem: 502 Bad Gateway

**Çözüm:**
```bash
# Gunicorn process'leri kontrol et
ps aux | grep gunicorn

# Nginx logs kontrol et
tail -f /var/log/nginx/error.log
```

#### Problem: Timeout on Large Files

**Çözüm:**
```nginx
# Nginx timeout ayarı
proxy_connect_timeout 300s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;
```

#### Problem: Out of Memory

**Çözüm:**
```python
# app.py içinde
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB'ye düşür
# IDSConfig.MAX_SAMPLES = 5000  # Sample sayısını azalt
```

---

### 11. Health Check Endpoint

```python
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_available': check_models()
    }), 200
```

**Monitoring:**

```bash
# Health check script
curl http://localhost:5000/health
```

---

### 12. Load Balancing (Multiple Instances)

**Nginx Configuration:**

```nginx
upstream ids_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://ids_backend;
        proxy_set_header Host $host;
    }
}
```

---

### 13. Scaling Strategy

1. **Vertical Scaling:** Server RAM/CPU arttır
2. **Horizontal Scaling:** Multiple instances + Load balancer
3. **Caching:** Redis ile sonuçları cache'le
4. **Database:** PostgreSQL ile model results sakla

---

### 14. Deployment Checklist

- [ ] Requirements.txt güncellenmiş
- [ ] debug=False yapılandırması
- [ ] HTTPS sertifikaları yüklü
- [ ] Gunicorn/uWSGI yapılandırılmış
- [ ] Nginx reverse proxy kurulmuş
- [ ] Logging ve monitoring aktif
- [ ] Firewall kuralları uygulanmış
- [ ] Backup strategy oluşturulmuş
- [ ] Health check endpoint test edilmiş
- [ ] Performans test edilmiş
- [ ] Security audit yapılmış

---

## 📞 Support

Deployment sorunları için:
1. Logs kontrol et
2. Health endpoint test et
3. Network bağlantısını doğrula
4. Model dosyalarını kontrol et
