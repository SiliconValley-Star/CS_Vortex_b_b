# 🔒 VORTEX Bug Bounty Otomasyon Sistemi - Kullanım Kılavuzu

## 📋 İçindekiler
1. [Hızlı Başlangıç](#hızlı-başlangıç)
2. [Temel Kullanım](#temel-kullanım)
3. [Web Arayüzü](#web-arayüzü)
4. [Manuel Review](#manuel-review)
5. [Raporlama](#raporlama)
6. [Gelişmiş Özellikler](#gelişmiş-özellikler)

---

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.13+
- OpenRouter API Key (`.env` dosyasında tanımlı)
- Authorized domain (sensoyelektrik.com.tr)

### İlk Kurulum
```bash
cd vortex
pip install -r requirements.txt  # veya pip install -e .
```

### Temel Tarama
```bash
# Tek URL taraması
python main.py scan https://sensoyelektrik.com.tr

# Belirli vulnerability türleri
python main.py scan https://sensoyelektrik.com.tr --include-vulns sqli xss

# Çoklu hedef
python main.py scan https://sensoyelektrik.com.tr https://www.sensoyelektrik.com.tr
```

---

## 🎯 Temel Kullanım

### 1. CLI Tarama

#### Basit Tarama
```bash
python main.py scan https://sensoyelektrik.com.tr
```

#### Gelişmiş Tarama
```bash
python main.py scan https://sensoyelektrik.com.tr \
    --mode active \
    --threads 10 \
    --delay 1.0 \
    --output ./my-scan \
    --include-vulns sqli xss lfi ssrf
```

#### Parametreler
- `--mode`: Tarama modu (passive/active/aggressive)
- `--threads`: Eşzamanlı thread sayısı
- `--delay`: İstekler arası gecikme (saniye)
- `--output`: Çıktı dizini
- `--include-vulns`: Taranacak vulnerability türleri
- `--exclude-vulns`: Hariç tutulacak türler
- `--ai-model`: Kullanılacak AI model
- `--quality-threshold`: Minimum kalite eşiği (0.0-1.0)

### 2. Sistem Durumu
```bash
python main.py status
```

Gösterir:
- Sistem sağlığı
- Memory kullanımı
- AI availability
- Active scans

---

## 🌐 Web Arayüzü

### Web Sunucu Başlatma
```bash
python main.py web --host 127.0.0.1 --port 8080
```

Tarayıcıda: `http://127.0.0.1:8080`

### Dashboard Özellikleri
- **Real-time Updates**: WebSocket ile canlı güncelleme
- **Scan Progress**: Tarama ilerlemesi
- **Finding Discovery**: Yeni bulunan vulnerabilities
- **System Health**: Sistem sağlık metrikleri

### API Endpoints

#### Tarama Başlatma
```bash
POST /api/scan
Content-Type: application/json

{
  "targets": ["https://sensoyelektrik.com.tr"],
  "mode": "active",
  "include_vulns": ["sqli", "xss"]
}
```

#### Findings Listeleme
```bash
GET /api/findings?severity=HIGH&status=SUBMIT_READY&limit=50
```

#### Manuel Review Gönderme
```bash
POST /api/findings/<finding_id>/review
Content-Type: application/json

{
  "decision": "approve",  # approve/reject/needs_info
  "comments": "Verified with manual testing",
  "confidence": 1.0
}
```

---

## 👤 Manuel Review

### Review Queue
```bash
# Web arayüzünde: http://127.0.0.1:8080/manual-review
```

### Review Süreci
1. **Queue'da bekleyen findings**: `NEEDS_MANUAL` status
2. **Priority assignment**: Severity ve confidence bazlı
3. **SLA tracking**: 24h (high priority), 72h (standard)
4. **Decision making**:
   - ✅ **Approve**: → `SUBMIT_READY`
   - ❌ **Reject**: → `FALSE_POSITIVE`
   - 🔄 **Needs Info**: → `NEEDS_MANUAL` (requeue)

### Manuel Verification Checklist
```markdown
- [ ] Payload gerçekten çalışıyor mu?
- [ ] Response anomali gösteriyor mu?
- [ ] Impact gerçek mi, teorik mi?
- [ ] PoC tekrarlanabilir mi?
- [ ] False positive indicator var mı?
```

---

## 📊 Raporlama

### Report Oluşturma

#### Markdown Rapor
```bash
python main.py report --format markdown --output report.md --include-poc
```

#### HTML Rapor (PoC dahil)
```bash
python main.py report --format html --output report.html --include-poc
```

#### JSON Export
```bash
python main.py report --format json --output findings.json
```

### Report İçeriği
- Executive Summary
- Severity breakdown
- Detailed findings
- Proof of Concept (PoC)
- Remediation guidance
- OWASP/CWE references

### PoC Format
Her finding için:
- **Markdown**: Detaylı açıklama
- **cURL**: Command-line PoC
- **Python**: Automated script

---

## 🔧 Gelişmiş Özellikler

### 1. AI-Powered Analysis

Her finding otomatik olarak:
1. **Heuristic Detection**: Scanner tarafından tespit
2. **AI Advisory Analysis**: OpenRouter (Hermes-3, Gemini)
3. **System Verification**: PoC replay + pattern matching
4. **Evidence Validation**: Determinism scoring
5. **Authority Determination**: Final status assignment

### 2. Workflow States

```
DETECTED
   ↓
AI_ANALYSIS_PENDING → AI_CONFIRMED/AI_FAILED
   ↓
SYSTEM_VERIFICATION_PENDING → SYSTEM_VERIFIED/VERIFICATION_FAILED
   ↓
NEEDS_MANUAL (human review required)
   ↓
SUBMIT_READY (approved for submission)
```

### 3. Authority Hierarchy

**Güvenilirlik Sırası:**
1. 🔵 **System Verification** (en güvenilir)
2. 👤 **Human Expert**
3. 🤖 **AI Advisory**
4. 📊 **Heuristic**

**Kural**: Higher authority always overrides lower authority.

### 4. Evidence Standards

- **Deterministic** (≥0.8): Kesin kanıt
- **Behavioral** (≥0.6): Davranışsal anomali
- **Pattern** (≥0.4): Pattern match

### 5. False Positive Filtering

Otomatik filtreleme:
- CDN detection
- WAF response patterns
- Generic error pages
- Rate limiting responses

---

## 📈 Performans & Limitler

### Rate Limiting
```env
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST_SIZE=10
```

### Memory Management
```env
MAX_MEMORY_MB=6000
MEMORY_CLEANUP_THRESHOLD=0.85
MEMORY_EMERGENCY_THRESHOLD=0.95
```

### Concurrency
```env
MAX_CONCURRENT_SCANS=2
MAX_CONCURRENT_REQUESTS_PER_DOMAIN=5
```

---

## 🔐 Güvenlik & Legal

### Authorized Domains
`.env` dosyasında:
```env
AUTHORIZED_DOMAINS=sensoyelektrik.com.tr,*.sensoyelektrik.com.tr
```

⚠️ **Uyarı**: Sadece yetkili domainler taranabilir!

### Legal Compliance
- Automatic scope validation
- PII detection & redaction
- Evidence retention policies
- Audit trail

---

## 🐛 Troubleshooting

### Sorun: AI analizi çalışmıyor
```bash
# OpenRouter API key'i kontrol et
cat .env | grep OPENROUTER_API_KEY

# Test et
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENROUTER_API_KEY'))"
```

### Sorun: Web sunucu başlamıyor
```bash
# Gevent kurulu mu?
pip install gevent gevent-websocket

# Port kullanımda mı?
lsof -i :8080
```

### Sorun: Database hatası
```bash
# Database dizini oluştur
mkdir -p output/database

# Permissions kontrol
chmod 755 output/database
```

---

## 📞 Destek & Katkı

- **Dokümantasyon**: `./docs/`
- **Issues**: Hataları bildirin
- **Contributions**: Pull request gönderin

---

## ✅ Başarı Kontrol Listesi

Sistem çalışıyor mu?

- [ ] CLI tarama yapabiliyorum
- [ ] Web arayüzü açılıyor
- [ ] AI analizi çalışıyor
- [ ] Manuel review yapabiliyorum
- [ ] Rapor oluşturabiliyorum
- [ ] PoC generate ediliyor

Hepsi ✅ ise: **Sistem hazır!** 🎉

---

**VORTEX v1.0 - Enterprise Bug Bounty Automation**  
*Generated: 2026-01-13*