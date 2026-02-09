# Private Payload Library

Bu klasör kendi özel payload koleksiyonunuz için ayrılmıştır.

## 📁 Dizin Yapısı

```
private/
├── README.md           # Bu dosya
├── examples.json       # Örnek payloadlar
├── my_xss.json        # XSS payloadlarınız
├── my_sqli.txt        # SQL injection payloadları (satır satır)
└── custom.yaml        # Özel payload'lar
```

## 🎯 Desteklenen Formatlar

### 1. JSON Format (Önerilen)

```json
[
  {
    "payload": "<script>alert(1)</script>",
    "category": "xss",
    "description": "Basic XSS test",
    "tags": ["xss", "basic", "alert"],
    "success_rate": 0.75,
    "notes": "Works on most unfiltered inputs",
    "source": "Own research",
    "discovered_date": "2024-01-15",
    "cvss_score": 6.1
  }
]
```

### 2. YAML Format

```yaml
- payload: "' OR '1'='1' --"
  category: sqli
  description: Classic SQL injection
  tags:
    - sqli
    - auth-bypass
  success_rate: 0.72
  notes: Still works on legacy systems
  source: OWASP
  cvss_score: 9.8
```

### 3. TXT Format (Basit)

Sadece payload'ları satır satır yazın:

```
<script>alert(1)</script>
<img src=x onerror=alert(1)>
<svg onload=alert(1)>
```

## 🚀 Kullanım

### Python'dan Kullanım

```python
from core.payloads.private_library import get_private_library

# Library'yi yükle
lib = get_private_library()

# Kategori bazlı payload al
xss_payloads = lib.get_payload_strings(category='xss')

# Tag bazlı filtrele
waf_bypass = lib.get_payload_strings(tags=['waf-bypass'])

# Success rate'e göre filtrele
high_success = lib.get_payload_strings(min_success_rate=0.7)

# Yeni payload ekle
from core.payloads.private_library import PrivatePayload

new_payload = PrivatePayload(
    payload="<script>alert(document.domain)</script>",
    category="xss",
    description="DOM-based XSS",
    tags=["xss", "dom"],
    success_rate=0.68
)

lib.add_payload(new_payload)

# Dışa aktar
lib.export_to_file('my_xss_collection.json', category='xss')

# İçe aktar
lib.import_from_file('downloaded_payloads.json')
```

### CLI'dan Kullanım (PayloadManager ile)

Private payload'lar otomatik olarak yüklenir:

```bash
python main.py scan https://example.com --include-private-payloads
```

## 📊 Metadata Alanları

- **payload**: Actual payload string (ZORUNLU)
- **category**: Kategori (xss, sqli, lfi, custom, etc.) (ZORUNLU)
- **description**: Açıklama
- **tags**: Etiketler (array)
- **success_rate**: Başarı oranı (0.0-1.0)
- **notes**: Notlar
- **source**: Kaynak (URL, araştırmacı adı, etc.)
- **discovered_date**: Keşif tarihi (YYYY-MM-DD)
- **cvss_score**: CVSS skoru (0.0-10.0)

## 💡 İpuçları

1. **Organize Et**: Her zafiyet tipi için ayrı dosya kullan
2. **Tag Kullan**: Bulması kolay olsun (waf-bypass, auth-bypass, etc.)
3. **Success Rate Takibi**: Gerçek testlerde başarı oranlarını güncelle
4. **Kaynak Belirt**: Nereden bulduysan not et
5. **Git'e Commit Etme**: `.gitignore`'da private/ klasörü var

## ⚠️ Güvenlik

- Private payload'lar SADECE local'de saklanır
- Hiçbir zaman dışarıya gönderilmez
- `.gitignore` ile koruma altında
- Kendi sorumluluğunuzda kullanın

## 📝 Örnekler

`examples.json` dosyasını kontrol edin:

```bash
cat vortex/payloads/private/examples.json