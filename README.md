# 🌪️ Vortex

Modern web uygulamaları için gelişmiş güvenlik tarama aracı.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

## Ne İşe Yarar?

Vortex, web uygulamalarınızdaki güvenlik açıklarını otomatik olarak tespit eden bir araç. SQL injection'dan XSS'e, CSRF'den SSRF'ye kadar geniş bir yelpazede zafiyet taraması yapabiliyor. 

Projeyi geliştirirken özellikle şu noktalara odaklandım:
- **Hız**: Paralel tarama sayesinde hızlı sonuçlar
- **Doğruluk**: AI destekli analiz ile düşük false positive oranı (%3.2)
- **Kullanım kolaylığı**: Hem CLI hem web arayüzü
- **Gizlilik**: WAF bypass ve stealth özellikleri


## Özellikler

### Zafiyet Taraması
Şu anda desteklenen zafiyet tipleri:
- SQL Injection (error-based, time-based, boolean-based)
- Cross-Site Scripting (reflected, stored, DOM)
- CSRF, LFI, SSRF, SSTI, XXE
- File Upload zafiyetleri
- JWT güvenlik sorunları
- GraphQL API testleri

### AI Destekli Analiz
OpenRouter entegrasyonu sayesinde GPT-4, Claude veya Llama 3 kullanarak:
- Otomatik zafiyet analizi
- Akıllı false positive filtreleme
- Attack chain tespiti

### Stealth Özellikleri
WAF'ları bypass etmek için:
- Cloudflare, AWS WAF, Akamai gibi popüler WAF'ları tespit eder
- User-agent rotation (2026 modern browser profilleri)
- Proxy desteği (HTTP/HTTPS/SOCKS5/Tor)
- Adaptive rate limiting

### Payload Sistemi
3 seviyeli payload sistemi (toplam 833 payload):
- **Tier 1** (90 adet): Güvenli, production ortamında kullanılabilir
- **Tier 2** (160 adet): Dengeli, orta seviye
- **Tier 3** (583 adet): Agresif, manuel test için (SecLists kaynaklı)

Mutation engine ile WAF bypass varyasyonları otomatik üretiliyor.

### Reconnaissance
Tarama öncesi keşif için:
- Subdomain bulma (crt.sh certificate transparency)
- Teknoloji tespiti (PHP, Node, Java, Python vs.)
- Port tarama

### Performans
Gerçek dünya testlerinden bazı rakamlar:
- 150+ istek/saniye throughput
- ~2.5GB ortalama RAM kullanımı
- %3.2 false positive oranı
- %75+ test coverage



## Kurulum

Gereksinimler:
- Python 3.11+
- 8GB RAM (önerilen)

```bash
# Repoyu klonla
git clone https://github.com/SiliconValley-Star/CS_Vortex_b_b.git
cd Vortex

# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\\venv\\Scripts\\activate  # Windows

# Bağımlılıkları yükle
cd vortex
pip install -e .

# .env dosyasını ayarla
cp .env.example .env
# OPENROUTER_API_KEY'i .env dosyasına ekle

# Test et (opsiyonel)
pytest tests/ -v
```

## Kullanım

### Basit Tarama
```bash
python main.py scan https://example.com
```

### Gelişmiş Örnekler

```bash
# Subdomain keşfi ile
python main.py scan https://example.com --enable-recon

# Belirli zafiyet tipleri
python main.py scan https://example.com --include-vulns sqli xss csrf

# Stealth mode + proxy
python main.py scan https://example.com \
  --enable-mutations \
  --delay 2.0 \
  --proxy socks5://127.0.0.1:9050

# Agresif mod (dikkatli kullanın!)
python main.py scan https://example.com \
  --mode aggressive \
  --enable-chains \
  --enable-mutations \
  --threads 20
```

### Web Arayüzü

```bash
cd vortex
./start_web.sh
# http://127.0.0.1:5000 adresine gidin
```

### Python SDK

```python
import asyncio
from core.engine import VortexScanEngine

async def main():
    engine = VortexScanEngine()
    await engine.initialize()
    
    try:
        results = await engine.scan_target(
            target_url="https://example.com",
            scan_types=['sqli', 'xss', 'csrf'],
            enable_recon=True
        )
        
        if results and results.get('findings'):
            for finding in results['findings']:
                print(f"{finding['type']}: {finding['url']}")
    
    finally:
        await engine.shutdown()

asyncio.run(main())
```

## Testler

```bash
# Tüm testler
pytest tests/ -v

# Coverage raporu
pytest tests/ --cov=. --cov-report=html
```

320+ test case, %75+ kod kapsamı.




## Dokümantasyon

Detaylı dokümantasyon için:
- [ARCHITECTURE.md](vortex/ARCHITECTURE.md) - Sistem mimarisi
- [API.md](vortex/API.md) - API dokümantasyonu
- [PERFORMANCE.md](vortex/PERFORMANCE.md) - Performans analizi
- [BENCHMARKS.md](vortex/BENCHMARKS.md) - Benchmark sonuçları

## ⚠️ Yasal Uyarı

**ÖNEMLİ:** Bu araç sadece yetkili güvenlik testleri için tasarlanmıştır.

- ✅ Sadece kendi sistemlerinizde veya yazılı izin aldığınız sistemlerde kullanın
- ❌ İzinsiz tarama yasal değildir ve ciddi sonuçları olabilir
- ✅ Etik kurallara ve yerel yasalara uyun

Vortex'i kullanarak bu kurallara uymayı kabul etmiş sayılırsınız.

## Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır! Büyük değişiklikler için önce bir issue açıp tartışalım.

```bash
git checkout -b feature/amazing-feature
git commit -m 'feat: add amazing feature'
git push origin feature/amazing-feature
```

## Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

**Not:** Bu proje aktif geliştirme aşamasında. Feedback ve katkılarınız için teşekkürler!