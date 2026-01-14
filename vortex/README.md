# Vortex Bug Bounty Automation

Modern bug bounty programları için güvenlik açığı tarama ve otomasyon aracı.

## Özellikler

- SQL Injection, XSS, LFI, SSRF taraması
- AI destekli güvenlik açığı analizi (OpenRouter)
- Web arayüzü ile kolay yönetim
- Otomatik false positive filtreleme
- Evidence chain ve raporlama sistemi

## Kurulum

```bash
git clone https://github.com/SiliconValley-Star/CS_Vortex_b_b.git
cd vortex

python -m venv venv
source venv/bin/activate

pip install -e .

cp .env.example .env
# .env dosyasını düzenle, OPENROUTER_API_KEY ekle
```

## Kullanım

### CLI

```bash
# Temel tarama
python main.py scan https://example.com

# Web server başlat
python web_server.py --host 127.0.0.1 --port 8080
```

### Web Interface

`http://localhost:8080` adresinden web arayüzüne erişebilirsiniz.

## Konfigürasyon

`.env` dosyasında temel ayarlar:

```bash
OPENROUTER_API_KEY=your_key_here
AUTHORIZED_DOMAINS=example.com,test.com
MAX_MEMORY_MB=6000
```

## Güvenlik Uyarısı

Bu araç sadece yetkili güvenlik testleri için kullanılabilir. Kullanıcı tüm yasal sorumluluğu kabul eder. Yetkisiz sistemlerde kullanmak yasaktır.

## Lisans

MIT License - Detaylar için LICENSE dosyasına bakın.