# 🌪️ Vortex - Enterprise-Grade Security Scanner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()
[![Test Coverage](https://img.shields.io/badge/Coverage-75%25-brightgreen)]()
[![Security](https://img.shields.io/badge/Security-Enterprise-red)]()

## 🚀 Proje Hakkında

**Vortex**, modern web uygulamalarının güvenlik açıklarını tespit etmek için geliştirilmiş enterprise-grade bir güvenlik tarama sistemidir. Yapay zeka destekli analiz, otomatik doğrulama ve production-ready altyapısı ile profesyonel güvenlik testleri için tasarlanmıştır.

### ✨ Temel Özellikler

#### 🧠 Yapay Zeka Destekli Analiz
- OpenRouter entegrasyonu ile GPT-4, Claude, Llama 3 desteği
- Otomatik zafiyet analizi ve öneri sistemi
- Akıllı yanlış pozitif filtreleme
- Sansürsüz güvenlik analizi modu

#### 🛡️ Kapsamlı Zafiyet Taraması
- **Web Zafiyetleri**: SQL Injection, XSS, CSRF, LFI, SSRF, SSTI, XXE, File Upload
- **API Güvenliği**: JWT, GraphQL
- **Gelişmiş**: DOM XSS (Playwright tabanlı)

#### 🥷 Stealth & Evasion (V19.0 ULTIMATE)
- **WAF Detection**: Cloudflare, AWS WAF, Akamai, Imperva, F5 BIG-IP, ModSecurity
- **User-Agent Rotation**: 2026 modern browser profiles (Chrome 124+, Firefox 125+)
- **Proxy Management**: HTTP/HTTPS/SOCKS4/SOCKS5/Tor proxy chains
- **Rate Limiting**: Adaptive throttling with jitter
- **Request Fingerprinting**: Header randomization, TLS fingerprint spoofing

#### 🔭 Reconnaissance System (V19.0)
- **Subdomain Discovery**: crt.sh certificate transparency logs
- **Technology Detection**: Stack fingerprinting (PHP, Node, Java, Python, etc.)
- **Asset Inventory**: Live probe and technology detection
- **Port Scanning**: Intelligent service discovery

#### 💣 Intelligent Payload System (V21.0 - 3-TIER SYSTEM)
- **3-Tier Curated Payloads**: 833 professional payloads (90 TIER 1 + 160 TIER 2 + 583 TIER 3)
- **Context-Aware Selection**: Technology-specific (PHP, Java, Node, Python)
- **Mutation Engine**: 413 satır WAF bypass logic
- **SecLists Integration**: 583 TIER 3 payloads from industry-standard sources
- **Encoding Variations**: URL, Base64, Hex, Unicode, Polyglot generation
- **Tier Descriptions**:
  - **TIER 1 (Safe)**: Production-safe, minimal impact payloads (90 payloads)
  - **TIER 2 (Balanced)**: Moderate risk, broader coverage (160 payloads)
  - **TIER 3 (Aggressive)**: Manual testing only, comprehensive SecLists (583 payloads)

#### 🔗 Attack Chain Intelligence (V20.0)
- **Multi-Step Attacks**: Automatic chain detection and execution
- **Causation Analysis**: Vulnerability relationship mapping
- **Pattern Recognition**: AI-powered attack path discovery

#### 🔍 Advanced Verification System (V21.0)
- **AI-Powered Triage**: Intelligent finding validation with confidence scoring
- **Deterministic Auto-Accept**: Pattern-based automatic validation (99% accuracy)
- **PoC Replay**: Structural and timing analysis for proof of concept
- **Evidence Quality**: Behavioral analysis and determinism verification
- **False Positive Rate**: <3.2% with multi-layer validation

#### ⚡ Production-Ready Altyapı
- **Performance Profiling**: Gerçek zamanlı performans izleme
- **Memory Management**: Dinamik bellek yönetimi ve leak detection
- **Error Handling**: Gelişmiş hata yönetimi ve retry mekanizmaları
- **Test Coverage**: %75+ kod kapsamı ile 320+ test

#### 📊 Monitoring & Analytics
- Real-time performans metrikleri
- Sistem sağlık izleme
- Otomatik performans optimizasyonu
- Prometheus/Grafana entegrasyonu hazır

#### 🌐 Dual Interface
- **CLI Mode**: Hızlı terminal tabanlı taramalar
- **Web Dashboard**: Modern real-time web arayüzü
- **API Mode**: RESTful API ile programatik erişim
- **WebSocket**: Canlı tarama güncellemeleri

## 📋 Performans Metrikleri

| Metrik | Değer | Durum |
|--------|-------|-------|
| Throughput | 150+ req/sec | ✅ Mükemmel |
| Memory Usage | 2.5GB avg | ✅ Optimal |
| CPU Usage | 45% avg | ✅ Verimli |
| Response Time | 65ms avg | ✅ Hızlı |
| False Positive | 3.2% | ✅ Düşük |
| Test Coverage | 75%+ | ✅ İyi |
| Total Payloads | 833 (3-TIER) | ✅ Kapsamlı |
| Verification Accuracy | 99%+ | ✅ Mükemmel |
## 🏗️ Sistem Mimarisi

### Genel Bakış

```
┌─────────────────────────────────────────────────────────┐
│                    Vortex Scanner                        │
├─────────────────────────────────────────────────────────┤
│  CLI Layer          │          Web Layer                │
│  (Terminal)         │    (Flask + SocketIO)             │
└────────┬────────────┴───────────┬─────────────────────┘
         │                        │
    ┌────▼────────────────────────▼─────┐
    │      Core Engine (Orchestrator)    │
    └────┬───────────┬──────────┬────────┘
         │           │          │
    ┌────▼────┐ ┌───▼────┐ ┌──▼──────┐
    │ Queue   │ │Network │ │Database │
    │ Manager │ │ Client │ │ Manager │
    └────┬────┘ └───┬────┘ └──┬──────┘
         │          │          │
    ┌────▼──────────▼──────────▼────────┐
    │       Scanner Layer (11)           │
    │  SQLi│XSS│CSRF│LFI│SSRF│JWT│...   │
    └───────────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │    Support Systems                 │
    │  AI│Memory│Health│Legal│Evidence  │
    └───────────────────────────────────┘
```

### Detaylı Komponent Diyagramı

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    🎯 USER INTERFACE LAYER                      ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌──────────────────┐         ┌──────────────────────────┐   ┃
┃  │   CLI Interface  │         │    Web Dashboard         │   ┃
┃  │   (main.py)      │         │  (Flask + SocketIO)      │   ┃
┃  │  - scan command  │         │  - Real-time updates     │   ┃
┃  │  - report gen    │         │  - Findings management   │   ┃
┃  └────────┬─────────┘         └──────────┬───────────────┘   ┃
┗━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━┛
            │                              │
            └──────────────┬───────────────┘
                           │
┏━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                🧠 CORE ORCHESTRATION LAYER                      ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌────────────────────────────────────────────────────────┐   ┃
┃  │           VortexScanEngine (Core Engine)               │   ┃
┃  │  - Scan orchestration    - Component coordination      │   ┃
┃  │  - Workflow management   - Resource allocation         │   ┃
┃  └────┬────────────────┬────────────────┬─────────────────┘   ┃
┃       │                │                │                      ┃
┃  ┌────▼──────┐   ┌────▼──────┐   ┌────▼──────────┐          ┃
┃  │  Workflow │   │   Queue   │   │  State        │          ┃
┃  │Orchestrator│   │  Manager  │   │  Manager      │          ┃
┃  └───────────┘   └───────────┘   └───────────────┘          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            │                │                │
            └────────────────┼────────────────┘
                             │
┏━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                  💣 PAYLOAD & MUTATION LAYER                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌────────────────────────────────────────────────────────┐   ┃
┃  │              Payload Manager (V21.0)                   │   ┃
┃  │  ┌─────────┐  ┌─────────┐  ┌──────────┐              │   ┃
┃  │  │ TIER 1  │  │ TIER 2  │  │  TIER 3  │              │   ┃
┃  │  │90 safe  │  │160 bal  │  │583 aggr  │              │   ┃
┃  │  └────┬────┘  └────┬────┘  └────┬─────┘              │   ┃
┃  │       └────────────┼─────────────┘                     │   ┃
┃  │                    │                                    │   ┃
┃  │       ┌────────────▼──────────────┐                   │   ┃
┃  │       │   Mutation Engine         │                   │   ┃
┃  │       │   - WAF bypass (413 LOC)  │                   │   ┃
┃  │       │   - Encoding variations   │                   │   ┃
┃  │       │   - Context-aware         │                   │   ┃
┃  │       └───────────────────────────┘                   │   ┃
┃  └────────────────────────────────────────────────────────┘   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                           │
┏━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    🔍 SCANNER LAYER (11 Scanners)               ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            ┃
┃  │  SQLi   │ │   XSS   │ │  CSRF   │ │   LFI   │            ┃
┃  │ Scanner │ │ Scanner │ │ Scanner │ │ Scanner │            ┃
┃  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            ┃
┃       │           │           │           │                   ┃
┃  ┌────▼────┐ ┌───▼────┐ ┌────▼────┐ ┌───▼─────┐            ┃
┃  │  SSRF   │ │  SSTI  │ │   XXE   │ │  File   │            ┃
┃  │ Scanner │ │Scanner │ │ Scanner │ │ Upload  │            ┃
┃  └────┬────┘ └───┬────┘ └────┬────┘ └───┬─────┘            ┃
┃       │          │           │           │                   ┃
┃  ┌────▼────┐ ┌──▼──────┐ ┌──▼──────┐                       ┃
┃  │   JWT   │ │ GraphQL │ │   DOM   │                       ┃
┃  │ Scanner │ │ Scanner │ │ Scanner │                       ┃
┃  └─────────┘ └─────────┘ └─────────┘                       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                           │
┏━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              🔬 VERIFICATION & VALIDATION LAYER                 ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌────────────────────────────────────────────────────────┐   ┃
┃  │       System Verification Engine (V21.0)               │   ┃
┃  │                                                         │   ┃
┃  │  ┌──────────────────┐      ┌────────────────────┐    │   ┃
┃  │  │  Deterministic   │      │   AI Triage        │    │   ┃
┃  │  │  Auto-Accept     │      │   System           │    │   ┃
┃  │  │  (99% accuracy)  │      │   - GPT-4/Claude   │    │   ┃
┃  │  │  - Pattern match │      │   - Confidence     │    │   ┃
┃  │  └────────┬─────────┘      └─────────┬──────────┘    │   ┃
┃  │           │                           │                │   ┃
┃  │           └───────────┬───────────────┘                │   ┃
┃  │                       │                                 │   ┃
┃  │           ┌───────────▼──────────────┐                │   ┃
┃  │           │   PoC Replay & Analysis  │                │   ┃
┃  │           │   - Structural analysis  │                │   ┃
┃  │           │   - Timing verification  │                │   ┃
┃  │           └──────────────────────────┘                │   ┃
┃  └────────────────────────────────────────────────────────┘   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                           │
┏━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                  🤖 AI INTELLIGENCE LAYER                       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌────────────────────────────────────────────────────────┐   ┃
┃  │            OpenRouter Integration                      │   ┃
┃  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐        │   ┃
┃  │  │  GPT-4   │  │  Claude  │  │  Llama 3     │        │   ┃
┃  │  └────┬─────┘  └────┬─────┘  └──────┬───────┘        │   ┃
┃  │       └─────────────┼────────────────┘                 │   ┃
┃  │                     │                                   │   ┃
┃  │  ┌──────────────────▼──────────────────┐              │   ┃
┃  │  │      AI Advisory & Analysis         │              │   ┃
┃  │  │  - Attack chain detection           │              │   ┃
┃  │  │  - Causation analysis                │              │   ┃
┃  │  │  - Uncensored security analysis     │              │   ┃
┃  │  └─────────────────────────────────────┘              │   ┃
┃  └────────────────────────────────────────────────────────┘   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            │                              │
┏━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━┓
┃                  🥷 STEALTH & RECON LAYER                       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌──────────────────────┐      ┌──────────────────────┐      ┃
┃  │  Stealth & Evasion   │      │   Recon Manager      │      ┃
┃  │  - WAF detection     │      │   - Subdomain enum   │      ┃
┃  │  - UA rotation       │      │   - Tech detection   │      ┃
┃  │  - Proxy chains      │      │   - Asset discovery  │      ┃
┃  │  - Rate limiting     │      │   - Port scanning    │      ┃
┃  └──────────────────────┘      └──────────────────────┘      ┃
┗━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                       │
┏━━━━━━━━━━━━━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                  🛡️ SUPPORT SYSTEMS LAYER                       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌─────────────┐    ┃
┃  │ Network  │ │ Database │ │  Memory   │ │   Health    │    ┃
┃  │  Client  │ │ Manager  │ │  Manager  │ │   Monitor   │    ┃
┃  └──────────┘ └──────────┘ └───────────┘ └─────────────┘    ┃
┃                                                                ┃
┃  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌─────────────┐    ┃
┃  │  Legal   │ │ Evidence │ │  Quality  │ │  Reporting  │    ┃
┃  │Compliance│ │Integrity │ │ Assurance │ │   System    │    ┃
┃  └──────────┘ └──────────┘ └───────────┘ └─────────────┘    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### 🔄 Veri Akışı (Data Flow)

```
1. User Request (CLI/Web)
   ↓
2. Core Engine → Workflow Orchestrator
   ↓
3. Recon Manager → Subdomain Discovery (opsiyonel)
   ↓
4. Payload Manager → TIER Selection (1/2/3)
   ↓
5. Mutation Engine → WAF Bypass Payloads
   ↓
6. Queue Manager → Scanner Distribution
   ↓
7. Scanner Execution → Stealth Layer
   ↓
8. Network Client → Target Application
   ↓
9. Response Analysis → Finding Detection
   ↓
10. Verification Engine:
    ├─→ Deterministic Analyzer (High Confidence)
    │   └─→ Auto-Accept (99% accuracy)
    └─→ AI Triage (Low Confidence)
        └─→ OpenRouter → Confidence Score
   ↓
11. PoC Replay & Evidence Collection
   ↓
12. Database Storage → Legal Compliance Check
   ↓
13. Report Generation → User Interface
```

### ⚙️ Komponent İlişkileri

**Core Engine:**
- Tüm komponentlerin orkestratörü
- Workflow ve state yönetimi
- Resource allocation ve coordination

**Payload System:**
- 3-TIER payload database (833 total)
- Context-aware selection
- Mutation engine integration

**Scanner Layer:**
- 11 specialized vulnerability scanners
- Parallel execution via Queue Manager
- Stealth layer integration

**Verification System:**
- Deterministic auto-accept (99% accuracy)
- AI-powered triage (GPT-4/Claude/Llama)
- PoC replay ve structural analysis

**AI Intelligence:**
- OpenRouter multi-model support
- Attack chain detection
- Causation analysis

**Support Systems:**
- Network client (HTTP/WebSocket)
- Database persistence (SQLite)
- Memory management (8GB optimization)
- Health monitoring (real-time)
- Legal compliance validation
- Evidence integrity (cryptographic)

Detaylı mimari için: [`ARCHITECTURE.md`](vortex/ARCHITECTURE.md)

## 🛠️ Kurulum

### Gereksinimler
- Python 3.11 veya üzeri
- Git
- 8GB RAM (önerilen)
- 4+ CPU cores (önerilen)

### Hızlı Başlangıç

```bash
# 1. Repository'yi klonlayın
git clone https://github.com/yourusername/Vortex.git
cd Vortex

# 2. Virtual environment oluşturun
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 3. Bağımlılıkları yükleyin
cd vortex
pip install -e .

# 4. Yapılandırma dosyasını oluşturun
cp .env.example .env
# .env dosyasına OPENROUTER_API_KEY ekleyin

# 5. Test edin (opsiyonel)
pytest tests/ -v
```

## 💻 Kullanım

### 🎯 CLI Kullanımı

#### Basit Tarama
```bash
python main.py scan https://example.com
```

#### Gelişmiş Tarama
```bash
# Subdomain keşfi ile tarama (Recon System)
python main.py scan https://example.com --enable-recon

# Çoklu zafiyet tipi
python main.py scan https://example.com \
  --include-vulns sqli xss csrf lfi

# Stealth mode + WAF bypass mutations
python main.py scan https://example.com \
  --enable-mutations \
  --delay 2.0 \
  --proxy socks5://127.0.0.1:9050  # Tor

# Aggressive mode + attack chains
python main.py scan https://example.com \
  --mode aggressive \
  --enable-chains \
  --enable-mutations \
  --enable-recon \
  --threads 20

# Tam özellikli tarama
python main.py scan https://example.com \
  --enable-recon \          # Subdomain discovery
  --enable-mutations \      # WAF bypass payloads
  --enable-chains \         # Multi-step attacks
  --enable-dom \            # DOM XSS (Playwright)
  --enable-graphql \        # GraphQL scanning
  --mode aggressive \
  --threads 20

# Proxy ile tarama
python main.py scan https://example.com \
  --proxy http://localhost:8080

# Custom headers
python main.py scan https://example.com \
  --headers "Authorization:Bearer token123"
```

#### Rapor Oluşturma
```bash
# Markdown rapor
python main.py report --format markdown --output report.md

# HTML rapor
python main.py report --format html --output report.html

# JSON rapor
python main.py report --format json --output report.json
```

#### Sistem Durumu
```bash
python main.py status
```

### 🌐 Web Arayüzü

```bash
# Web sunucusunu başlat
cd vortex
./start_web.sh

# Tarayıcıda aç
# http://127.0.0.1:5000
```

**Web Özellikler:**
- Real-time tarama izleme
- Canlı performans grafikleri
- Bulgu yönetimi ve filtreleme
- Otomatik rapor oluşturma
- Sistem sağlık monitörü

### 📚 Python SDK

```python
import asyncio
from core.engine import VortexScanEngine

async def main():
    engine = VortexScanEngine()
    await engine.initialize()
    
    try:
        # Tarama yap
        results = await engine.scan_target(
            target_url="https://example.com",
            scan_types=['sqli', 'xss', 'csrf'],
            enable_recon=True
        )
        
        # Bulguları işle
        if results and results.get('findings'):
            for finding in results['findings']:
                print(f"{finding['type']}: {finding['url']}")
    
    finally:
        await engine.shutdown()

asyncio.run(main())
```

## 🧪 Test Suite

Vortex kapsamlı bir test suite ile gelir:

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Scanner testleri
python run_scanner_tests.py

# Core testleri
python run_core_tests.py

# Integration testleri
python run_integration_tests.py

# Performance testleri
pytest tests/test_performance/ -v -m performance

# Coverage raporu
pytest tests/ --cov=. --cov-report=html
```

**Test İstatistikleri:**
- 21 test dosyası
- 320+ test case
- %75+ kod kapsamı
- Scanner, Core, Performance, Integration testleri

## 📖 Dokümantasyon

### Teknik Dokümantasyon
- **[ARCHITECTURE.md](vortex/ARCHITECTURE.md)** - Sistem mimarisi ve component'ler
- **[API.md](vortex/API.md)** - CLI, Web API ve Python SDK dokümantasyonu
- **[PERFORMANCE.md](vortex/PERFORMANCE.md)** - Performans analizi ve optimizasyon
- **[BENCHMARKS.md](vortex/BENCHMARKS.md)** - Detaylı benchmark sonuçları

### Kullanım Kılavuzları
- **[KULLANIM_KILAVUZU.md](vortex/KULLANIM_KILAVUZU.md)** - Türkçe kullanım kılavuzu
- **[LEGAL_COMPLIANCE.md](vortex/LEGAL_COMPLIANCE.md)** - Yasal uyumluluk

## 🔧 Yapılandırma

### Environment Variables

```bash
# .env dosyası
OPENROUTER_API_KEY=your_api_key_here
DB_PATH=output/database/vortex.db
HTTP_TIMEOUT=30
MAX_CONCURRENT=50
MAX_MEMORY_MB=8192
```

### Scan Modları

- **passive**: Sessiz keşif, minimum etkileşim
- **active**: Standart tarama (varsayılan)
- **aggressive**: Yoğun tarama, maksimum kapsam

## 🎯 Özellikler ve Yetenekler

### Zafiyet Tarama
✅ SQL Injection (Error-based, Time-based, Boolean-based)  
✅ Cross-Site Scripting (Reflected, Stored, DOM)  
✅ Cross-Site Request Forgery  
✅ Local File Inclusion  
✅ Server-Side Request Forgery  
✅ Server-Side Template Injection  
✅ XML External Entity  
✅ File Upload Vulnerabilities  
✅ JWT Security Issues  
✅ GraphQL API Security  
✅ DOM-based XSS (Playwright)

### Gelişmiş Özellikler

#### 🥷 Stealth & Evasion (V19.0 ULTIMATE)
✅ WAF detection (8 major WAFs)
✅ User-Agent rotation (2026 browsers)
✅ Proxy chains (HTTP/SOCKS5/Tor)
✅ Adaptive rate limiting
✅ TLS fingerprint spoofing
✅ Header randomization
✅ Request timing jitter

#### 🔭 Reconnaissance (V19.0)
✅ Subdomain enumeration (crt.sh)
✅ Technology fingerprinting
✅ Asset discovery & probing
✅ Port scanning integration

#### 💣 Payload Management (V21.0 - 3-TIER)
✅ 3-Tier curated payload system (833 total)
✅ TIER 1: Safe production payloads (90)
✅ TIER 2: Balanced coverage (160)
✅ TIER 3: Aggressive SecLists (583)
✅ Context-aware payload selection
✅ Mutation engine (413 lines)
✅ WAF bypass techniques
✅ Polyglot generation

#### 🔗 Attack Intelligence (V20.0)
✅ Multi-step attack chains
✅ Causation analysis
✅ Pattern recognition
✅ AI-powered path discovery

#### 🎯 Core Features
✅ AI-powered analysis
✅ Intelligent payload mutations
✅ Automatic PoC generation
✅ Evidence integrity verification
✅ Legal compliance checks
✅ Real-time monitoring
✅ Performance auto-tuning

#### 🔍 Verification & Quality (V21.0)
✅ AI-powered triage system
✅ Deterministic auto-accept (99% accuracy)
✅ PoC replay & analysis
✅ Structural pattern matching
✅ Timing analysis
✅ Behavioral verification
✅ Evidence quality standards
✅ Multi-layer validation (<3.2% FP rate)

## 📊 Benchmark Sonuçları

```
Scanner Performance:
├─ SQLi Scanner:     2.48s avg (20 scans/min)
├─ XSS Scanner:      3.21s avg (18 scans/min)
├─ CSRF Scanner:     1.82s avg (33 scans/min)
└─ Overall:          150+ requests/sec

System Resources:
├─ Memory:           2.5GB average
├─ CPU:              45% average
└─ Response Time:    65ms average

Scalability:
├─ Concurrent Scans: 100+ supported
├─ Horizontal:       Linear to 10 instances
└─ Vertical:         Good up to 8 cores
```

Detaylı sonuçlar: [`BENCHMARKS.md`](vortex/BENCHMARKS.md)

## 🔒 Güvenlik ve Yasal Uyumluluk

### ⚠️ Yasal Uyarı

**ÖNEMLİ:** Vortex güvenlik araştırmaları ve yetkili penetrasyon testleri için tasarlanmıştır.

- ✅ **Sadece** yetkili olduğunuz sistemlerde kullanın
- ✅ Yazılı izin almadan tarama yapmayın
- ✅ Yasal ve etik kurallara uyun
- ❌ İzinsiz tarama yasal değildir ve suçtur

### Güvenlik Özellikleri

- Kriptografik evidence integrity
- Audit trail logging
- Legal compliance validator
- Authority hierarchy system
- Scope boundary enforcement

## 🚀 Production Deployment

### Docker

```bash
# Docker ile çalıştır
docker build -t vortex .
docker run -p 5000:5000 vortex

# Docker Compose
docker-compose up -d
```

### Production Checklist

- [ ] Environment variables yapılandırıldı
- [ ] SSL/TLS sertifikaları kuruldu
- [ ] Database backup otomasyonu ayarlandı
- [ ] Log rotation yapılandırıldı
- [ ] Monitoring sistemi kuruldu
- [ ] Rate limiting ayarlandı
- [ ] Firewall kuralları oluşturuldu
- [ ] Incident response planı hazırlandı

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Pull request göndermeden önce:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- OpenRouter API için amazing LLM access
- Tüm açık kaynak katkıda bulunanlara
- Güvenlik araştırma topluluğuna

## 📞 İletişim

- **GitHub**: [Your GitHub Profile]
- **Email**: [Your Email]
- **Issues**: [GitHub Issues](https://github.com/yourusername/Vortex/issues)

---

<p align="center">
  <sub>🛡️ Vortex Security Scanner - Production-Ready Enterprise Security Testing</sub><br>
  <sub>Made with ❤️ for the security community</sub>
</p>