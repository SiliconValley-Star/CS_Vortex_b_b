# 🌐 Vortex Browser Automation Guide (V22.0)

## 🎭 Stealth Browser Automation (FREE)

Vortex uses **Playwright** for real browser automation with anti-detection features - completely FREE!

---

## 🚀 Features

### ✅ Anti-Bot Detection Bypass (FREE)
- navigator.webdriver removal
- Plugin array mocking
- Chrome runtime hiding
- Canvas fingerprinting noise
- Permissions API mocking

### ✅ Human-Like Behavior (FREE)
- Random mouse movements
- Realistic scrolling
- Variable page interaction delays
- Natural reading time simulation

### ✅ Proxy Support (FREE)
- HTTP/HTTPS proxy
- SOCKS5 proxy
- Tor integration
- User supplies proxy

---

## 📦 Installation

```bash
# Install Playwright
pip install playwright

# Install Chromium browser
playwright install chromium
```

---

## 🎯 Usage

### Basic DOM XSS Scanning

```bash
# Enable DOM XSS scanner
python main.py scan https://example.com --enable-dom
```

### With Proxy

```bash
# Use with Tor
python main.py scan https://example.com \
  --enable-dom \
  --use-tor

# Use with HTTP proxy
python main.py scan https://example.com \
  --enable-dom \
  --proxy http://127.0.0.1:8080
```

### Python API

```python
from scanners.advanced.dom_scanner import PlaywrightDOMScanner

# Initialize with stealth mode (FREE)
scanner = PlaywrightDOMScanner(
    stealth_mode=True,  # Anti-detection
    proxy="socks5://127.0.0.1:9050"  # Tor proxy
)

await scanner.initialize()

# Scan for DOM XSS
results = await scanner.scan_url("https://example.com")

# Close browser
await scanner.close()
```

---

## 🎭 Stealth Mode Features

### 1. Anti-Detection Scripts (FREE)

Automatically injected into every page:

```javascript
// Remove webdriver flag
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined
});

// Mock realistic plugins
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        {name: 'Chrome PDF Plugin'},
        {name: 'Chrome PDF Viewer'},
        {name: 'Native Client'}
    ]
});

// Canvas fingerprinting noise
// Adds minimal randomness to prevent tracking
```

### 2. Human-Like Behavior (FREE)

```python
# Random delays between actions
await asyncio.sleep(0.5 + random.uniform(0, 0.5))

# Mouse movements to random positions
await page.mouse.move(
    random.randint(100, 900),
    random.randint(100, 700)
)

# Realistic scrolling
await page.evaluate("""
    window.scrollTo({
        top: Math.random() * 200,
        behavior: 'smooth'
    });
""")
```

### 3. Realistic Browser Context (FREE)

```python
context_options = {
    'viewport': {'width': 1920, 'height': 1080},
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...',
    'locale': 'en-US',
    'timezone_id': 'America/New_York',
}
```

---

## 🔍 Detection Capabilities

### DOM XSS Detection

- **Reflected XSS**: Payload injection and execution detection
- **DOM-based XSS**: JavaScript sink monitoring
- **Template Injection**: Angular/Vue/React expression evaluation
- **Event Handler Injection**: Mouse/keyboard event payloads

### Monitored Sinks

- `document.write` / `document.writeln`
- `innerHTML` / `outerHTML`
- `eval` / `setTimeout` / `setInterval`
- `location` / `location.href`
- `insertAdjacentHTML`

---

## ⚙️ Configuration

### Timeout Settings

```python
scanner.timeout = 30000  # 30 seconds (default)
```

### Disable Stealth Mode

```python
# If you don't need anti-detection
scanner = PlaywrightDOMScanner(stealth_mode=False)
```

### Custom Browser Arguments

Edit `dom_scanner.py`:

```python
launch_args = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--your-custom-arg',
]
```

---

## 🛠️ Troubleshooting

### Playwright not installed?

```bash
pip install playwright
playwright install chromium
```

### Browser launch fails?

```bash
# Linux: Install dependencies
sudo apt-get install -y \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2
```

### Proxy not working?

```bash
# Test proxy manually
curl --proxy socks5://127.0.0.1:9050 https://check.torproject.org
```

---

## 📊 Performance

### Resource Usage (Headless Chrome)

- **RAM**: ~150-300MB per browser instance
- **CPU**: Low (headless mode)
- **Disk**: ~150MB (Chromium binary)

### Scan Speed

- **Single Page**: 2-5 seconds
- **With Payloads**: 30-60 seconds
- **With Stealth**: +20% overhead (human delays)

---

## 🔒 Security Notes

### ⚠️ Important

1. **Only scan authorized targets**
2. **Respect rate limits**
3. **Use proxies for stealth** (Tor recommended)
4. **Monitor resource usage** (lightweight mode available)

### 💡 Best Practices

- Use `--lightweight` flag for resource-constrained environments
- Combine with `--delay 2.0` for slower, stealthier scans
- Test on staging environments first
- Always get written authorization before scanning

---

## 🎯 Example Workflow

```bash
# 1. Start Tor (optional but recommended)
tor

# 2. Run scan with all stealth features
python main.py scan https://target.com \
  --enable-dom \
  --use-tor \
  --lightweight \
  --delay 2.0 \
  --threads 3

# 3. Review results
# Findings will be in output/reports/
```

---

## 📝 Advanced Usage

### Custom XSS Payloads

Edit `dom_scanner.py`:

```python
XSS_PAYLOADS = [
    '<script>alert(document.domain)</script>',
    # Add your custom payloads
]
```

### DOM Sink Monitoring

```javascript
// Custom sink monitoring
window.__VORTEX_SINKS__ = [];

// Monitor custom sinks
const origFunc = window.dangerousFunction;
window.dangerousFunction = function(data) {
    window.__VORTEX_SINKS__.push({
        sink: 'dangerousFunction',
        data: data
    });
    return origFunc.apply(this, arguments);
};
```

---

**Last Updated:** 2026-01-16  
**Version:** V22.0 (Stealth Browser Automation)  
**Status:** Production-Ready (FREE)