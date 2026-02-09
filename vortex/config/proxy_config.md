# 🔒 Vortex Proxy Configuration Guide

## ⚠️ FREE Proxy Methods Only

Vortex supports **FREE proxy methods** for development, testing, and educational use:

- ✅ Tor SOCKS5 proxy (FREE)
- ✅ Local intercepting proxies like Burp Suite, OWASP ZAP (FREE)
- ✅ Your own proxy list (FREE)
- ✅ SOCKS4/SOCKS5/HTTP/HTTPS protocols (FREE)

**For real-world bug bounty workflows**, third-party residential or ISP proxies are recommended and **must be supplied by the user**.

---

## 📋 Usage Examples

### 1. Tor Proxy (100% FREE)

```bash
# Install and start Tor
brew install tor  # macOS
sudo apt install tor  # Linux

# Run Tor
tor

# Use with Vortex
cd vortex
python main.py scan https://example.com --use-tor
```

### 2. Single Proxy (FREE)

```bash
# HTTP proxy
python main.py scan https://example.com \
  --proxy http://127.0.0.1:8080

# SOCKS5 proxy
python main.py scan https://example.com \
  --proxy socks5://127.0.0.1:1080
```

### 3. Proxy List File (FREE)

Create `proxies.txt`:
```
# Format: host:port or protocol://host:port
1.2.3.4:8080
5.6.7.8:3128
socks5://9.10.11.12:1080
http://proxy.example.com:8888
```

Use with Vortex:
```bash
python main.py scan https://example.com \
  --proxy-list proxies.txt
```

### 4. Local Intercepting Proxy (FREE)

```bash
# Burp Suite default
python main.py scan https://example.com \
  --proxy http://127.0.0.1:8080

# OWASP ZAP default
python main.py scan https://example.com \
  --proxy http://127.0.0.1:8081
```

---

## 🔧 Advanced Configuration

### Proxy with Authentication

Create `proxies.txt`:
```
http://username:password@proxy.example.com:8080
```

### Multiple Proxy Types

```
# Mix different protocols
socks5://127.0.0.1:9050          # Tor
http://127.0.0.1:8080            # Burp Suite
http://user:pass@proxy.com:3128  # Authenticated HTTP
```

---

## 📊 Proxy Statistics

View proxy stats during scan:
```bash
python main.py scan https://example.com \
  --proxy-list proxies.txt \
  --verbose
```

---

## ⚠️ Important Notes

### FREE vs PAID

✅ **FREE Methods:**
- Tor network (anonymity)
- Local intercepting proxies (Burp, ZAP)
- Your own proxy list
- Public proxies (unreliable)

❌ **NOT Included (User Must Supply):**
- Residential proxy services (Bright Data, Oxylabs, etc.)
- Datacenter proxy pools
- Premium proxy providers

### Recommendations

**For Development/Testing:**
- Use Tor or local intercepting proxies
- Perfect for learning and testing

**For Bug Bounty Programs:**
- Use your own residential/ISP proxies
- Third-party services require paid subscription
- Vortex supports them but doesn't provide them

---

## 🚀 Best Practices

1. **Always test proxies first**
   ```bash
   curl --proxy socks5://127.0.0.1:9050 https://check.torproject.org
   ```

2. **Rotate proxies for stealth**
   - Use `--proxy-list` with multiple proxies
   - Vortex automatically rotates them

3. **Monitor proxy health**
   - Vortex tracks failures and bans bad proxies
   - Check verbose output for proxy statistics

4. **Respect rate limits**
   - Use `--delay` flag to slow down requests
   - Recommended: 1-2 seconds between requests

---

## 🛠️ Troubleshooting

### Tor not working?
```bash
# Check if Tor is running
ps aux | grep tor

# Start Tor manually
tor
```

### Proxy connection failed?
```bash
# Test proxy manually
curl --proxy http://127.0.0.1:8080 https://httpbin.org/ip
```

### All proxies banned?
- Wait 30 minutes (automatic cooldown)
- Or restart scan with fresh proxies

---

## 📝 Example Workflow

```bash
# 1. Start Tor
tor

# 2. Create proxy list
cat > proxies.txt << EOF
# Tor
socks5://127.0.0.1:9050
# Burp Suite
http://127.0.0.1:8080
EOF

# 3. Run scan with proxies
cd vortex
python main.py scan https://target.com \
  --proxy-list proxies.txt \
  --lightweight \
  --delay 2.0 \
  --threads 3
```

---

**Last Updated:** 2026-01-16
**Version:** 1.0.0 (FREE Proxy Support)