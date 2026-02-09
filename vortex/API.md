# Vortex API Documentation

## Overview

Vortex provides both a command-line interface (CLI) and a RESTful web API for security scanning. This document covers the complete API reference for programmatic integration.

## Table of Contents

1. [Authentication](#authentication)
2. [CLI Interface](#cli-interface)
3. [Web API](#web-api)
4. [Python SDK](#python-sdk)
5. [WebSocket Events](#websocket-events)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)

## Authentication

### API Key Authentication

For web API access, use API key authentication:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://localhost:5000/api/scans
```

### Session Authentication

For web dashboard access, use session-based authentication:

```python
import requests

session = requests.Session()
response = session.post('https://localhost:5000/auth/login', json={
    'username': 'admin',
    'password': 'your_password'
})
```

## CLI Interface

### Installation

```bash
cd vortex
pip install -e .
```

### Basic Commands

#### 1. Scan Command

Start a security scan on one or more targets.

```bash
python main.py scan <TARGETS> [OPTIONS]
```

**Arguments:**
- `TARGETS`: One or more target URLs (required)

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode, -m` | choice | active | Scan mode: passive, active, aggressive |
| `--output, -o` | path | - | Output directory for results |
| `--threads, -t` | int | 10 | Number of concurrent threads |
| `--delay, -d` | float | 1.0 | Delay between requests (seconds) |
| `--timeout` | int | 30 | Request timeout (seconds) |
| `--user-agent` | string | - | Custom User-Agent string |
| `--proxy` | url | - | HTTP proxy (http://host:port) |
| `--auth` | string | - | Authentication (username:password) |
| `--headers` | string | - | Custom headers (key:value), can be used multiple times |
| `--scope-file` | path | - | File containing authorized targets |
| `--exclude` | pattern | - | Exclude patterns, can be used multiple times |
| `--include-vulns` | types | - | Include specific vulnerability types |
| `--exclude-vulns` | types | - | Exclude specific vulnerability types |
| `--quality-threshold` | float | 0.7 | Minimum quality threshold (0.0-1.0) |
| `--legal-check` | flag | false | Enable legal compliance checking |
| `--enable-recon` | flag | false | Enable subdomain reconnaissance |
| `--enable-dom` | flag | false | Enable DOM-based XSS scanning |
| `--enable-graphql` | flag | false | Enable GraphQL API scanning |
| `--enable-chains` | flag | false | Enable multi-step attack chains |
| `--enable-mutations` | flag | false | Enable payload mutations for WAF bypass |

**Examples:**

```bash
# Basic scan
python main.py scan https://example.com

# Aggressive scan with multiple targets
python main.py scan https://site1.com https://site2.com --mode aggressive

# Scan with subdomain reconnaissance
python main.py scan https://example.com --enable-recon

# Scan specific vulnerability types
python main.py scan https://example.com --include-vulns sqli xss csrf

# Scan with proxy and custom headers
python main.py scan https://example.com \
  --proxy http://localhost:8080 \
  --headers "Authorization:Bearer token123"

# Full-featured scan
python main.py scan https://example.com \
  --mode aggressive \
  --threads 20 \
  --delay 0.5 \
  --enable-recon \
  --enable-dom \
  --enable-chains \
  --enable-mutations \
  --quality-threshold 0.8
```

#### 2. Report Command

Generate security reports from scan results.

```bash
python main.py report [SCAN_ID] [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--format, -f` | choice | markdown | Report format: json, html, markdown, pdf |
| `--output, -o` | path | - | Output file path |
| `--include-poc` | flag | true | Include Proof of Concept |

**Examples:**

```bash
# Generate markdown report
python main.py report --format markdown --output report.md

# Generate HTML report
python main.py report --format html --output report.html

# Generate JSON report without PoC
python main.py report --format json --include-poc=false
```

#### 3. Status Command

Show system status and health metrics.

```bash
python main.py status
```

**Output:**
- System component health
- Resource utilization
- Active scans
- Queue status

## Web API

### Base URL

```
http://localhost:5000/api
```

### Endpoints

#### 1. Start Scan

Start a new security scan.

**Endpoint:** `POST /api/scans`

**Request Body:**
```json
{
  "target": "https://example.com",
  "scan_types": ["sqli", "xss", "csrf"],
  "mode": "active",
  "options": {
    "enable_recon": false,
    "enable_chains": false,
    "quality_threshold": 0.7
  }
}
```

**Response:**
```json
{
  "scan_id": "scan-123456",
  "status": "running",
  "target": "https://example.com",
  "started_at": "2024-01-15T10:30:00Z"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/scans \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://example.com",
    "scan_types": ["sqli", "xss"]
  }'
```

#### 2. Get Scan Status

Get the status of a running or completed scan.

**Endpoint:** `GET /api/scans/<scan_id>`

**Response:**
```json
{
  "scan_id": "scan-123456",
  "status": "completed",
  "target": "https://example.com",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z",
  "findings_count": 5,
  "progress": 100
}
```

**Example:**
```bash
curl http://localhost:5000/api/scans/scan-123456
```

#### 3. Get Findings

Retrieve findings from a scan.

**Endpoint:** `GET /api/scans/<scan_id>/findings`

**Query Parameters:**
- `severity`: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)
- `status`: Filter by status (VERIFIED, PENDING_VERIFICATION, etc.)
- `limit`: Maximum number of results (default: 100)
- `offset`: Pagination offset (default: 0)

**Response:**
```json
{
  "total": 5,
  "findings": [
    {
      "id": "finding-001",
      "type": "SQL_INJECTION",
      "severity": "HIGH",
      "status": "VERIFIED",
      "url": "https://example.com/search?q=test",
      "parameter": "q",
      "payload": "' OR '1'='1",
      "evidence": "SQL error detected",
      "confidence": 0.95,
      "discovered_at": "2024-01-15T10:32:00Z"
    }
  ]
}
```

**Example:**
```bash
# Get all findings
curl http://localhost:5000/api/scans/scan-123456/findings

# Get only high severity findings
curl http://localhost:5000/api/scans/scan-123456/findings?severity=HIGH

# Pagination
curl http://localhost:5000/api/scans/scan-123456/findings?limit=10&offset=20
```

#### 4. Stop Scan

Stop a running scan.

**Endpoint:** `POST /api/scans/<scan_id>/stop`

**Response:**
```json
{
  "scan_id": "scan-123456",
  "status": "stopped",
  "message": "Scan stopped successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/scans/scan-123456/stop
```

#### 5. Get System Status

Get overall system health and metrics.

**Endpoint:** `GET /api/status`

**Response:**
```json
{
  "status": "healthy",
  "uptime": 86400,
  "active_scans": 3,
  "queue_size": 15,
  "memory_usage": 65.5,
  "cpu_usage": 45.2,
  "components": {
    "database": "healthy",
    "queue": "healthy",
    "network": "healthy",
    "ai": "healthy"
  }
}
```

**Example:**
```bash
curl http://localhost:5000/api/status
```

#### 6. Export Report

Export scan results in various formats.

**Endpoint:** `GET /api/scans/<scan_id>/export`

**Query Parameters:**
- `format`: Report format (json, html, markdown)
- `include_poc`: Include PoC (true/false)

**Response:**
- Content-Type varies by format
- Download link or report data

**Example:**
```bash
# Export as HTML
curl http://localhost:5000/api/scans/scan-123456/export?format=html \
  -o report.html

# Export as JSON
curl http://localhost:5000/api/scans/scan-123456/export?format=json \
  > report.json
```

## Python SDK

### Installation

```python
from core.engine import VortexScanEngine
from domain.enums import ScanMode
```

### Basic Usage

```python
import asyncio
from core.engine import VortexScanEngine

async def main():
    # Initialize engine
    engine = VortexScanEngine()
    await engine.initialize()
    
    try:
        # Start scan
        results = await engine.scan_target(
            target_url="https://example.com",
            scan_types=['sqli', 'xss', 'csrf'],
            enable_recon=False
        )
        
        # Process results
        if results and results.get('findings'):
            for finding in results['findings']:
                print(f"Found: {finding['type']} at {finding['url']}")
        
    finally:
        await engine.shutdown()

asyncio.run(main())
```

### Advanced Usage

```python
import asyncio
from core.engine import VortexScanEngine
from domain.enums import ScanMode, Severity

async def advanced_scan():
    engine = VortexScanEngine()
    await engine.initialize()
    
    try:
        # Configure advanced options
        results = await engine.scan_target(
            target_url="https://example.com",
            scan_types=['sqli', 'xss', 'csrf', 'lfi', 'ssrf'],
            enable_recon=True,        # Enable subdomain discovery
            enable_chains=True,       # Enable attack chains
            mode=ScanMode.AGGRESSIVE
        )
        
        # Filter findings by severity
        if results and results.get('findings'):
            critical = [f for f in results['findings'] 
                       if f.get('severity') == 'CRITICAL']
            
            print(f"Found {len(critical)} critical vulnerabilities")
            
            # Get detailed findings from database
            findings = await engine.get_findings(
                severity=Severity.CRITICAL,
                limit=10
            )
            
            for finding in findings:
                print(f"\nType: {finding.finding_type}")
                print(f"URL: {finding.url}")
                print(f"Confidence: {finding.heuristic_score:.2%}")
                
    finally:
        await engine.shutdown()

asyncio.run(advanced_scan())
```

### Scanner Integration

```python
from scanners.vulns.xss import XSSScanner
from core.network import HTTPClient

async def custom_scan():
    http_client = HTTPClient()
    scanner = XSSScanner()
    scanner.http_client = http_client
    
    try:
        findings = await scanner.scan("https://example.com/search?q=test")
        
        for finding in findings:
            print(f"XSS found: {finding.payload}")
            
    finally:
        await http_client.close()

asyncio.run(custom_scan())
```

## WebSocket Events

### Connection

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected to Vortex');
});
```

### Events

#### scan_started
```javascript
socket.on('scan_started', (data) => {
    console.log('Scan started:', data.scan_id);
});
```

#### scan_progress
```javascript
socket.on('scan_progress', (data) => {
    console.log(`Progress: ${data.progress}%`);
    console.log(`Findings: ${data.findings_count}`);
});
```

#### finding_discovered
```javascript
socket.on('finding_discovered', (finding) => {
    console.log('New finding:', finding.type);
    console.log('Severity:', finding.severity);
});
```

#### scan_completed
```javascript
socket.on('scan_completed', (data) => {
    console.log('Scan completed');
    console.log('Total findings:', data.total_findings);
});
```

#### system_status
```javascript
socket.on('system_status', (status) => {
    console.log('Memory:', status.memory_percent);
    console.log('Queue size:', status.queue_size);
});
```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error occurred |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_TARGET",
    "message": "Target URL is not accessible",
    "details": {
      "url": "https://example.com",
      "reason": "Connection timeout"
    }
  }
}
```

### Common Errors

#### Invalid Target
```json
{
  "error": {
    "code": "INVALID_TARGET",
    "message": "Target URL is invalid or not accessible"
  }
}
```

#### Rate Limit Exceeded
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests. Please try again later.",
    "retry_after": 60
  }
}
```

#### Scan Not Found
```json
{
  "error": {
    "code": "SCAN_NOT_FOUND",
    "message": "Scan with ID scan-123456 not found"
  }
}
```

## Rate Limiting

### Limits

- **CLI**: No rate limiting
- **Web API**: 100 requests per minute per IP
- **WebSocket**: 1000 events per minute

### Headers

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1610000000
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_retry(url):
    while True:
        response = requests.get(url)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
            time.sleep(retry_after)
            continue
            
        return response
```

## Best Practices

### 1. Use Async Operations

```python
# Good - Async
results = await engine.scan_target(url)

# Avoid - Blocking
results = asyncio.run(engine.scan_target(url))
```

### 2. Handle Errors Gracefully

```python
try:
    results = await engine.scan_target(url)
except ConnectionError:
    logger.error("Failed to connect to target")
except TimeoutError:
    logger.error("Scan timeout")
```

### 3. Implement Pagination

```python
offset = 0
limit = 100

while True:
    findings = await engine.get_findings(offset=offset, limit=limit)
    if not findings:
        break
    
    process_findings(findings)
    offset += limit
```

### 4. Use WebSockets for Real-time Updates

```javascript
socket.on('scan_progress', (data) => {
    updateProgressBar(data.progress);
    updateFindingsCount(data.findings_count);
});
```

### 5. Close Resources

```python
try:
    await engine.scan_target(url)
finally:
    await engine.shutdown()  # Always cleanup
```

## Examples

### Complete Scan Workflow

```python
import asyncio
from core.engine import VortexScanEngine
from domain.enums import Severity

async def complete_workflow():
    engine = VortexScanEngine()
    await engine.initialize()
    
    try:
        # 1. Start scan
        print("Starting scan...")
        results = await engine.scan_target(
            target_url="https://example.com",
            scan_types=['sqli', 'xss', 'csrf'],
            enable_recon=True
        )
        
        # 2. Process findings
        if results and results.get('findings'):
            print(f"Found {len(results['findings'])} vulnerabilities")
            
            # 3. Filter critical findings
            critical = await engine.get_findings(
                severity=Severity.CRITICAL
            )
            
            # 4. Generate report
            from utils.poc_generator import poc_generator
            
            for finding in critical:
                poc = poc_generator.generate_poc(finding.to_dict())
                print(f"\n{poc['markdown']}")
        
    finally:
        await engine.shutdown()

asyncio.run(complete_workflow())
```

## Support

For additional support:
- Documentation: `vortex/docs/`
- Issues: GitHub Issues
- Examples: `vortex/examples/`