# Vortex Architecture Documentation

## System Overview

Vortex is an enterprise-grade bug bounty automation framework designed for comprehensive web application security testing. The system employs a modular, scalable architecture with advanced features including AI-powered analysis, memory management, and legal compliance.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Vortex Scanner                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐         ┌────────────────┐                  │
│  │   CLI Layer    │         │   Web Layer    │                  │
│  │   (main.py)    │         │  (Flask/SocketIO)                 │
│  └───────┬────────┘         └───────┬────────┘                  │
│          │                          │                            │
│          └──────────┬───────────────┘                           │
│                     │                                            │
│          ┌──────────▼──────────┐                                │
│          │   Core Engine       │                                │
│          │  (VortexScanEngine) │                                │
│          └──────────┬──────────┘                                │
│                     │                                            │
│     ┌───────────────┼───────────────┐                           │
│     │               │               │                           │
│ ┌───▼────┐    ┌────▼─────┐   ┌────▼─────┐                      │
│ │Queue   │    │  Network │   │ Database │                      │
│ │Manager │    │  Client  │   │ Manager  │                      │
│ └───┬────┘    └────┬─────┘   └────┬─────┘                      │
│     │              │               │                            │
│ ┌───▼──────────────▼───────────────▼─────┐                     │
│ │         Scanner Layer                   │                     │
│ │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │                     │
│ │  │ SQLi │ │ XSS  │ │ CSRF │ │ LFI  │  │                     │
│ │  └──────┘ └──────┘ └──────┘ └──────┘  │                     │
│ │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │                     │
│ │  │ SSRF │ │ SSTI │ │ XXE  │ │ JWT  │  │                     │
│ │  └──────┘ └──────┘ └──────┘ └──────┘  │                     │
│ └─────────────────────────────────────────┘                     │
│                                                                   │
│ ┌───────────────────────────────────────────┐                   │
│ │       Support Systems                      │                   │
│ │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │                   │
│ │  │AI Engine │ │  Memory  │ │ Profiler │  │                   │
│ │  │(OpenRouter)│ Manager  │ │          │  │                   │
│ │  └──────────┘ └──────────┘ └──────────┘  │                   │
│ │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │                   │
│ │  │  Legal   │ │ Evidence │ │  Health  │  │                   │
│ │  │Guardian  │ │Integrity │ │ Monitor  │  │                   │
│ │  └──────────┘ └──────────┘ └──────────┘  │                   │
│ └───────────────────────────────────────────┘                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. VortexScanEngine (core/engine.py)

The central orchestrator that coordinates all scanning activities.

**Responsibilities:**
- Initialize and manage scanner instances
- Coordinate scan execution across multiple targets
- Manage queue-based task processing
- Handle scan lifecycle (start, pause, resume, stop)
- Aggregate and deduplicate findings
- Integrate with AI analysis and verification systems

**Key Methods:**
```python
async def initialize() -> None
async def scan_target(target_url: str, scan_types: List[str]) -> Dict
async def get_findings(limit: int = 100) -> List[Finding]
async def shutdown() -> None
```

### 2. Scanner Layer (scanners/)

Modular vulnerability scanners implementing the `BaseScanner` interface.

**Scanner Types:**
- **Web Vulnerabilities**: SQLi, XSS, LFI, SSRF, CSRF, SSTI, XXE, File Upload
- **API Security**: JWT, GraphQL
- **Advanced**: DOM XSS (Playwright-based)

**Base Scanner Interface:**
```python
class BaseScanner:
    async def scan(self, url: str) -> List[Finding]
    async def verify(self, finding: Finding) -> bool
    def get_payloads(self) -> List[str]
```

### 3. Queue Manager (core/queue_manager.py)

Priority-based asynchronous task queue for managing scan workloads.

**Features:**
- Priority-based task scheduling
- Backpressure handling
- Concurrent task execution limits
- Thread-safe operations
- Memory-efficient queue management

**Queue Structure:**
```python
class QueueManager:
    async def enqueue(task: Dict, priority: int = 5) -> bool
    async def dequeue() -> Optional[Dict]
    async def start() -> None
    async def stop() -> None
```

### 4. Network Layer (core/network.py)

Resilient HTTP client with advanced retry and circuit breaker patterns.

**Features:**
- Automatic retry with exponential backoff
- Circuit breaker for failing endpoints
- Connection pooling and reuse
- Request/response logging
- Timeout management
- Proxy support

**Resilience Components:**
```python
class HTTPClient:
    - RetryManager: Exponential backoff, jitter
    - CircuitBreaker: Fail-fast on persistent errors
    - Connection Pool: Efficient connection reuse
    - Rate Limiter: Request throttling
```

### 5. Database Layer (core/database.py)

Persistent storage for findings, scan history, and metadata.

**Schema:**
```
findings: id, target_url, type, severity, status, evidence, timestamp
scans: id, target, start_time, end_time, findings_count
evidence: finding_id, type, content, integrity_hash
```

**Operations:**
```python
async def save_finding(finding: Finding) -> str
async def get_findings(filters: Dict) -> List[Finding]
async def update_finding_status(id: str, status: FindingStatus) -> bool
```

## Advanced Features

### 1. AI-Powered Analysis (core/ai/)

Integration with OpenRouter for intelligent vulnerability analysis.

**Components:**
- **Advisory System**: AI-driven exploit generation
- **False Positive Filter**: ML-based finding validation
- **Payload Mutations**: AI-generated WAF bypasses

**Workflow:**
```
Finding → AI Analysis → Confidence Score → Verification → Submit Ready
```

### 2. Memory Management (core/memory_manager.py)

Multi-zone dynamic memory management system.

**Memory Zones:**
- **GREEN** (0-60%): Normal operations
- **YELLOW** (60-75%): Warning, apply optimizations
- **ORANGE** (75-85%): Stress, reduce workload
- **RED** (85%+): Critical, emergency cleanup

**Features:**
- Real-time memory monitoring
- Automatic garbage collection
- Cache eviction strategies
- Resource pool management

### 3. Legal Compliance (core/legal_compliance.py)

Enterprise-grade compliance and authorization framework.

**Features:**
- Target authorization validation
- Scope boundary enforcement
- Evidence chain-of-custody
- Audit trail logging
- Rate limiting compliance

### 4. Evidence Integrity (core/evidence_integrity.py)

Cryptographic evidence validation and tamper detection.

**Components:**
- SHA-256 hashing for evidence
- Merkle tree for batch validation
- Timestamp verification
- Chain-of-custody tracking

### 5. Health Monitoring (core/health/)

System health tracking and auto-tuning.

**Metrics:**
- CPU/Memory utilization
- Request success/failure rates
- Queue depth and processing time
- Scanner performance
- Database connection health

**Auto-tuning:**
- Dynamic concurrency adjustment
- Automatic resource allocation
- Performance optimization

## Data Flow

### Scan Execution Flow

```
1. User Request (CLI/Web)
   ↓
2. VortexScanEngine.scan_target()
   ↓
3. Queue Manager (enqueue tasks)
   ↓
4. Queue Processor (dequeue and distribute)
   ↓
5. Scanner Execution (parallel)
   ↓
6. Finding Collection
   ↓
7. AI Analysis & Verification
   ↓
8. Database Storage
   ↓
9. Evidence Integrity Check
   ↓
10. Results Return to User
```

### Finding Lifecycle

```
DETECTED → PENDING_VERIFICATION → VERIFIED → SUBMIT_READY → SUBMITTED
                ↓
              FALSE_POSITIVE (filtered out)
```

## Scalability Considerations

### Horizontal Scaling

The architecture supports horizontal scaling through:

1. **Stateless Scanners**: Can run multiple scanner instances
2. **Shared Queue**: Distributed task processing
3. **Database Connection Pooling**: Multiple writers/readers
4. **Load Balancing**: Web interface can scale horizontally

### Performance Optimization

1. **Caching Layer**: Response caching to reduce redundant requests
2. **Connection Pooling**: Reuse HTTP connections
3. **Async I/O**: Non-blocking operations throughout
4. **Batch Processing**: Aggregate database writes
5. **Memory Zones**: Proactive memory management

## Security Architecture

### Defense in Depth

1. **Input Validation**: All user inputs sanitized
2. **Output Encoding**: Prevent injection in reports
3. **Authentication**: API key and session-based auth
4. **Authorization**: Role-based access control
5. **Audit Logging**: All operations logged
6. **Encryption**: Evidence encrypted at rest

### Secure Development

1. **Type Safety**: Strong typing with Python type hints
2. **Error Handling**: Comprehensive exception handling
3. **Testing**: 70%+ code coverage
4. **Code Review**: All changes reviewed
5. **Dependency Scanning**: Regular security updates

## Extension Points

### Custom Scanners

Create new scanners by extending `BaseScanner`:

```python
from scanners.base import BaseScanner

class CustomScanner(BaseScanner):
    async def scan(self, url: str) -> List[Finding]:
        # Custom scan logic
        pass
```

### Custom AI Providers

Extend AI analysis with new providers:

```python
from core.ai.openrouter import OpenRouterClient

class CustomAIProvider(OpenRouterClient):
    async def analyze_finding(self, finding: Finding) -> Dict:
        # Custom AI logic
        pass
```

### Custom Evidence Collectors

Add new evidence collection methods:

```python
from core.evidence_integrity import EvidenceCollector

class CustomCollector(EvidenceCollector):
    async def collect(self, finding: Finding) -> Dict:
        # Custom collection logic
        pass
```

## Configuration

### Environment Variables

```bash
# Database
DB_PATH=output/database/vortex.db

# Network
HTTP_TIMEOUT=30
MAX_CONCURRENT=50
RATE_LIMIT=100

# AI
OPENROUTER_API_KEY=your_key
DEFAULT_AI_MODEL=anthropic/claude-opus-4

# Memory
MAX_MEMORY_MB=8192
MEMORY_WARNING_THRESHOLD=75

# Legal
ENABLE_LEGAL_CHECKS=false
SCOPE_FILE=scope.txt
```

### Configuration Files

- `config/settings.py`: Main configuration
- `config/prompts.py`: AI prompt templates
- `config/legal_config.py`: Legal compliance rules
- `config/health_thresholds.py`: Health monitoring thresholds

## Deployment

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up database with proper permissions
- [ ] Configure reverse proxy (nginx/caddy)
- [ ] Enable SSL/TLS
- [ ] Set up monitoring and alerting
- [ ] Configure log rotation
- [ ] Test backup and restore procedures
- [ ] Review security settings
- [ ] Load test critical paths
- [ ] Document incident response procedures

### Monitoring

**Key Metrics:**
- Scan throughput (scans/hour)
- Finding accuracy (false positive rate)
- System resource utilization
- API response times
- Error rates by component

## Future Enhancements

### Planned Features

1. **Distributed Scanning**: Multi-node scan coordination
2. **Machine Learning**: Advanced finding classification
3. **Blockchain Evidence**: Immutable evidence storage
4. **Real-time Collaboration**: Multi-user scanning
5. **Plugin Ecosystem**: Community scanner extensions

### Performance Goals

- **Scan Speed**: 100+ requests/second per node
- **Memory Efficiency**: < 4GB for typical workload
- **Availability**: 99.9% uptime
- **Response Time**: < 100ms API responses
- **Scalability**: Support 1000+ concurrent scans

## References

- [API Documentation](API.md)
- [Performance Analysis](PERFORMANCE.md)
- [Benchmark Results](BENCHMARKS.md)
- [Usage Guide](KULLANIM_KILAVUZU.md)
- [Legal Compliance](LEGAL_COMPLIANCE.md)