"""
VORTEX Network Layer - V17.0 ULTIMATE
Enhanced HTTP client with resilience and compliance

FEATURES:
- Rate limiting per domain
- Automatic retry with backoff
- Circuit breaker pattern
- WAF detection and evasion
- Scope validation
- Response truncation for memory safety
"""

import asyncio
import aiohttp
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Set
from urllib.parse import urlparse
from collections import defaultdict

from core.exceptions import (
    NetworkError, HTTPRequestError, ConnectionTimeoutError,
    RateLimitExceededError, WAFDetectedError, ScopeViolationError,
    CircuitBreakerOpenError
)

# V21.0 - Performance Profiling Integration
try:
    from core.metrics import global_metrics
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logging.warning("Metrics module not available")

# V19.0 - Stealth Integration
try:
    from core.stealth.evasion import UserAgentRotator, ProxyManager, WAFDetector, ProxyConfig
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False
    logging.warning("Stealth module not available - running in standard mode")

# V20.0 - Adaptive WAF Evasion Integration
try:
    from core.adaptive_waf_evasion import global_evasion_engine
    ADAPTIVE_EVASION_AVAILABLE = True
except ImportError:
    ADAPTIVE_EVASION_AVAILABLE = False
    logging.warning("Adaptive WAF Evasion not available")

# V22.0 - ML-Based Request Timing (PHASE 3.3)
try:
    from core.stealth.request_timing import timing_analyzer
    TIMING_AVAILABLE = True
except ImportError:
    TIMING_AVAILABLE = False
    logging.warning("Request timing analyzer not available")

logger = logging.getLogger(__name__)


@dataclass
class HTTPResponse:
    """HTTP response with metadata."""
    url: str
    status_code: int
    headers: Dict[str, str]
    body: str
    response_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    truncated: bool = False
    original_size: int = 0


@dataclass
class RateLimitState:
    """Per-domain rate limiting state."""
    domain: str
    requests_count: int = 0
    window_start: datetime = field(default_factory=datetime.utcnow)
    last_request: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a domain."""
    domain: str
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    open_until: Optional[datetime] = None


class NetworkClient:
    """
    Production-grade HTTP client with resilience patterns.
    
    FEATURES:
    - Rate limiting (per domain)
    - Automatic retry with exponential backoff
    - Circuit breaker pattern
    - WAF detection
    - Scope validation
    - Memory-safe response handling
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        
        # V21.0 - Metrics integration
        self.metrics = global_metrics if METRICS_ENABLED else None
        
        # Rate limiting (per domain)
        self.rate_limits: Dict[str, RateLimitState] = {}
        self.max_requests_per_minute = 60
        self.rate_limit_window = 60  # seconds
        
        # Circuit breakers (per domain)
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.circuit_breaker_threshold = 5  # failures before opening
        self.circuit_breaker_timeout = 60  # seconds
        
        # Retry configuration
        self.max_retries = 3
        self.retry_backoff_base = 2
        self.retry_statuses = {408, 429, 500, 502, 503, 504}
        
        # Timeouts
        self.connect_timeout = 10
        self.request_timeout = 30
        
        # Response handling
        self.max_response_size = 5 * 1024 * 1024  # 5MB
        self.truncate_at = 1 * 1024 * 1024  # 1MB for memory safety
        
        # Scope validation
        self.allowed_schemes = {'http', 'https'}
        self.blocked_networks: Set[str] = set()
        
        # WAF detection
        self.waf_indicators = [
            'cloudflare', 'incapsula', 'imperva', 'akamai',
            'mod_security', 'wordfence', 'sucuri'
        ]
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'rate_limited': 0,
            'circuit_breaker_opens': 0,
            'waf_detections': 0,
            'stealth_enabled': False
        }

        # Stealth Components (V19.0)
        if STEALTH_AVAILABLE:
            self.ua_rotator = UserAgentRotator()
            self.proxy_manager = ProxyManager()
            self.waf_detector = WAFDetector()
            self.stats['stealth_enabled'] = True
            
            # Configure default stealth settings
            self.use_proxies = True
            self.rotate_ua = True
            self.detect_waf = True
        else:
            self.ua_rotator = None
            self.proxy_manager = None
            self.waf_detector = None
            self.use_proxies = False
            self.rotate_ua = False
            self.detect_waf = False
        
        # V20.0 - Adaptive WAF Evasion Engine
        if ADAPTIVE_EVASION_AVAILABLE:
            self.evasion_engine = global_evasion_engine
            self.adaptive_evasion_enabled = True
            self.stats['adaptive_evasion_enabled'] = True
            logger.info("Adaptive WAF Evasion Engine enabled")
        else:
            self.evasion_engine = None
            self.adaptive_evasion_enabled = False
            self.stats['adaptive_evasion_enabled'] = False
        
        # V22.0 - ML-Based Request Timing (PHASE 3.3)
        if TIMING_AVAILABLE:
            self.timing_analyzer = timing_analyzer
            self.adaptive_timing_enabled = True
            self.stats['adaptive_timing_enabled'] = True
            logger.info("Adaptive request timing enabled (statistical)")
        else:
            self.timing_analyzer = None
            self.adaptive_timing_enabled = False
            self.stats['adaptive_timing_enabled'] = False
    
    async def initialize(self):
        """Initialize HTTP session."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(
                connect=self.connect_timeout,
                total=self.request_timeout
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            logger.info("Network client initialized")
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Network client closed")
    
    async def request(self, 
                     method: str,
                     url: str,
                     **kwargs) -> HTTPResponse:
        """
        Make HTTP request with resilience patterns.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            **kwargs: Additional request parameters
            
        Returns:
            HTTPResponse object
            
        Raises:
            Various NetworkError subclasses
        """
        # Validate scope
        self._validate_scope(url)
        
        # Get domain for rate limiting and circuit breaker
        domain = self._get_domain(url)
        
        # Check circuit breaker
        self._check_circuit_breaker(domain)
        
        # Apply rate limiting
        await self._apply_rate_limit(domain)
        
        # V22.0 - Apply smart delay if timing analyzer available
        if self.adaptive_timing_enabled and self.timing_analyzer:
            await self.timing_analyzer.smart_delay(url)
        
        # Make request with retries
        return await self._request_with_retry(method, url, domain, **kwargs)
    
    def _validate_scope(self, url: str):
        """Validate URL is within allowed scope."""
        try:
            parsed = urlparse(url)
            
            # Check scheme - allow empty for relative URLs in scanning context
            if parsed.scheme and parsed.scheme not in self.allowed_schemes:
                raise ScopeViolationError(f"Scheme not allowed: {parsed.scheme}")
            
            # If no scheme at all, it might be malformed - but be lenient for scanner testing
            if not parsed.scheme and not parsed.netloc:
                logger.debug(f"URL has no scheme/netloc, might be malformed: {url[:100]}")
                # Don't raise, let it fail naturally at request time
                return
            
            # Check for localhost/private IPs (basic check)
            hostname = parsed.hostname
            if hostname:
                hostname_lower = hostname.lower()
                if (hostname_lower in ['localhost', '127.0.0.1'] or
                    hostname_lower.startswith('192.168.') or
                    hostname_lower.startswith('10.') or
                    hostname_lower.startswith('172.')):
                    
                    # Allow if explicitly configured
                    if not self._is_allowed_internal(hostname):
                        raise ScopeViolationError(f"Private network access not allowed: {hostname}")
            
        except ScopeViolationError:
            raise
        except Exception as e:
            # Log but don't fail validation - let request fail naturally
            logger.warning(f"URL validation warning: {e}")
    
    def _is_allowed_internal(self, hostname: str) -> bool:
        """Check if internal network access is allowed."""
        # Would check configuration
        return False
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            # Return "unknown" if domain is empty
            return domain if domain else "unknown"
        except Exception:
            return "unknown"
    
    def _check_circuit_breaker(self, domain: str):
        """Check if circuit breaker is open for domain."""
        # Skip circuit breaker for unknown/empty domains
        if not domain or domain == "unknown":
            return
            
        if domain not in self.circuit_breakers:
            self.circuit_breakers[domain] = CircuitBreakerState(domain=domain)
            return
        
        cb = self.circuit_breakers[domain]
        
        if cb.state == "OPEN":
            # Check if timeout expired
            if cb.open_until and datetime.utcnow() >= cb.open_until:
                # Transition to HALF_OPEN
                cb.state = "HALF_OPEN"
                cb.failure_count = 0
                logger.info(f"Circuit breaker HALF_OPEN for {domain}")
            else:
                # Still open
                retry_after = (cb.open_until - datetime.utcnow()).total_seconds() if cb.open_until else self.circuit_breaker_timeout
                raise CircuitBreakerOpenError(domain, int(retry_after))
    
    async def _apply_rate_limit(self, domain: str):
        """Apply rate limiting for domain."""
        # Skip rate limiting for unknown/empty domains
        if not domain or domain == "unknown":
            return
            
        if domain not in self.rate_limits:
            self.rate_limits[domain] = RateLimitState(domain=domain)
            return
        
        rl = self.rate_limits[domain]
        now = datetime.utcnow()
        
        # Reset window if expired
        window_age = (now - rl.window_start).total_seconds()
        if window_age >= self.rate_limit_window:
            rl.requests_count = 0
            rl.window_start = now
        
        # Check rate limit
        if rl.requests_count >= self.max_requests_per_minute:
            # Calculate wait time
            wait_time = self.rate_limit_window - window_age
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {domain}, waiting {wait_time:.1f}s")
                self.stats['rate_limited'] += 1
                await asyncio.sleep(wait_time)
                # Reset after waiting
                rl.requests_count = 0
                rl.window_start = datetime.utcnow()
        
        # Increment counter
        rl.requests_count += 1
        rl.last_request = now
    
    async def _request_with_retry(self,
                                  method: str,
                                  url: str,
                                  domain: str,
                                  **kwargs) -> HTTPResponse:
        """Make request with automatic retry."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self._make_request(method, url, **kwargs)
                
                # Success - record for circuit breaker
                self._record_success(domain)
                
                return response
                
            except (ConnectionTimeoutError, HTTPRequestError) as e:
                last_error = e
                
                # Check if retryable
                if isinstance(e, HTTPRequestError):
                    if e.details.get('status_code') not in self.retry_statuses:
                        # Not retryable
                        self._record_failure(domain)
                        raise
                
                # Retry with backoff
                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_base ** attempt
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} after {backoff}s")
                    self.stats['retried_requests'] += 1
                    await asyncio.sleep(backoff)
                else:
                    # Max retries exceeded
                    self._record_failure(domain)
                    raise
            
            except Exception as e:
                # Non-retryable error
                self._record_failure(domain)
                raise NetworkError(f"Request failed: {e}") from e
        
        # Should not reach here
        self._record_failure(domain)
        raise last_error or NetworkError("Request failed after retries")
    
    async def _make_request(self, method: str, url: str, **kwargs) -> HTTPResponse:
        """Make actual HTTP request with Stealth features."""
        if not self.session:
            await self.initialize()
        
        start_time = datetime.utcnow()
        self.stats['total_requests'] += 1
        
        # V21.0 - Track request metrics
        request_start = datetime.utcnow()
        
        # Prepare Stealth Headers
        headers = kwargs.get('headers', {})
        if self.rotate_ua and self.ua_rotator:
            domain = self._get_domain(url)
            # Use domain-consistent UA to avoid tripping simple logic
            headers['User-Agent'] = self.ua_rotator.get_for_domain(domain)
        
        # Merge back into kwargs
        kwargs['headers'] = headers
        
        # Prepare Proxy
        proxy = None
        current_proxy_config = None
        
        if self.use_proxies and self.proxy_manager:
            current_proxy_config = self.proxy_manager.get_next_proxy()
            if current_proxy_config:
                # Format for aiohttp: http://user:pass@host:port
                proxy = current_proxy_config.url
                kwargs['proxy'] = proxy
                # Note: aiohttp handles authentication in the proxy URL usually, 
                # or via proxy_auth param. ProxyConfig.url includes auth.
        
        try:
            # Check for ssl verification skip if using proxy (common need)
            if proxy and 'ssl' not in kwargs:
                kwargs['ssl'] = False

            async with self.session.request(method, url, **kwargs) as response:
                # Read response body
                try:
                    # Check size before reading
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self.max_response_size:
                        raise NetworkError(f"Response too large: {content_length} bytes")
                    
                    # Read with limit
                    body_bytes = await response.content.read(self.truncate_at + 1)
                    
                    truncated = False
                    original_size = len(body_bytes)
                    
                    if len(body_bytes) > self.truncate_at:
                        body_bytes = body_bytes[:self.truncate_at]
                        truncated = True
                        logger.debug(f"Response truncated: {original_size} → {self.truncate_at}")
                    
                    # Decode
                    try:
                        body = body_bytes.decode('utf-8', errors='replace')
                    except Exception:
                        body = str(body_bytes)
                    
                except Exception as e:
                    logger.error(f"Failed to read response body: {e}")
                    body = ""
                    truncated = False
                    original_size = 0
                
                # Calculate response time
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Check for WAF
                self._detect_waf(
                    url=str(response.url),
                    status=response.status,
                    headers=dict(response.headers),
                    cookies={k: v.value for k, v in response.cookies.items()},
                    body=body
                )
                
                # Create response object
                http_response = HTTPResponse(
                    url=str(response.url),
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=body,
                    response_time=response_time,
                    truncated=truncated,
                    original_size=original_size
                )
                
                self.stats['successful_requests'] += 1
                
                # V21.0 - Record metrics
                if self.metrics:
                    self.metrics.record_request(
                        url=url,
                        duration=response_time,
                        status_code=response.status
                    )
                
                # V22.0 - Record timing for adaptive delay
                if self.adaptive_timing_enabled and self.timing_analyzer:
                    self.timing_analyzer.record_request(
                        target=url,
                        response_time=response_time,
                        status_code=response.status,
                        response_body=body[:1000] if body else None,  # Sample only
                        success=True
                    )
                
                # Report success to proxy manager
                if current_proxy_config and self.proxy_manager:
                    self.proxy_manager.mark_success(current_proxy_config, response_time * 1000)
                
                return http_response
                
        except asyncio.TimeoutError:
            self.stats['failed_requests'] += 1
            
            # V22.0 - Record timeout for adaptive timing
            if self.adaptive_timing_enabled and self.timing_analyzer:
                self.timing_analyzer.record_request(
                    target=url,
                    response_time=self.request_timeout,
                    status_code=0,
                    success=False
                )
            
            # Report failure to proxy manager
            if current_proxy_config and self.proxy_manager:
                self.proxy_manager.mark_failure(current_proxy_config)
            raise ConnectionTimeoutError(f"Request timeout: {url}")
        
        except aiohttp.ClientError as e:
            self.stats['failed_requests'] += 1
            
            # V22.0 - Record error for adaptive timing
            if self.adaptive_timing_enabled and self.timing_analyzer:
                self.timing_analyzer.record_request(
                    target=url,
                    response_time=0,
                    status_code=0,
                    success=False
                )
            
            # Report failure to proxy manager
            if current_proxy_config and self.proxy_manager:
                self.proxy_manager.mark_failure(current_proxy_config)
            raise HTTPRequestError(url, method, message=str(e))
    
    def _detect_waf(self, url: str, status: int, headers: Dict[str, str],
                   cookies: Dict[str, str], body: str):
        """Detect WAF from response using Stealth module if available."""
        # V20.0 - Use Adaptive Evasion Engine for WAF detection and profiling
        if self.adaptive_evasion_enabled and self.evasion_engine:
            waf_profile = self.evasion_engine.detect_and_profile_waf(
                target_url=url,
                response_headers=headers,
                response_body=body,
                status_code=status
            )
            
            if waf_profile:
                self.stats['waf_detections'] += 1
                logger.warning(
                    f"WAF detected via Adaptive Engine: {waf_profile.waf_type} "
                    f"(confidence: {waf_profile.confidence:.2f})"
                )
                return
        
        # Use advanced WAF detector if available (V19.0)
        elif self.detect_waf and self.waf_detector:
            waf_profile = self.waf_detector.analyze_response(
                url=url,
                status=status,
                headers=headers,
                cookies=cookies,
                body=body
            )
            
            if waf_profile:
                self.stats['waf_detections'] += 1
                logger.warning(f"Advanced WAF detected: {waf_profile.name}")
                return

        # Fallback to legacy basic detection
        # Check headers
        for header, value in headers.items():
            header_lower = header.lower()
            value_lower = value.lower() if value else ""
            
            for indicator in self.waf_indicators:
                if indicator in header_lower or indicator in value_lower:
                    self.stats['waf_detections'] += 1
                    logger.warning(f"Legacy WAF detected: {indicator}")
                    return
        
        # Check body (sample)
        body_sample = body[:1000].lower() if body else ""
        for indicator in self.waf_indicators:
            if indicator in body_sample:
                self.stats['waf_detections'] += 1
                logger.warning(f"Legacy WAF detected in body: {indicator}")
                return
    
    def _record_success(self, domain: str):
        """Record successful request for circuit breaker."""
        # Skip for unknown/empty domains
        if not domain or domain == "unknown":
            return
            
        if domain in self.circuit_breakers:
            cb = self.circuit_breakers[domain]
            
            if cb.state == "HALF_OPEN":
                # Success in half-open, close circuit
                cb.state = "CLOSED"
                cb.failure_count = 0
                logger.info(f"Circuit breaker CLOSED for {domain}")
            elif cb.state == "CLOSED":
                # Reset failure count on success
                cb.failure_count = 0
    
    def _record_failure(self, domain: str):
        """Record failed request for circuit breaker."""
        # Skip for unknown/empty domains
        if not domain or domain == "unknown":
            return
            
        if domain not in self.circuit_breakers:
            self.circuit_breakers[domain] = CircuitBreakerState(domain=domain)
        
        cb = self.circuit_breakers[domain]
        cb.failure_count += 1
        cb.last_failure = datetime.utcnow()
        
        # Check if should open circuit
        if cb.failure_count >= self.circuit_breaker_threshold:
            if cb.state != "OPEN":
                cb.state = "OPEN"
                cb.open_until = datetime.utcnow() + timedelta(seconds=self.circuit_breaker_timeout)
                self.stats['circuit_breaker_opens'] += 1
                logger.error(f"Circuit breaker OPEN for {domain} (failures: {cb.failure_count})")
    
    def get_stats(self) -> Dict[str, int]:
        """Get network statistics."""
        return self.stats.copy()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global network client instance
global_network_client = NetworkClient()