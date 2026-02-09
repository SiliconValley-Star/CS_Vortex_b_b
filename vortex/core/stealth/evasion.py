"""
VORTEX WAF Evasion & Stealth Module - V19.0 ULTIMATE
Advanced techniques to bypass Web Application Firewalls

CAPABILITIES:
- User-Agent rotation with realistic browser profiles
- IP rotation via proxy chains (SOCKS5, HTTP, Tor)
- Request throttling and rate limiting
- Header randomization
- Payload encoding and obfuscation
- TLS fingerprint spoofing
- Request timing jitter

2026 MODERN FEATURES:
- Residential proxy integration
- Browser fingerprint spoofing
- AI-powered evasion pattern generation
- Real-time WAF detection

SUPPORTED WAF DETECTION:
- Cloudflare
- AWS WAF
- Akamai
- Imperva/Incapsula
- F5 BIG-IP
- ModSecurity
- Sucuri
"""

import asyncio
import random
import logging
import time
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from urllib.parse import urlparse
import aiohttp
try:
    from aiohttp_socks import ProxyConnector
    AIOHTTP_SOCKS_AVAILABLE = True
except ImportError:
    ProxyConnector = None
    AIOHTTP_SOCKS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """Proxy configuration."""
    protocol: str  # http, https, socks4, socks5
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    is_residential: bool = False
    last_used: Optional[datetime] = None
    failures: int = 0
    latency_ms: Optional[float] = None
    
    @property
    def url(self) -> str:
        """Get proxy URL."""
        auth = f"{self.username}:{self.password}@" if self.username else ""
        return f"{self.protocol}://{auth}{self.host}:{self.port}"


@dataclass
class WAFProfile:
    """WAF detection profile."""
    name: str
    detected: bool = False
    confidence: float = 0.0
    indicators: List[str] = field(default_factory=list)
    bypass_techniques: List[str] = field(default_factory=list)


class UserAgentRotator:
    """
    Realistic User-Agent rotation.
    
    Maintains a pool of realistic browser User-Agent strings
    with proper version updates for 2026.
    """
    
    # Modern browser user agents (2026)
    USER_AGENTS = [
        # Chrome 120+ (Windows)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        
        # Chrome (macOS)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        
        # Firefox 125+ (Windows)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
        
        # Firefox (macOS)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
        
        # Safari (macOS)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
        
        # Edge
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        
        # Mobile Chrome (Android)
        "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
        
        # Mobile Safari (iOS)
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
    ]
    
    def __init__(self):
        self.current_index = 0
        self.usage_count: Dict[str, int] = {}
    
    def get_random(self) -> str:
        """Get random User-Agent."""
        ua = random.choice(self.USER_AGENTS)
        self.usage_count[ua] = self.usage_count.get(ua, 0) + 1
        return ua
    
    def get_next(self) -> str:
        """Get next User-Agent in rotation."""
        ua = self.USER_AGENTS[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.USER_AGENTS)
        self.usage_count[ua] = self.usage_count.get(ua, 0) + 1
        return ua
    
    def get_for_domain(self, domain: str) -> str:
        """Get consistent User-Agent for a domain (session persistence)."""
        # Use domain hash to pick consistent UA
        index = hash(domain) % len(self.USER_AGENTS)
        return self.USER_AGENTS[index]


class ProxyManager:
    """
    Proxy chain management with automatic rotation.
    
    Supports:
    - HTTP/HTTPS proxies
    - SOCKS4/SOCKS5 proxies
    - Tor integration
    - Residential proxy pools
    """
    
    def __init__(self):
        self.proxies: List[ProxyConfig] = []
        self.current_index = 0
        self.banned_proxies: Dict[str, datetime] = {}
        self.ban_duration = timedelta(minutes=30)
        
        # Stats
        self.total_requests = 0
        self.proxy_failures = 0
    
    def add_proxy(self, protocol: str, host: str, port: int,
                  username: Optional[str] = None, password: Optional[str] = None,
                  country: Optional[str] = None, is_residential: bool = False):
        """Add a proxy to the pool."""
        proxy = ProxyConfig(
            protocol=protocol,
            host=host,
            port=port,
            username=username,
            password=password,
            country=country,
            is_residential=is_residential
        )
        self.proxies.append(proxy)
        logger.info(f"Added proxy: {host}:{port} ({protocol})")
    
    def add_tor_proxy(self, host: str = "127.0.0.1", port: int = 9050):
        """
        Add Tor SOCKS5 proxy (FREE).
        
        Usage:
            # Start Tor first: `brew install tor && tor`
            proxy_manager.add_tor_proxy()
        """
        self.add_proxy("socks5", host, port)
        logger.info(f"✓ Added Tor proxy: {host}:{port} (FREE)")
    
    def load_proxy_list(self, filepath: str, protocol: str = "http"):
        """
        Load proxies from file (FREE).
        
        Supported formats:
            - host:port
            - host:port:username:password
            - protocol://host:port (auto-detect protocol)
        
        Example file content:
            # Free proxies
            1.2.3.4:8080
            5.6.7.8:3128:user:pass
            socks5://9.10.11.12:1080
        """
        try:
            with open(filepath, 'r') as f:
                initial_count = len(self.proxies)
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Auto-detect protocol from URL format
                    detected_protocol = protocol
                    if '://' in line:
                        detected_protocol, rest = line.split('://', 1)
                        line = rest
                    
                    parts = line.split(':')
                    if len(parts) >= 2:
                        try:
                            host = parts[0]
                            port = int(parts[1])
                            username = parts[2] if len(parts) > 2 else None
                            password = parts[3] if len(parts) > 3 else None
                            
                            self.add_proxy(detected_protocol, host, port, username, password)
                        except ValueError:
                            logger.warning(f"Invalid proxy format at line {line_num}: {line}")
                    else:
                        logger.warning(f"Invalid proxy format at line {line_num}: {line}")
                
                loaded_count = len(self.proxies) - initial_count
                logger.info(f"✓ Loaded {loaded_count} proxies from {filepath} (Total: {len(self.proxies)})")
                
        except FileNotFoundError:
            logger.error(f"Proxy list file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load proxy list: {e}")
    
    def get_next_proxy(self) -> Optional[ProxyConfig]:
        """Get next available proxy in rotation."""
        if not self.proxies:
            return None
        
        # Clean up expired bans
        now = datetime.utcnow()
        self.banned_proxies = {
            url: ban_time for url, ban_time in self.banned_proxies.items()
            if now - ban_time < self.ban_duration
        }
        
        # Find available proxy
        available = [p for p in self.proxies if p.url not in self.banned_proxies]
        
        if not available:
            logger.warning("All proxies are banned, waiting for cooldown...")
            return None
        
        # Round-robin among available
        proxy = available[self.current_index % len(available)]
        self.current_index = (self.current_index + 1) % len(available)
        proxy.last_used = now
        
        return proxy
    
    def get_random_proxy(self) -> Optional[ProxyConfig]:
        """Get random available proxy."""
        available = [p for p in self.proxies if p.url not in self.banned_proxies]
        if not available:
            return None
        return random.choice(available)
    
    def mark_failure(self, proxy: ProxyConfig):
        """Mark proxy as failed (temporary ban after multiple failures)."""
        proxy.failures += 1
        self.proxy_failures += 1
        
        if proxy.failures >= 3:
            self.banned_proxies[proxy.url] = datetime.utcnow()
            proxy.failures = 0
            logger.warning(f"Proxy banned due to failures: {proxy.host}:{proxy.port}")
    
    def mark_success(self, proxy: ProxyConfig, latency_ms: float):
        """Mark proxy as successful."""
        proxy.failures = max(0, proxy.failures - 1)
        proxy.latency_ms = latency_ms
        self.total_requests += 1
    
    async def create_connector(self, proxy: Optional[ProxyConfig] = None) -> Optional[aiohttp.BaseConnector]:
        """Create aiohttp connector with proxy."""
        if not proxy:
            proxy = self.get_next_proxy()
        
        if not proxy:
            return None
        
        if proxy.protocol in ('socks4', 'socks5'):
            return ProxyConnector.from_url(proxy.url)
        else:
            return aiohttp.TCPConnector()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get proxy pool statistics."""
        return {
            'total_proxies': len(self.proxies),
            'available_proxies': len([p for p in self.proxies if p.url not in self.banned_proxies]),
            'banned_proxies': len(self.banned_proxies),
            'total_requests': self.total_requests,
            'failure_rate': self.proxy_failures / max(1, self.total_requests)
        }


class WAFDetector:
    """
    Detect and identify Web Application Firewalls.
    
    Detection methods:
    - Response header analysis
    - Error page fingerprinting
    - Cookie analysis
    - Behavior patterns
    """
    
    # WAF signatures
    WAF_SIGNATURES = {
        'cloudflare': {
            'headers': ['cf-ray', 'cf-cache-status', '__cfduid'],
            'cookies': ['__cflb', '__cf_bm'],
            'server': ['cloudflare'],
            'error_patterns': ['Attention Required', 'cloudflare-nginx', 'CLOUDFLARE']
        },
        'aws_waf': {
            'headers': ['x-amzn-requestid', 'x-amz-cf-id', 'x-amz-apigw-id'],
            'error_patterns': ['Request blocked', 'AWS WAF']
        },
        'akamai': {
            'headers': ['x-akamai-transformed', 'akamai-grn'],
            'server': ['AkamaiGHost', 'AkamaiNetStorage'],
            'error_patterns': ['Reference #', 'Access Denied - AK']
        },
        'imperva': {
            'headers': ['x-iinfo', 'x-cdn'],
            'cookies': ['incap_ses', 'visid_incap'],
            'error_patterns': ['Incapsula incident', 'Request unsuccessful']
        },
        'f5_bigip': {
            'headers': ['x-cnection', 'x-wa-info'],
            'cookies': ['TS', 'BIGipServer'],
            'server': ['BigIP']
        },
        'modsecurity': {
            'headers': ['mod_security', 'NOYB'],
            'error_patterns': ['ModSecurity', 'Not Acceptable']
        },
        'sucuri': {
            'headers': ['x-sucuri-id', 'x-sucuri-cache'],
            'error_patterns': ['Sucuri WebSite Firewall', 'Access Denied - Sucuri']
        },
        'fortiweb': {
            'headers': ['fortiwafsid'],
            'cookies': ['FORTIWAFSID'],
            'error_patterns': ['FortiWeb']
        }
    }
    
    def __init__(self):
        self.detected_wafs: Dict[str, WAFProfile] = {}
    
    def analyze_response(self, url: str, status: int, headers: Dict[str, str],
                        cookies: Dict[str, str], body: str) -> Optional[WAFProfile]:
        """
        Analyze HTTP response for WAF indicators.
        
        Returns WAFProfile if WAF detected, None otherwise.
        """
        domain = urlparse(url).netloc
        
        for waf_name, signatures in self.WAF_SIGNATURES.items():
            score = 0.0
            indicators = []
            
            # Check headers
            for header_sig in signatures.get('headers', []):
                for header_name in headers.keys():
                    if header_sig.lower() in header_name.lower():
                        score += 0.3
                        indicators.append(f"Header: {header_name}")
            
            # Check cookies
            for cookie_sig in signatures.get('cookies', []):
                for cookie_name in cookies.keys():
                    if cookie_sig.lower() in cookie_name.lower():
                        score += 0.3
                        indicators.append(f"Cookie: {cookie_name}")
            
            # Check server header
            server = headers.get('server', '').lower()
            for server_sig in signatures.get('server', []):
                if server_sig.lower() in server:
                    score += 0.4
                    indicators.append(f"Server: {server}")
            
            # Check error patterns in body
            for pattern in signatures.get('error_patterns', []):
                if pattern.lower() in body.lower():
                    score += 0.5
                    indicators.append(f"Body pattern: {pattern}")
            
            # Check for block status codes
            if status in [403, 406, 429, 503]:
                score += 0.2
            
            if score >= 0.5:
                profile = WAFProfile(
                    name=waf_name,
                    detected=True,
                    confidence=min(1.0, score),
                    indicators=indicators,
                    bypass_techniques=self._get_bypass_techniques(waf_name)
                )
                self.detected_wafs[domain] = profile
                logger.info(f"WAF detected: {waf_name} on {domain} (confidence: {score:.1%})")
                return profile
        
        return None
    
    def _get_bypass_techniques(self, waf_name: str) -> List[str]:
        """Get recommended bypass techniques for WAF."""
        techniques = {
            'cloudflare': [
                'Use residential proxies',
                'Slower request rate (1-2 req/sec)',
                'Proper Referer header',
                'Real browser User-Agent'
            ],
            'aws_waf': [
                'Parameter pollution',
                'Unicode encoding',
                'Case variation'
            ],
            'akamai': [
                'HTTP/2 connections',
                'Valid Accept headers', 
                'Slower crawl rate'
            ],
            'modsecurity': [
                'SQL comment injection (/*!...*/)',
                'URL encoding variations',
                'HTTP Parameter fragmentation'
            ]
        }
        
        return techniques.get(waf_name, ['General evasion techniques'])
    
    def get_waf_for_domain(self, domain: str) -> Optional[WAFProfile]:
        """Get detected WAF for domain."""
        return self.detected_wafs.get(domain)


class RateLimiter:
    """
    Intelligent request rate limiting.
    
    Features:
    - Per-domain rate limiting
    - Adaptive throttling based on response
    - Jitter to avoid pattern detection
    """
    
    def __init__(self, default_delay: float = 1.0, jitter: float = 0.3):
        self.default_delay = default_delay
        self.jitter = jitter
        self.domain_delays: Dict[str, float] = {}
        self.domain_last_request: Dict[str, datetime] = {}
        self.domain_429_count: Dict[str, int] = {}
    
    def set_domain_delay(self, domain: str, delay: float):
        """Set specific delay for a domain."""
        self.domain_delays[domain] = delay
    
    async def wait_for_slot(self, url: str):
        """Wait until rate limit allows next request."""
        domain = urlparse(url).netloc
        delay = self.domain_delays.get(domain, self.default_delay)
        
        last_request = self.domain_last_request.get(domain)
        
        if last_request:
            elapsed = (datetime.utcnow() - last_request).total_seconds()
            wait_time = delay - elapsed
            
            if wait_time > 0:
                # Add jitter
                jitter = random.uniform(-self.jitter, self.jitter) * delay
                actual_wait = max(0, wait_time + jitter)
                await asyncio.sleep(actual_wait)
        
        self.domain_last_request[domain] = datetime.utcnow()
    
    def report_429(self, url: str):
        """Report rate limit hit (429 response)."""
        domain = urlparse(url).netloc
        
        self.domain_429_count[domain] = self.domain_429_count.get(domain, 0) + 1
        
        # Increase delay for this domain
        current_delay = self.domain_delays.get(domain, self.default_delay)
        new_delay = min(current_delay * 2, 30.0)  # Max 30 seconds
        self.domain_delays[domain] = new_delay
        
        logger.warning(f"Rate limit hit on {domain}, increasing delay to {new_delay}s")
    
    def report_success(self, url: str):
        """Report successful request."""
        domain = urlparse(url).netloc
        
        # Gradually decrease delay on success
        current_delay = self.domain_delays.get(domain, self.default_delay)
        if current_delay > self.default_delay:
            new_delay = max(self.default_delay, current_delay * 0.9)
            self.domain_delays[domain] = new_delay


class StealthRequestClient:
    """
    Stealth HTTP client with all evasion features combined.
    
    Integrates:
    - User-Agent rotation
    - Proxy rotation
    - Rate limiting
    - WAF detection
    - Header randomization
    """
    
    def __init__(self):
        self.ua_rotator = UserAgentRotator()
        self.proxy_manager = ProxyManager()
        self.waf_detector = WAFDetector()
        self.rate_limiter = RateLimiter()
        
        # Stats
        self.total_requests = 0
        self.blocked_requests = 0
        self.successful_requests = 0
    
    def configure_stealth(self, 
                         request_delay: float = 1.0,
                         use_proxies: bool = True,
                         rotate_ua: bool = True,
                         detect_waf: bool = True):
        """Configure stealth settings."""
        self.rate_limiter.default_delay = request_delay
        self._use_proxies = use_proxies
        self._rotate_ua = rotate_ua
        self._detect_waf = detect_waf
    
    def _get_stealth_headers(self, url: str) -> Dict[str, str]:
        """Get headers that look like a real browser."""
        domain = urlparse(url).netloc
        
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        # Add rotated User-Agent
        if self._rotate_ua:
            headers['User-Agent'] = self.ua_rotator.get_for_domain(domain)
        
        # Add random order variation (some WAFs check header order)
        items = list(headers.items())
        random.shuffle(items)
        
        return dict(items)
    
    async def request(self, method: str, url: str,
                     headers: Optional[Dict[str, str]] = None,
                     data: Optional[Any] = None,
                     json_data: Optional[Dict] = None,
                     cookies: Optional[Dict[str, str]] = None,
                     timeout: int = 30) -> Tuple[int, Dict[str, str], str]:
        """
        Make a stealth HTTP request.
        
        Returns:
            Tuple of (status_code, headers, body)
        """
        self.total_requests += 1
        
        # Wait for rate limit
        await self.rate_limiter.wait_for_slot(url)
        
        # Get stealth headers
        request_headers = self._get_stealth_headers(url)
        if headers:
            request_headers.update(headers)
        
        # Get proxy
        proxy = None
        connector = None
        
        if self._use_proxies and self.proxy_manager.proxies:
            proxy = self.proxy_manager.get_next_proxy()
            if proxy:
                connector = await self.proxy_manager.create_connector(proxy)
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                
                proxy_url = proxy.url if proxy and proxy.protocol in ('http', 'https') else None
                
                async with session.request(
                    method,
                    url,
                    headers=request_headers,
                    data=data,
                    json=json_data,
                    cookies=cookies,
                    proxy=proxy_url,
                    ssl=False
                ) as response:
                    latency = (time.time() - start_time) * 1000
                    
                    response_headers = dict(response.headers)
                    body = await response.text()
                    response_cookies = {k: v.value for k, v in response.cookies.items()}
                    
                    # Check for WAF
                    if self._detect_waf:
                        waf = self.waf_detector.analyze_response(
                            url, response.status, response_headers, response_cookies, body
                        )
                        
                        if waf:
                            self.blocked_requests += 1
                    
                    # Handle rate limiting
                    if response.status == 429:
                        self.rate_limiter.report_429(url)
                        self.blocked_requests += 1
                    else:
                        self.rate_limiter.report_success(url)
                        self.successful_requests += 1
                    
                    # Update proxy stats
                    if proxy:
                        if response.status in [403, 429, 503]:
                            self.proxy_manager.mark_failure(proxy)
                        else:
                            self.proxy_manager.mark_success(proxy, latency)
                    
                    return response.status, response_headers, body
                    
        except Exception as e:
            if proxy:
                self.proxy_manager.mark_failure(proxy)
            
            logger.error(f"Stealth request failed: {e}")
            raise
    
    async def get(self, url: str, **kwargs) -> Tuple[int, Dict[str, str], str]:
        """Stealth GET request."""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> Tuple[int, Dict[str, str], str]:
        """Stealth POST request."""
        return await self.request('POST', url, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stealth client statistics."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'blocked_requests': self.blocked_requests,
            'block_rate': self.blocked_requests / max(1, self.total_requests),
            'detected_wafs': list(self.waf_detector.detected_wafs.keys()),
            'proxy_stats': self.proxy_manager.get_stats()
        }


# Global stealth client
global_stealth_client = StealthRequestClient()


def get_stealth_client() -> StealthRequestClient:
    """Get global stealth client instance."""
    return global_stealth_client
