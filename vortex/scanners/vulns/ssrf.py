"""
VORTEX SSRF Scanner - V17.0 ULTIMATE
Heuristic Server-Side Request Forgery detection

DETECTION METHOD:
- Internal network access indicators
- Response differential analysis
- Time-based detection

EVIDENCE LEVEL: HEURISTIC_ONLY
"""

import re
import logging
from typing import List, Dict, Any

from domain.enums import FindingType
from core.network import HTTPResponse
from scanners.base import BaseScanner
from core.payloads.manager import get_payload_manager, PayloadType

logger = logging.getLogger(__name__)


class SSRFScanner(BaseScanner):
    """
    SSRF vulnerability scanner.
    
    Detects SSRF through:
    - Internal network response indicators
    - Private IP address access
    - Localhost connections
    """
    
    def __init__(self):
        super().__init__(FindingType.SSRF)
        self.payload_manager = get_payload_manager()
        
        # Internal network indicators
        self.internal_indicators = [
            # Private IP patterns
            r'192\.168\.\d+\.\d+',
            r'10\.\d+\.\d+\.\d+',
            r'172\.(1[6-9]|2\d|3[01])\.\d+\.\d+',
            r'127\.0\.0\.1',
            r'localhost',
            
            # Internal services
            r'internal',
            r'metadata',
            r'169\.254\.169\.254',  # AWS metadata
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.internal_indicators
        ]
    
    async def scan(self, url: str, parameter: str = 'url', **kwargs) -> List:
        """Scan URL for SSRF."""
        logger.info(f"Starting SSRF scan: {url}")
        
        payloads = self.generate_payloads()
        self.stats['payloads_generated'] += len(payloads)
        
        findings = await self.execute_scan(url, parameter, payloads)
        
        logger.info(f"SSRF scan complete: {len(findings)} findings")
        return findings
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """Generate SSRF test payloads using PayloadManager."""
        return self.payload_manager.get_payloads(
            attack_type=PayloadType.SSRF,
            technologies=kwargs.get('technologies', [])
        )
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """Analyze response for SSRF indicators."""
        detected = False
        confidence = 0.0
        evidence_parts = []
        matched_patterns = []
        
        # Check for internal network indicators in response
        for i, pattern in enumerate(self.compiled_patterns):
            match = pattern.search(response.body)
            if match:
                detected = True
                matched_text = match.group(0)
                matched_patterns.append(self.internal_indicators[i])
                evidence_parts.append(f"Internal network indicator: {matched_text}")
                confidence += 0.4
        
        # Check for successful internal access
        if response.status_code == 200:
            # If payload contained internal address and we got 200
            if any(indicator in payload for indicator in ['localhost', '127.0.0.1', '192.168', '10.', '172.']):
                detected = True
                confidence += 0.3
                evidence_parts.append(f"Successful response to internal address: {response.status_code}")
        
        # Check for metadata service indicators
        if '169.254.169.254' in payload:
            if 'ami-id' in response.body or 'instance-id' in response.body:
                detected = True
                confidence += 0.5
                evidence_parts.append("AWS metadata service content detected")
        
        # Check response time (internal requests are usually faster)
        if response.response_time < 0.1:  # Very fast response
            if any(indicator in payload for indicator in ['localhost', '127.0.0.1']):
                confidence += 0.2
                evidence_parts.append(f"Fast response time ({response.response_time:.3f}s) suggests local access")
        
        # Cap at 0.95
        confidence = min(confidence, 0.95)
        
        if detected:
            evidence = f"SSRF indicators detected:\n"
            evidence += "\n".join(f"- {part}" for part in evidence_parts)
            if matched_patterns:
                evidence += f"\nMatched patterns: {', '.join(matched_patterns[:3])}"
        else:
            evidence = "No SSRF indicators detected"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'matched_patterns': matched_patterns
        }