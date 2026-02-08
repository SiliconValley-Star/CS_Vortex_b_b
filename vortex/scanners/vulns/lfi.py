"""
VORTEX LFI Scanner - V17.0 ULTIMATE
Heuristic Local File Inclusion detection

DETECTION METHOD:
- File content pattern matching
- System file indicators
- Path traversal detection

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


class LFIScanner(BaseScanner):
    """
    Local File Inclusion vulnerability scanner.
    
    Detects LFI through:
    - System file content patterns
    - File path indicators
    - Directory traversal
    """
    
    def __init__(self):
        super().__init__(FindingType.LFI)
        self.payload_manager = get_payload_manager()
        
        # File content patterns
        self.file_patterns = [
            # /etc/passwd
            r'root:.*:0:0:',
            r'bin:.*:/bin/',
            r'daemon:.*:/sbin/',
            
            # Windows
            r'\[boot loader\]',
            r'\[operating systems\]',
            
            # PHP files
            r'<\?php',
            r'\$_GET',
            r'\$_POST',
            
            # Config files
            r'mysql.*host',
            r'database.*password',
            r'DB_PASSWORD'
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.file_patterns
        ]
    
    async def scan(self, url: str, parameter: str = 'file', **kwargs) -> List:
        """Scan URL for LFI."""
        logger.info(f"Starting LFI scan: {url}")
        
        payloads = self.generate_payloads()
        self.stats['payloads_generated'] += len(payloads)
        
        findings = await self.execute_scan(url, parameter, payloads)
        
        logger.info(f"LFI scan complete: {len(findings)} findings")
        return findings
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """Generate LFI test payloads using PayloadManager."""
        return self.payload_manager.get_payloads(
            attack_type=PayloadType.LFI,
            technologies=kwargs.get('technologies', [])
        )
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """Analyze response for LFI indicators."""
        detected = False
        confidence = 0.0
        evidence_parts = []
        matched_patterns = []
        
        # Check for file content patterns
        for i, pattern in enumerate(self.compiled_patterns):
            match = pattern.search(response.body)
            if match:
                detected = True
                matched_text = match.group(0)
                matched_patterns.append(self.file_patterns[i])
                evidence_parts.append(f"File content pattern: {matched_text[:100]}")
                confidence += 0.4
        
        # Check for specific file indicators
        if '/etc/passwd' in payload:
            if 'root:' in response.body and ':0:0:' in response.body:
                detected = True
                confidence += 0.5
                evidence_parts.append("Unix passwd file structure detected")
        
        if 'win.ini' in payload or 'boot.ini' in payload:
            if '[' in response.body and ']' in response.body:
                detected = True
                confidence += 0.4
                evidence_parts.append("Windows INI file structure detected")
        
        # Check response length (unusually long might indicate file content)
        if len(response.body) > 1000:
            confidence += 0.1
            evidence_parts.append(f"Long response ({len(response.body)} bytes)")
        
        # Cap at 0.95
        confidence = min(confidence, 0.95)
        
        if detected:
            evidence = f"LFI indicators detected:\n"
            evidence += "\n".join(f"- {part}" for part in evidence_parts)
            if matched_patterns:
                evidence += f"\nMatched patterns: {', '.join(matched_patterns[:3])}"
        else:
            evidence = "No LFI indicators detected"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'matched_patterns': matched_patterns
        }