"""
VORTEX SQL Injection Scanner - V17.0 ULTIMATE
Heuristic SQL injection detection

DETECTION METHOD:
- Error-based SQL injection
- Database error pattern matching
- Response differential analysis

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


class SQLInjectionScanner(BaseScanner):
    """
    SQL Injection vulnerability scanner.
    
    Detects error-based SQL injection through:
    - Database error messages
    - Syntax error patterns
    - Response behavior changes
    """
    
    def __init__(self):
        super().__init__(FindingType.SQLI_ERROR)
        self.payload_manager = get_payload_manager()
        
        # SQL error patterns (deterministic indicators)
        self.error_patterns = [
            # MySQL
            r'mysql.*error',
            r'you have an error in your sql syntax',
            r'warning.*mysql_',
            r'unclosed quotation mark after the character string',
            
            # PostgreSQL
            r'postgresql.*error',
            r'pg_query\(\)',
            r'unterminated quoted string',
            
            # MSSQL
            r'microsoft.*sql.*server',
            r'odbc.*sql.*server',
            r'sqlexception',
            
            # Oracle
            r'ora-\d{5}',
            r'oracle.*error',
            
            # Generic
            r'sql.*syntax.*error',
            r'database.*error',
            r'syntax.*error.*near'
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.error_patterns
        ]
    
    async def scan(self, url: str, parameter: str = 'id', **kwargs) -> List:
        """
        Scan URL for SQL injection.
        
        Args:
            url: Target URL
            parameter: Parameter to test (default: 'id')
            
        Returns:
            List of heuristic findings
        """
        logger.info(f"Starting SQL injection scan: {url}")
        
        # Generate payloads
        payloads = self.generate_payloads()
        self.stats['payloads_generated'] += len(payloads)
        
        # Execute scan
        findings = await self.execute_scan(url, parameter, payloads)
        
        logger.info(f"SQL injection scan complete: {len(findings)} findings")
        return findings
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """Generate SQLi test payloads using PayloadManager."""
        # Get payloads from smart manager
        return self.payload_manager.get_payloads(
            attack_type=PayloadType.SQLI,
            technologies=kwargs.get('technologies', [])
        )
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """
        Analyze response for SQL injection indicators.
        
        Args:
            response: HTTP response
            payload: Payload used
            
        Returns:
            Analysis dictionary with detection status
        """
        detected = False
        confidence = 0.0
        evidence_parts = []
        matched_patterns = []
        
        # Check for database error patterns
        for i, pattern in enumerate(self.compiled_patterns):
            match = pattern.search(response.body)
            if match:
                detected = True
                matched_text = match.group(0)
                matched_patterns.append(self.error_patterns[i])
                evidence_parts.append(f"Database error pattern: {matched_text[:100]}")
                
                # Increase confidence based on pattern specificity
                confidence += 0.3
        
        # Check status code (500 errors indicate backend issues)
        if response.status_code >= 500:
            evidence_parts.append(f"Server error: {response.status_code}")
            confidence += 0.2
        
        # Check response time anomalies (for time-based)
        if 'SLEEP' in payload.upper() or 'WAITFOR' in payload.upper():
            if response.response_time > 4.0:  # 5s sleep with margin
                evidence_parts.append(f"Time delay detected: {response.response_time:.1f}s")
                confidence += 0.4
                detected = True
        
        # Cap confidence at 0.95 for heuristic
        confidence = min(confidence, 0.95)
        
        # Build evidence string
        if detected:
            evidence = f"SQL Injection indicators detected:\n"
            evidence += "\n".join(f"- {part}" for part in evidence_parts)
            if matched_patterns:
                evidence += f"\nMatched patterns: {', '.join(matched_patterns[:3])}"
        else:
            evidence = "No SQL injection indicators detected"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'matched_patterns': matched_patterns,
            'response_analysis': {
                'status_code': response.status_code,
                'response_time': response.response_time,
                'body_length': len(response.body)
            }
        }