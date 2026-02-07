"""
VORTEX Base Scanner - V17.0 ULTIMATE
Base class for all vulnerability scanners

CRITICAL: All scanners produce HEURISTIC_ONLY evidence.
This is NOT authoritative - requires AI analysis and system verification.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import quote_plus

from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus, ConfidenceSource
from core.network import global_network_client, HTTPResponse
from core.exceptions import ScannerError, PayloadGenerationError

logger = logging.getLogger(__name__)


class BaseScanner(ABC):
    """
    Abstract base scanner for vulnerability detection.
    
    SCANNER ROLE:
    - Generate payloads
    - Execute heuristic detection
    - Calculate confidence scores
    - Mark findings as HEURISTIC_ONLY
    
    NOT RESPONSIBLE FOR:
    - AI analysis (done by workflow)
    - System verification (done by verification engine)
    - Final determination (done by authority enforcer)
    """
    
    def __init__(self, finding_type: FindingType):
        self.finding_type = finding_type
        self.network_client = global_network_client
        
        # Scanner statistics
        self.stats = {
            'scans_performed': 0,
            'findings_detected': 0,
            'payloads_generated': 0,
            'requests_made': 0
        }
    
    @abstractmethod
    async def scan(self, url: str, **kwargs) -> List[AssessmentResult]:
        """
        Scan target URL for vulnerability.
        
        Args:
            url: Target URL to scan
            **kwargs: Scanner-specific parameters
            
        Returns:
            List of assessment results (heuristic detections)
        """
        pass
    
    @abstractmethod
    def generate_payloads(self, **kwargs) -> List[str]:
        """
        Generate vulnerability-specific payloads.
        
        Returns:
            List of test payloads
        """
        pass
    
    @abstractmethod
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """
        Analyze response for vulnerability indicators.
        
        Args:
            response: HTTP response
            payload: Payload that was sent
            
        Returns:
            Analysis dict with confidence and evidence
        """
        pass
    
    async def execute_scan(self,
                          url: str,
                          parameter: str,
                          payloads: List[str]) -> List[AssessmentResult]:
        """
        Execute scan with given payloads.
        
        Common scan execution logic for all scanners.
        """
        findings = []
        self.stats['scans_performed'] += 1
        
        for payload in payloads:
            try:
                # Build test URL
                test_url = self._build_test_url(url, parameter, payload)
                
                # Make request
                response = await self.network_client.request('GET', test_url)
                self.stats['requests_made'] += 1
                
                # Analyze response
                analysis = self.analyze_response(response, payload)
                
                # Check if vulnerability detected
                if analysis['detected']:
                    finding = self._create_finding(
                        url=url,
                        parameter=parameter,
                        payload=payload,
                        response=response,
                        analysis=analysis
                    )
                    findings.append(finding)
                    self.stats['findings_detected'] += 1
                    
                    logger.info(f"Heuristic detection: {self.finding_type.value} at {url}")
                
            except Exception as e:
                logger.error(f"Scan error for payload '{payload}': {e}")
                continue
        
        return findings
    
    def _build_test_url(self, url: str, parameter: str, payload: str) -> str:
        """Build test URL with payload - URL encodes payload for safe transmission."""
        # Ensure URL has a scheme (http:// or https://)
        if not url.startswith(('http://', 'https://')):
            # Add https:// as default scheme
            url = f"https://{url}"
        
        separator = '&' if '?' in url else '?'
        # URL encode payload to handle special characters safely
        encoded_payload = quote_plus(payload)
        return f"{url}{separator}{parameter}={encoded_payload}"
    
    def _create_finding(self,
                       url: str,
                       parameter: str,
                       payload: str,
                       response: HTTPResponse,
                       analysis: Dict[str, Any]) -> AssessmentResult:
        """
        Create assessment result from heuristic detection.
        
        CRITICAL: Marks finding as HEURISTIC_ONLY.
        """
        finding = AssessmentResult(
            id=uuid.uuid4(),
            url=url,
            finding_type=self.finding_type,
            severity=self._determine_severity(analysis),
            status=VerificationStatus.DETECTED,
            
            # Heuristic confidence
            heuristic_score=analysis['confidence'],
            confidence_source=ConfidenceSource.HEURISTIC_ONLY,  # CRITICAL marker
            
            # Evidence
            evidence=analysis['evidence'],
            vulnerable_parameter=parameter,
            payload=payload,
            
            # Response metadata in metadata dict
            metadata={
                'response_status_code': response.status_code,
                'response_time': response.response_time,
                'response_body_length': len(response.body),
                'analysis': analysis.get('response_analysis', {})
            },
            
            # Timestamp
            detected_at=datetime.utcnow()
        )
        
        return finding
    
    def _determine_severity(self, analysis: Dict[str, Any]) -> FindingSeverity:
        """Determine finding severity from analysis."""
        confidence = analysis.get('confidence', 0.0)
        
        if confidence >= 0.8:
            return FindingSeverity.HIGH
        elif confidence >= 0.6:
            return FindingSeverity.MEDIUM
        else:
            return FindingSeverity.LOW
    
    def get_stats(self) -> Dict[str, int]:
        """Get scanner statistics."""
        return self.stats.copy()