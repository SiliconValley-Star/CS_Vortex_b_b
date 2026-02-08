"""
VORTEX CSRF (Cross-Site Request Forgery) Scanner - V19.0
Detects missing or weak CSRF protection in web applications

DETECTION METHODS:
1. POST/PUT/DELETE requests without CSRF tokens
2. Predictable token patterns
3. Token validation bypass
4. SameSite cookie attribute missing
5. Referer header validation missing

AUTHORITY COMPLIANCE:
- Produces HEURISTIC_ONLY detections
- Requires AI analysis and system verification
- Final determination by authority enforcer
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, parse_qs

from scanners.base import BaseScanner
from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus, ConfidenceSource
from core.network import HTTPResponse

logger = logging.getLogger(__name__)


class CSRFScanner(BaseScanner):
    """
    CSRF vulnerability scanner.
    
    Tests for:
    - Missing CSRF tokens in state-changing operations
    - Weak token implementation
    - SameSite cookie misconfiguration
    - Origin/Referer validation bypass
    """
    
    # Common CSRF token parameter names
    CSRF_TOKEN_NAMES = [
        'csrf_token', 'csrftoken', '_csrf', 'csrf',
        'authenticity_token', '_token', 'token',
        '__RequestVerificationToken', 'anti-csrf-token',
        'xsrf-token', 'xsrf_token'
    ]
    
    # State-changing HTTP methods
    STATE_CHANGING_METHODS = ['POST', 'PUT', 'DELETE', 'PATCH']
    
    def __init__(self):
        super().__init__(FindingType.CSRF)
        self.detected_forms: Set[str] = set()
        
    async def scan(self, url: str, **kwargs) -> List[AssessmentResult]:
        """
        Scan URL for CSRF vulnerabilities.
        
        Args:
            url: Target URL
            **kwargs: Optional parameters:
                - method: HTTP method to test (default: POST)
                - cookies: Session cookies for authenticated testing
                - params: Form parameters
        
        Returns:
            List of CSRF vulnerability findings
        """
        findings = []
        self.stats['scans_performed'] += 1
        
        method = kwargs.get('method', 'POST')
        cookies = kwargs.get('cookies', {})
        params = kwargs.get('params', {})
        
        try:
            # Test 1: Check for CSRF token in form/request
            token_finding = await self._test_csrf_token_presence(url, method, params)
            if token_finding:
                findings.append(token_finding)
                self.stats['findings_detected'] += 1
            
            # Test 2: Check SameSite cookie attribute
            cookie_finding = await self._test_samesite_cookies(url, cookies)
            if cookie_finding:
                findings.append(cookie_finding)
                self.stats['findings_detected'] += 1
            
            # Test 3: Test Origin/Referer validation
            origin_finding = await self._test_origin_validation(url, method, params)
            if origin_finding:
                findings.append(origin_finding)
                self.stats['findings_detected'] += 1
            
            # Test 4: Test token validation if present
            if not token_finding and params:
                validation_finding = await self._test_token_validation(url, method, params)
                if validation_finding:
                    findings.append(validation_finding)
                    self.stats['findings_detected'] += 1
        
        except Exception as e:
            logger.error(f"CSRF scan error for {url}: {e}")
        
        return findings
    
    async def _test_csrf_token_presence(self, url: str, method: str, 
                                       params: Dict[str, str]) -> Optional[AssessmentResult]:
        """Test if CSRF token is present in state-changing request."""
        
        if method not in self.STATE_CHANGING_METHODS:
            return None
        
        # Check if any CSRF token parameter exists
        has_csrf_token = any(
            token_name in params or token_name.lower() in [k.lower() for k in params.keys()]
            for token_name in self.CSRF_TOKEN_NAMES
        )
        
        if not has_csrf_token:
            # Missing CSRF token - HIGH severity
            confidence = 0.75
            
            return AssessmentResult(
                id=uuid.uuid4(),
                url=url,
                finding_type=FindingType.CSRF,
                severity=FindingSeverity.HIGH,
                status=VerificationStatus.DETECTED,
                heuristic_score=confidence,
                confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                evidence=f"No CSRF token found in {method} request to {url}",
                vulnerable_parameter='csrf_protection',
                payload='N/A',
                description=f"State-changing {method} request lacks CSRF protection",
                remediation="Implement CSRF tokens for all state-changing operations"
            )
        
        return None
    
    async def _test_samesite_cookies(self, url: str, 
                                     cookies: Dict[str, str]) -> Optional[AssessmentResult]:
        """Test SameSite cookie attribute."""
        
        try:
            # Make request to get Set-Cookie headers
            response = await self.network_client.request('GET', url)
            self.stats['requests_made'] += 1
            
            # Check Set-Cookie headers
            set_cookie_headers = response.headers.get('set-cookie', '')
            
            if set_cookie_headers and 'samesite' not in set_cookie_headers.lower():
                confidence = 0.65
                
                return AssessmentResult(
                    id=uuid.uuid4(),
                    url=url,
                    finding_type=FindingType.CSRF,
                    severity=FindingSeverity.MEDIUM,
                    status=VerificationStatus.DETECTED,
                    heuristic_score=confidence,
                    confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                    evidence=f"Session cookies lack SameSite attribute: {set_cookie_headers[:200]}",
                    vulnerable_parameter='cookie_security',
                    payload='N/A',
                    description="Missing SameSite cookie attribute allows CSRF attacks",
                    remediation="Set SameSite=Strict or SameSite=Lax for session cookies"
                )
        
        except Exception as e:
            logger.debug(f"SameSite cookie test error: {e}")
        
        return None
    
    async def _test_origin_validation(self, url: str, method: str,
                                      params: Dict[str, str]) -> Optional[AssessmentResult]:
        """Test Origin and Referer header validation."""
        
        if method not in self.STATE_CHANGING_METHODS:
            return None
        
        try:
            # Test with malicious origin
            malicious_origin = "https://evil.com"
            
            headers = {
                'Origin': malicious_origin,
                'Referer': f"{malicious_origin}/attack.html"
            }
            
            response = await self.network_client.request(
                method,
                url,
                headers=headers,
                data=params
            )
            self.stats['requests_made'] += 1
            
            # If request succeeds, origin validation is weak
            if response.status_code in [200, 201, 204]:
                confidence = 0.70
                
                return AssessmentResult(
                    id=uuid.uuid4(),
                    url=url,
                    finding_type=FindingType.CSRF,
                    severity=FindingSeverity.HIGH,
                    status=VerificationStatus.DETECTED,
                    heuristic_score=confidence,
                    confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                    evidence=f"Request accepted with Origin: {malicious_origin}",
                    vulnerable_parameter='origin_validation',
                    payload=malicious_origin,
                    description="Application accepts requests from arbitrary origins",
                    remediation="Implement strict Origin and Referer header validation"
                )
        
        except Exception as e:
            logger.debug(f"Origin validation test error: {e}")
        
        return None
    
    async def _test_token_validation(self, url: str, method: str,
                                     params: Dict[str, str]) -> Optional[AssessmentResult]:
        """Test CSRF token validation strength."""
        
        # Find CSRF token in parameters
        csrf_param = None
        csrf_value = None
        
        for param_name, param_value in params.items():
            if any(token_name.lower() in param_name.lower() 
                   for token_name in self.CSRF_TOKEN_NAMES):
                csrf_param = param_name
                csrf_value = param_value
                break
        
        if not csrf_param:
            return None
        
        try:
            # Test with invalid token
            test_params = params.copy()
            test_params[csrf_param] = "invalid_token_12345"
            
            response = await self.network_client.request(
                method,
                url,
                data=test_params
            )
            self.stats['requests_made'] += 1
            
            # If request succeeds with invalid token, validation is weak
            if response.status_code in [200, 201, 204]:
                confidence = 0.80
                
                return AssessmentResult(
                    id=uuid.uuid4(),
                    url=url,
                    finding_type=FindingType.CSRF,
                    severity=FindingSeverity.HIGH,
                    status=VerificationStatus.DETECTED,
                    heuristic_score=confidence,
                    confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                    evidence=f"Request succeeded with invalid CSRF token: {test_params[csrf_param]}",
                    vulnerable_parameter=csrf_param,
                    payload=test_params[csrf_param],
                    description="CSRF token validation is not enforced",
                    remediation="Implement server-side CSRF token validation"
                )
        
        except Exception as e:
            logger.debug(f"Token validation test error: {e}")
        
        return None
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """
        Generate CSRF test payloads.
        
        Returns:
            List of test tokens/origins
        """
        return [
            # Invalid tokens
            "invalid_csrf_token",
            "12345678",
            "00000000-0000-0000-0000-000000000000",
            "",
            
            # Malicious origins
            "https://evil.com",
            "https://attacker.com",
            "null",
            
            # Token manipulation
            "token123",
            "aaa",
            "zzz"
        ]
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """
        Analyze response for CSRF vulnerability indicators.
        
        Args:
            response: HTTP response
            payload: Payload that was sent
        
        Returns:
            Analysis dict with detection results
        """
        detected = False
        confidence = 0.0
        evidence = ""
        
        # Check if request was successful despite CSRF attack
        if response.status_code in [200, 201, 204]:
            detected = True
            confidence = 0.75
            evidence = f"State-changing request succeeded without proper CSRF protection"
        
        # Check for CSRF error messages (absence indicates vulnerability)
        csrf_error_indicators = [
            'csrf', 'token', 'invalid', 'forbidden',
            'cross-site', 'verification failed'
        ]
        
        body_lower = response.body.lower()
        has_csrf_error = any(indicator in body_lower for indicator in csrf_error_indicators)
        
        if not has_csrf_error and response.status_code == 200:
            detected = True
            confidence = 0.70
            evidence = "No CSRF validation error despite potential attack"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'response_analysis': {
                'status_code': response.status_code,
                'has_csrf_protection': has_csrf_error
            }
        }


# Global scanner instance
global_csrf_scanner: Optional[CSRFScanner] = None


def get_csrf_scanner() -> CSRFScanner:
    """Get or create global CSRF scanner instance."""
    global global_csrf_scanner
    
    if global_csrf_scanner is None:
        global_csrf_scanner = CSRFScanner()
    
    return global_csrf_scanner