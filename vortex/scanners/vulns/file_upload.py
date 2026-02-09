"""
VORTEX File Upload Vulnerability Scanner - V19.1
Detects insecure file upload implementations with MIME bypass

DETECTION METHODS:
1. Extension bypass (double extensions, null bytes, case sensitivity)
2. Content-Type manipulation
3. Magic byte bypass
4. Path traversal in filenames
5. Web shell upload attempts
6. Executable upload (PHP, JSP, ASP)

AUTHORITY COMPLIANCE:
- Produces HEURISTIC_ONLY detections
- Requires AI analysis and system verification
- Final determination by authority enforcer
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO

from scanners.base import BaseScanner
from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus, ConfidenceSource
from core.network import HTTPResponse

logger = logging.getLogger(__name__)


class FileUploadScanner(BaseScanner):
    """
    File upload vulnerability scanner.
    
    Tests for:
    - Extension bypasses
    - Content-Type bypasses
    - Magic byte validation bypasses
    - Web shell uploads
    - Path traversal in filenames
    """
    
    # Web shell payloads (minimal, safe)
    WEB_SHELL_PAYLOADS = {
        'php': b'<?php echo "VORTEX_TEST_".md5("test"); ?>',
        'jsp': b'<%@ page import="java.security.MessageDigest" %><% out.println("VORTEX_TEST"); %>',
        'asp': b'<%Response.Write("VORTEX_TEST")%>',
        'aspx': b'<%@ Page Language="C#" %><%Response.Write("VORTEX_TEST");%>',
    }
    
    # Extension bypass techniques
    EXTENSION_BYPASSES = [
        # Double extensions
        '{base}.jpg.php',
        '{base}.png.php',
        '{base}.php.jpg',
        
        # Null byte (legacy)
        '{base}.php%00.jpg',
        '{base}.php\x00.jpg',
        
        # Case manipulation
        '{base}.PHP',
        '{base}.pHp',
        '{base}.PhP',
        
        # Special characters
        '{base}.php.',
        '{base}.php ',
        '{base}.php::$DATA',
        
        # Alternative extensions
        '{base}.php3',
        '{base}.php4',
        '{base}.php5',
        '{base}.phtml',
        '{base}.phar',
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        '../{base}.php',
        '..\\{base}.php',
        '../../{base}.php',
        '....//....//....//etc/passwd',
    ]
    
    # Magic bytes for common formats
    MAGIC_BYTES = {
        'png': b'\x89PNG\r\n\x1a\n',
        'gif': b'GIF89a',
        'jpg': b'\xff\xd8\xff',
        'pdf': b'%PDF',
    }
    
    # Indicators of successful upload
    UPLOAD_SUCCESS_INDICATORS = [
        'uploaded successfully',
        'file uploaded',
        'upload complete',
        'saved successfully',
        'file saved',
        'thank you for uploading'
    ]
    
    def __init__(self):
        # Use CODE_INJECTION as finding type since file upload leads to code execution
        super().__init__(FindingType.CODE_INJECTION)
        self.uploaded_files: List[str] = []
        
    async def scan(self, url: str, **kwargs) -> List[AssessmentResult]:
        """
        Scan for file upload vulnerabilities.
        
        Args:
            url: Target upload endpoint
            **kwargs: Optional parameters:
                - file_param: Name of file parameter (default: 'file')
                - additional_params: Additional form fields
        
        Returns:
            List of file upload vulnerability findings
        """
        findings = []
        self.stats['scans_performed'] += 1
        
        file_param = kwargs.get('file_param', 'file')
        additional_params = kwargs.get('additional_params', {})
        
        try:
            # Test 1: Extension bypass
            ext_findings = await self._test_extension_bypass(url, file_param, additional_params)
            findings.extend(ext_findings)
            
            # Test 2: Content-Type manipulation
            ct_findings = await self._test_content_type_bypass(url, file_param, additional_params)
            findings.extend(ct_findings)
            
            # Test 3: Magic byte bypass
            magic_findings = await self._test_magic_byte_bypass(url, file_param, additional_params)
            findings.extend(magic_findings)
            
            # Test 4: Path traversal
            path_findings = await self._test_path_traversal(url, file_param, additional_params)
            findings.extend(path_findings)
            
            self.stats['findings_detected'] += len(findings)
        
        except Exception as e:
            logger.error(f"File upload scan error for {url}: {e}")
        
        return findings
    
    async def _test_extension_bypass(self, url: str, file_param: str,
                                    additional_params: Dict[str, str]) -> List[AssessmentResult]:
        """Test extension validation bypasses."""
        findings = []
        
        # Test PHP shell with various extension bypasses
        base_filename = 'test_shell'
        
        for bypass_pattern in self.EXTENSION_BYPASSES:
            filename = bypass_pattern.format(base=base_filename)
            
            try:
                # Prepare multipart form data
                files = {
                    file_param: (filename, self.WEB_SHELL_PAYLOADS['php'], 'application/octet-stream')
                }
                
                response = await self.network_client.request(
                    'POST',
                    url,
                    data=additional_params,
                    files=files
                )
                self.stats['requests_made'] += 1
                
                # Check if upload succeeded
                if self._is_upload_successful(response):
                    # Try to detect upload path
                    upload_path = self._extract_upload_path(response.body)
                    
                    confidence = 0.75
                    
                    # If we can access uploaded file, increase confidence
                    if upload_path:
                        access_result = await self._verify_file_access(upload_path)
                        if access_result:
                            confidence = 0.90
                    
                    finding = AssessmentResult(
                        id=uuid.uuid4(),
                        url=url,
                        finding_type=FindingType.CODE_INJECTION,
                        severity=FindingSeverity.CRITICAL,
                        status=VerificationStatus.DETECTED,
                        heuristic_score=confidence,
                        confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                        evidence=f"PHP file uploaded with extension bypass: {filename}",
                        vulnerable_parameter=file_param,
                        payload=filename,
                        description=f"File upload validation bypass allows executable upload",
                        remediation="Implement strict file extension whitelist and validate file content"
                    )
                    findings.append(finding)
                    
                    # Track uploaded file
                    if upload_path:
                        self.uploaded_files.append(upload_path)
                    
                    break  # One successful bypass is enough
            
            except Exception as e:
                logger.debug(f"Extension bypass test error ({filename}): {e}")
                continue
        
        return findings
    
    async def _test_content_type_bypass(self, url: str, file_param: str,
                                       additional_params: Dict[str, str]) -> List[AssessmentResult]:
        """Test Content-Type validation bypasses."""
        findings = []
        
        # Test with malicious content but legitimate Content-Type
        legitimate_content_types = [
            'image/png',
            'image/jpeg',
            'image/gif',
            'application/pdf'
        ]
        
        filename = 'shell.php'
        
        for content_type in legitimate_content_types:
            try:
                files = {
                    file_param: (filename, self.WEB_SHELL_PAYLOADS['php'], content_type)
                }
                
                response = await self.network_client.request(
                    'POST',
                    url,
                    data=additional_params,
                    files=files
                )
                self.stats['requests_made'] += 1
                
                if self._is_upload_successful(response):
                    upload_path = self._extract_upload_path(response.body)
                    
                    confidence = 0.80
                    
                    finding = AssessmentResult(
                        id=uuid.uuid4(),
                        url=url,
                        finding_type=FindingType.CODE_INJECTION,
                        severity=FindingSeverity.CRITICAL,
                        status=VerificationStatus.DETECTED,
                        heuristic_score=confidence,
                        confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                        evidence=f"PHP file uploaded with Content-Type bypass: {content_type}",
                        vulnerable_parameter=file_param,
                        payload=f"{filename} (Content-Type: {content_type})",
                        description="File upload bypassed using Content-Type manipulation",
                        remediation="Validate file content, not just Content-Type header"
                    )
                    findings.append(finding)
                    
                    if upload_path:
                        self.uploaded_files.append(upload_path)
                    
                    break
            
            except Exception as e:
                logger.debug(f"Content-Type bypass test error: {e}")
                continue
        
        return findings
    
    async def _test_magic_byte_bypass(self, url: str, file_param: str,
                                     additional_params: Dict[str, str]) -> List[AssessmentResult]:
        """Test magic byte validation bypasses."""
        findings = []
        
        # Test with valid magic bytes + malicious code
        for format_name, magic_bytes in self.MAGIC_BYTES.items():
            filename = f'test.{format_name}.php'
            
            # Prepend magic bytes to payload
            payload = magic_bytes + b'\n' + self.WEB_SHELL_PAYLOADS['php']
            
            try:
                files = {
                    file_param: (filename, payload, f'image/{format_name}')
                }
                
                response = await self.network_client.request(
                    'POST',
                    url,
                    data=additional_params,
                    files=files
                )
                self.stats['requests_made'] += 1
                
                if self._is_upload_successful(response):
                    confidence = 0.85
                    
                    finding = AssessmentResult(
                        id=uuid.uuid4(),
                        url=url,
                        finding_type=FindingType.CODE_INJECTION,
                        severity=FindingSeverity.CRITICAL,
                        status=VerificationStatus.DETECTED,
                        heuristic_score=confidence,
                        confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                        evidence=f"Executable uploaded with {format_name.upper()} magic bytes prepended",
                        vulnerable_parameter=file_param,
                        payload=filename,
                        description="File upload validation bypassed using magic byte prepending",
                        remediation="Validate entire file content, not just magic bytes"
                    )
                    findings.append(finding)
                    break
            
            except Exception as e:
                logger.debug(f"Magic byte bypass test error: {e}")
                continue
        
        return findings
    
    async def _test_path_traversal(self, url: str, file_param: str,
                                   additional_params: Dict[str, str]) -> List[AssessmentResult]:
        """Test path traversal in filenames."""
        findings = []
        
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            filename = pattern.format(base='shell')
            
            try:
                files = {
                    file_param: (filename, self.WEB_SHELL_PAYLOADS['php'], 'application/octet-stream')
                }
                
                response = await self.network_client.request(
                    'POST',
                    url,
                    data=additional_params,
                    files=files
                )
                self.stats['requests_made'] += 1
                
                # Check for path traversal indicators
                if self._is_upload_successful(response):
                    # Check if filename appears in unexpected location
                    if '..' in response.body or 'path' in response.body.lower():
                        confidence = 0.70
                        
                        finding = AssessmentResult(
                            id=uuid.uuid4(),
                            url=url,
                            finding_type=FindingType.CODE_INJECTION,
                            severity=FindingSeverity.HIGH,
                            status=VerificationStatus.DETECTED,
                            heuristic_score=confidence,
                            confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                            evidence=f"File uploaded with path traversal in filename: {filename}",
                            vulnerable_parameter=file_param,
                            payload=filename,
                            description="Path traversal in file upload allows arbitrary file placement",
                            remediation="Sanitize filenames and use secure path joining"
                        )
                        findings.append(finding)
                        break
            
            except Exception as e:
                logger.debug(f"Path traversal test error: {e}")
                continue
        
        return findings
    
    def _is_upload_successful(self, response: HTTPResponse) -> bool:
        """Check if upload was successful."""
        # Success status codes
        if response.status_code not in [200, 201]:
            return False
        
        # Check for success indicators
        body_lower = response.body.lower()
        return any(indicator in body_lower for indicator in self.UPLOAD_SUCCESS_INDICATORS)
    
    def _extract_upload_path(self, body: str) -> Optional[str]:
        """Extract uploaded file path from response."""
        # Common patterns
        patterns = [
            r'(?:path|url|location|file)["\':\s]+([^\s\'"<>]+\.php)',
            r'uploads?/([^\s\'"<>]+)',
            r'/files?/([^\s\'"<>]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex == 1 else match.group(0)
        
        return None
    
    async def _verify_file_access(self, file_path: str) -> bool:
        """Verify if uploaded file is accessible."""
        try:
            # Try to access the file
            response = await self.network_client.request('GET', file_path)
            self.stats['requests_made'] += 1
            
            # If we can access it and see our test string
            if response.status_code == 200 and 'VORTEX_TEST' in response.body:
                return True
        
        except Exception as e:
            logger.debug(f"File access verification error: {e}")
        
        return False
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """
        Generate file upload test payloads.
        
        Returns:
            List of malicious filenames
        """
        payloads = []
        
        base = 'test'
        
        # Extension bypasses
        for pattern in self.EXTENSION_BYPASSES:
            payloads.append(pattern.format(base=base))
        
        # Path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            payloads.append(pattern.format(base=base))
        
        # Simple executables
        payloads.extend([
            'shell.php',
            'shell.jsp',
            'shell.asp',
            'shell.aspx',
        ])
        
        return payloads
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """
        Analyze response for file upload vulnerability indicators.
        
        Args:
            response: HTTP response
            payload: Payload (filename) that was sent
        
        Returns:
            Analysis dict with detection results
        """
        detected = False
        confidence = 0.0
        evidence = ""
        
        # Check if upload was successful
        if self._is_upload_successful(response):
            detected = True
            confidence = 0.75
            evidence = f"File upload succeeded with payload: {payload}"
            
            # Check for uploaded file path
            upload_path = self._extract_upload_path(response.body)
            if upload_path:
                confidence = 0.85
                evidence += f" - File accessible at: {upload_path}"
        
        # Check for validation errors (indicates proper security)
        validation_errors = ['invalid', 'not allowed', 'forbidden', 'rejected', 'denied']
        body_lower = response.body.lower()
        
        if response.status_code in [400, 403] or any(err in body_lower for err in validation_errors):
            evidence = "Upload properly rejected by validation"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'response_analysis': {
                'status_code': response.status_code,
                'upload_successful': self._is_upload_successful(response),
                'upload_path': self._extract_upload_path(response.body)
            }
        }


# Global scanner instance
global_file_upload_scanner: Optional[FileUploadScanner] = None


def get_file_upload_scanner() -> FileUploadScanner:
    """Get or create global file upload scanner instance."""
    global global_file_upload_scanner
    
    if global_file_upload_scanner is None:
        global_file_upload_scanner = FileUploadScanner()
    
    return global_file_upload_scanner