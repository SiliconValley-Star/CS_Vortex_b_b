"""
Test suite for File Upload Scanner
Tests extension bypass, content-type manipulation, and upload validation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from scanners.vulns.file_upload import FileUploadScanner, get_file_upload_scanner
from core.network import HTTPResponse
from domain.enums import FindingType, FindingSeverity


class TestFileUploadScanner:
    """Test File Upload Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return FileUploadScanner()
    
    @pytest.fixture
    def mock_response(self):
        def _create(status_code=200, body=""):
            response = Mock(spec=HTTPResponse)
            response.status_code = status_code
            response.body = body
            response.headers = {}
            response.response_time = 0.1
            return response
        return _create
    
    def test_scanner_initialization(self, scanner):
        assert scanner.finding_type == FindingType.CODE_INJECTION
        assert len(FileUploadScanner.WEB_SHELL_PAYLOADS) > 0
        assert len(FileUploadScanner.EXTENSION_BYPASSES) > 0
    
    def test_web_shell_payloads_defined(self, scanner):
        """Test web shell payloads for different languages."""
        assert 'php' in FileUploadScanner.WEB_SHELL_PAYLOADS
        assert 'jsp' in FileUploadScanner.WEB_SHELL_PAYLOADS
        assert 'asp' in FileUploadScanner.WEB_SHELL_PAYLOADS
        
        # Check payloads are safe test payloads
        php_payload = FileUploadScanner.WEB_SHELL_PAYLOADS['php']
        assert b'VORTEX_TEST' in php_payload
    
    def test_extension_bypasses_defined(self, scanner):
        """Test extension bypass techniques."""
        bypasses = FileUploadScanner.EXTENSION_BYPASSES
        
        # Should include double extensions
        assert any('.jpg.php' in b for b in bypasses)
        # Should include null byte
        assert any('%00' in b for b in bypasses)
        # Should include case variations
        assert any('.PHP' in b for b in bypasses)
    
    def test_generate_payloads(self, scanner):
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        # Should include various bypass techniques
        assert any('.php' in p for p in payloads)
        assert any('../' in p for p in payloads)
    
    def test_is_upload_successful(self, scanner, mock_response):
        """Test upload success detection."""
        success_response = mock_response(
            status_code=200,
            body="File uploaded successfully"
        )
        
        assert scanner._is_upload_successful(success_response) is True
        
        fail_response = mock_response(
            status_code=400,
            body="Invalid file"
        )
        
        assert scanner._is_upload_successful(fail_response) is False
    
    def test_extract_upload_path(self, scanner):
        """Test upload path extraction."""
        body = 'File saved to: /uploads/shell.php'
        
        path = scanner._extract_upload_path(body)
        
        assert path is not None
        assert 'php' in path.lower()
    
    def test_analyze_response_successful_upload(self, scanner, mock_response):
        """Test detection of successful malicious upload."""
        response = mock_response(
            status_code=200,
            body="File uploaded successfully to /uploads/shell.php"
        )
        
        result = scanner.analyze_response(response, 'shell.php')
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.75
        assert 'succeeded' in result['evidence'].lower()
    
    def test_analyze_response_rejected_upload(self, scanner, mock_response):
        """Test detection of properly rejected upload."""
        response = mock_response(
            status_code=403,
            body="File type not allowed"
        )
        
        result = scanner.analyze_response(response, 'shell.php')
        
        assert result['detected'] is False
        assert 'properly rejected' in result['evidence'].lower()
    
    @pytest.mark.asyncio
    async def test_test_extension_bypass(self, scanner, mock_response):
        """Test extension bypass testing."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="File uploaded successfully"
        ))
        
        findings = await scanner._test_extension_bypass(
            'https://example.com/upload',
            'file',
            {}
        )
        
        assert isinstance(findings, list)
        if findings:
            assert findings[0].severity == FindingSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_test_content_type_bypass(self, scanner, mock_response):
        """Test content-type bypass testing."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="Upload successful"
        ))
        
        findings = await scanner._test_content_type_bypass(
            'https://example.com/upload',
            'file',
            {}
        )
        
        assert isinstance(findings, list)
    
    @pytest.mark.asyncio
    async def test_scan_full_workflow(self, scanner):
        """Test complete scan workflow."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=Mock(
            status_code=200,
            body="OK",
            headers={}
        ))
        
        findings = await scanner.scan(
            'https://example.com/upload',
            file_param='avatar'
        )
        
        assert isinstance(findings, list)
        assert scanner.stats['scans_performed'] > 0
    
    def test_get_file_upload_scanner_singleton(self):
        """Test global scanner instance."""
        scanner1 = get_file_upload_scanner()
        scanner2 = get_file_upload_scanner()
        
        assert scanner1 is scanner2