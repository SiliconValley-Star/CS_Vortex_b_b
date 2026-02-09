"""
Test suite for XXE (XML External Entity) Scanner
Tests XXE payloads, file read detection, and DTD expansion
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from scanners.vulns.xxe import XXEScanner, get_xxe_scanner
from core.network import HTTPResponse
from domain.enums import FindingType, FindingSeverity


class TestXXEScanner:
    """Test XXE Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return XXEScanner()
    
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
        assert scanner.finding_type == FindingType.XXE
        assert len(XXEScanner.CLASSIC_XXE_PAYLOADS) > 0
        assert len(XXEScanner.FILE_READ_INDICATORS) > 0
    
    def test_xxe_payloads_defined(self, scanner):
        """Test XXE payloads are properly defined."""
        payloads = XXEScanner.CLASSIC_XXE_PAYLOADS
        
        # Should include file read payloads
        assert any('/etc/passwd' in p for p in payloads)
        assert any('win.ini' in p for p in payloads)
        # Should have proper XML structure
        assert any('<!DOCTYPE' in p for p in payloads)
        assert any('<!ENTITY' in p for p in payloads)
    
    def test_generate_payloads(self, scanner):
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        # Should include various XXE types
        assert any('SYSTEM' in p for p in payloads)
        assert any('<!ENTITY' in p for p in payloads)
    
    def test_analyze_response_unix_file_read(self, scanner, mock_response):
        """Test detection of /etc/passwd file read."""
        response = mock_response(
            status_code=200,
            body="root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon"
        )
        
        result = scanner.analyze_response(response, '<!DOCTYPE foo>')
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.90
        assert 'File content leaked' in result['evidence']
    
    def test_analyze_response_windows_file_read(self, scanner, mock_response):
        """Test detection of Windows INI file read."""
        response = mock_response(
            status_code=200,
            body="[extensions]\n[fonts]\n[mci extensions]"
        )
        
        result = scanner.analyze_response(response, '<!DOCTYPE foo>')
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.90
    
    def test_analyze_response_xml_error(self, scanner, mock_response):
        """Test detection of XML parsing errors."""
        response = mock_response(
            status_code=500,
            body="XML parse error: External entity not allowed"
        )
        
        result = scanner.analyze_response(response, '<!DOCTYPE foo>')
        
        # XML error alone should give some confidence
        assert result['confidence'] > 0.0
        assert 'XML processing error' in result['evidence']
    
    def test_analyze_response_no_detection(self, scanner, mock_response):
        """Test no detection on safe response."""
        response = mock_response(
            status_code=200,
            body="<result>Normal XML response</result>"
        )
        
        result = scanner.analyze_response(response, '<!DOCTYPE foo>')
        
        assert result['detected'] is False
    
    def test_extract_evidence(self, scanner):
        """Test evidence extraction."""
        body = "prefix " + "root:x:0:0:root" + " suffix " * 50
        
        evidence = scanner._extract_evidence(body, "root:")
        
        assert isinstance(evidence, str)
        assert len(evidence) <= 150
        assert 'root:' in evidence
    
    @pytest.mark.asyncio
    async def test_test_classic_xxe(self, scanner, mock_response):
        """Test classic XXE file read testing."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="root:x:0:0:root:/root:/bin/bash"
        ))
        
        findings = await scanner._test_classic_xxe(
            'https://example.com/xml',
            'application/xml'
        )
        
        assert len(findings) > 0
        assert findings[0].severity == FindingSeverity.CRITICAL
        assert 'file system access' in findings[0].description.lower()
    
    @pytest.mark.asyncio
    async def test_test_soap_xxe(self, scanner, mock_response):
        """Test SOAP XXE testing."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="root:x:0:0:"
        ))
        
        # Only tests SOAP-like URLs
        findings = await scanner._test_soap_xxe('https://example.com/soap/service')
        
        assert isinstance(findings, list)
    
    @pytest.mark.asyncio
    async def test_test_svg_xxe(self, scanner, mock_response):
        """Test SVG XXE testing."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="root:x:0:0:"
        ))
        
        # Only tests upload-like URLs
        findings = await scanner._test_svg_xxe('https://example.com/upload')
        
        assert isinstance(findings, list)
    
    @pytest.mark.asyncio
    async def test_test_dtd_expansion(self, scanner, mock_response):
        """Test DTD entity expansion (Billion Laughs)."""
        scanner.network_client = Mock()
        
        # Simulate slow response (DoS indicator)
        slow_response = mock_response(status_code=503, body="Service Unavailable")
        slow_response.response_time = 5.0
        
        scanner.network_client.request = AsyncMock(return_value=slow_response)
        
        findings = await scanner._test_dtd_expansion(
            'https://example.com/xml',
            'application/xml'
        )
        
        assert len(findings) > 0
        assert 'expansion' in findings[0].description.lower()
    
    @pytest.mark.asyncio
    async def test_scan_full_workflow(self, scanner):
        """Test complete scan workflow."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=Mock(
            status_code=200,
            body="<response>OK</response>",
            headers={}
        ))
        
        findings = await scanner.scan(
            'https://example.com/api/xml',
            content_type='application/xml'
        )
        
        assert isinstance(findings, list)
        assert scanner.stats['scans_performed'] > 0
    
    def test_get_xxe_scanner_singleton(self):
        """Test global scanner instance."""
        scanner1 = get_xxe_scanner()
        scanner2 = get_xxe_scanner()
        
        assert scanner1 is scanner2