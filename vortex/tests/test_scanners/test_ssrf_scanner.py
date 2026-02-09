"""
Test suite for SSRF (Server-Side Request Forgery) Scanner
Tests internal network detection and SSRF indicators
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from scanners.vulns.ssrf import SSRFScanner
from core.network import HTTPResponse
from domain.enums import FindingType


class TestSSRFScanner:
    """Test SSRF Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return SSRFScanner()
    
    @pytest.fixture
    def mock_response(self):
        def _create(status_code=200, body="", response_time=0.1):
            response = Mock(spec=HTTPResponse)
            response.status_code = status_code
            response.body = body
            response.response_time = response_time
            response.headers = {}
            return response
        return _create
    
    def test_scanner_initialization(self, scanner):
        assert scanner.finding_type == FindingType.SSRF
        assert len(scanner.internal_indicators) > 0
        assert len(scanner.compiled_patterns) > 0
    
    def test_generate_payloads(self, scanner):
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        assert any('localhost' in p for p in payloads)
        assert any('127.0.0.1' in p for p in payloads)
    
    def test_analyze_response_localhost_indicator(self, scanner, mock_response):
        """Test detection of localhost in response."""
        response = mock_response(
            status_code=200,
            body="Connected to localhost:8080"
        )
        
        result = scanner.analyze_response(response, 'http://localhost')
        
        assert result['detected'] is True
        assert result['confidence'] > 0.4
        assert 'localhost' in result['evidence'].lower()
    
    def test_analyze_response_private_ip(self, scanner, mock_response):
        """Test detection of private IP addresses."""
        response = mock_response(
            status_code=200,
            body="Connecting to 192.168.1.1..."
        )
        
        result = scanner.analyze_response(response, 'http://192.168.1.1')
        
        assert result['detected'] is True
        assert '192.168' in result['evidence']
    
    def test_analyze_response_aws_metadata(self, scanner, mock_response):
        """Test detection of AWS metadata service."""
        response = mock_response(
            status_code=200,
            body='{"ami-id": "ami-12345", "instance-id": "i-67890"}'
        )
        
        result = scanner.analyze_response(response, 'http://169.254.169.254/latest/meta-data/')
        
        assert result['detected'] is True
        assert result['confidence'] > 0.5
        assert 'metadata' in result['evidence'].lower()
    
    def test_analyze_response_successful_internal_access(self, scanner, mock_response):
        """Test detection of successful internal network access."""
        response = mock_response(
            status_code=200,
            body="Internal server response"
        )
        
        result = scanner.analyze_response(response, 'http://10.0.0.1/admin')
        
        assert result['detected'] is True
        assert result['confidence'] > 0.3
    
    def test_analyze_response_fast_response_time(self, scanner, mock_response):
        """Test detection based on fast response time (indicates local access)."""
        response = mock_response(
            status_code=200,
            body="OK",
            response_time=0.001  # Very fast
        )
        
        result = scanner.analyze_response(response, 'http://127.0.0.1')
        
        # Fast response to localhost should boost confidence
        assert result['confidence'] > 0.0
        assert 'Fast response time' in result['evidence']
    
    def test_analyze_response_no_detection(self, scanner, mock_response):
        """Test no detection on external URL."""
        response = mock_response(
            status_code=200,
            body="Normal external content"
        )
        
        result = scanner.analyze_response(response, 'https://example.com')
        
        assert result['detected'] is False
    
    @pytest.mark.asyncio
    async def test_scan_basic(self, scanner):
        with patch.object(scanner, 'execute_scan', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = []
            
            findings = await scanner.scan('https://example.com/fetch?url=test')
            
            assert mock_exec.called