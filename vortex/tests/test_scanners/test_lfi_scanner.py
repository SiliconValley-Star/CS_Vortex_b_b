"""
Test suite for LFI (Local File Inclusion) Scanner
Tests file content detection and path traversal
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from scanners.vulns.lfi import LFIScanner
from core.network import HTTPResponse
from domain.enums import FindingType


class TestLFIScanner:
    """Test LFI Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return LFIScanner()
    
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
        assert scanner.finding_type == FindingType.LFI
        assert len(scanner.file_patterns) > 0
        assert len(scanner.compiled_patterns) > 0
    
    def test_generate_payloads(self, scanner):
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        assert any('../' in p for p in payloads)
        assert any('etc/passwd' in p for p in payloads)
    
    def test_analyze_response_unix_passwd(self, scanner, mock_response):
        """Test detection of /etc/passwd content."""
        response = mock_response(
            status_code=200,
            body="root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin"
        )
        
        result = scanner.analyze_response(response, '../../../../etc/passwd')
        
        assert result['detected'] is True
        assert result['confidence'] > 0.4
        assert 'passwd file structure' in result['evidence'].lower()
    
    def test_analyze_response_windows_ini(self, scanner, mock_response):
        """Test detection of Windows INI files."""
        response = mock_response(
            status_code=200,
            body="[extensions]\n[fonts]\n[mci]"
        )
        
        result = scanner.analyze_response(response, '..\\..\\windows\\win.ini')
        
        assert result['detected'] is True
        assert 'Windows INI' in result['evidence'] or 'ini' in result['evidence'].lower()
    
    def test_analyze_response_php_file(self, scanner, mock_response):
        """Test detection of PHP file content."""
        response = mock_response(
            status_code=200,
            body="<?php\n$_GET['user']\n$_POST['password']"
        )
        
        result = scanner.analyze_response(response, '../config.php')
        
        assert result['detected'] is True
        assert result['confidence'] > 0
    
    def test_analyze_response_config_file(self, scanner, mock_response):
        """Test detection of configuration file."""
        response = mock_response(
            status_code=200,
            body="mysql_host=localhost\ndatabase_password=secret123\nDB_PASSWORD=admin"
        )
        
        result = scanner.analyze_response(response, '../config/database.ini')
        
        assert result['detected'] is True
    
    def test_analyze_response_no_detection(self, scanner, mock_response):
        """Test no detection on normal response."""
        response = mock_response(
            status_code=200,
            body="Normal page content"
        )
        
        result = scanner.analyze_response(response, '../../../../etc/passwd')
        
        assert result['detected'] is False
    
    def test_analyze_response_long_content(self, scanner, mock_response):
        """Test confidence boost for long responses."""
        response = mock_response(
            status_code=200,
            body="x" * 2000  # Long content might indicate file read
        )
        
        result = scanner.analyze_response(response, '../large_file.txt')
        
        assert 'Long response' in result['evidence']
    
    @pytest.mark.asyncio
    async def test_scan_basic(self, scanner):
        with patch.object(scanner, 'execute_scan', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = []
            
            findings = await scanner.scan('https://example.com/page?file=test.txt')
            
            assert mock_exec.called
            assert scanner.stats['payloads_generated'] > 0