"""
Test suite for SQL Injection Scanner - V19.1
Tests payload generation, response analysis, and detection logic with WAF bypass
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from scanners.vulns.sqli import SQLInjectionScanner
from core.network import HTTPResponse
from domain.enums import FindingType, FindingSeverity


class TestSQLInjectionScanner:
    """Test SQLInjectionScanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner instance."""
        return SQLInjectionScanner()
    
    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        def _create(status_code=200, body="", response_time=0.1):
            response = Mock(spec=HTTPResponse)
            response.status_code = status_code
            response.body = body
            response.response_time = response_time
            response.headers = {}
            return response
        return _create
    
    # Initialization Tests
    
    def test_scanner_initialization(self, scanner):
        """Test scanner initializes correctly."""
        assert scanner.finding_type == FindingType.SQLI_ERROR
        assert len(scanner.error_patterns) > 0
        assert len(scanner.compiled_patterns) > 0
        assert scanner.payload_manager is not None
    
    def test_scanner_stats_initialization(self, scanner):
        """Test scanner statistics are initialized."""
        assert 'scans_performed' in scanner.stats
        assert 'payloads_generated' in scanner.stats
        assert 'findings_detected' in scanner.stats
        assert scanner.stats['scans_performed'] == 0
    
    # Payload Generation Tests
    
    def test_generate_payloads(self, scanner):
        """Test payload generation."""
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        # Check for common SQLi patterns
        assert any("'" in p for p in payloads)
        assert any("--" in p for p in payloads)
    
    def test_generate_payloads_with_technologies(self, scanner):
        """Test payload generation with specific technologies."""
        payloads = scanner.generate_payloads(technologies=['mysql'])
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
    
    # Response Analysis Tests
    
    def test_analyze_response_mysql_error(self, scanner, mock_response):
        """Test detection of MySQL error."""
        response = mock_response(
            status_code=200,
            body="MySQL error: You have an error in your SQL syntax"
        )
        
        result = scanner.analyze_response(response, "' OR 1=1--")
        
        assert result['detected'] is True
        assert result['confidence'] > 0.3
        assert 'MySQL' in result['evidence'] or 'mysql' in result['evidence'].lower()
        assert len(result['matched_patterns']) > 0
    
    def test_analyze_response_postgresql_error(self, scanner, mock_response):
        """Test detection of PostgreSQL error."""
        response = mock_response(
            status_code=500,
            body="PostgreSQL error: syntax error at or near"
        )
        
        result = scanner.analyze_response(response, "'; DROP TABLE--")
        
        assert result['detected'] is True
        assert result['confidence'] > 0.3
        assert 'postgresql' in result['evidence'].lower()
    
    def test_analyze_response_mssql_error(self, scanner, mock_response):
        """Test detection of MSSQL error."""
        response = mock_response(
            status_code=500,
            body="Microsoft SQL Server error: Unclosed quotation mark"
        )
        
        result = scanner.analyze_response(response, "' UNION SELECT")
        
        assert result['detected'] is True
        assert result['confidence'] > 0.3
    
    def test_analyze_response_oracle_error(self, scanner, mock_response):
        """Test detection of Oracle error."""
        response = mock_response(
            status_code=500,
            body="ORA-00933: SQL command not properly ended"
        )
        
        result = scanner.analyze_response(response, "' OR 1=1")
        
        assert result['detected'] is True
        assert result['confidence'] > 0.3
    
    def test_analyze_response_500_error(self, scanner, mock_response):
        """Test detection based on 500 status code."""
        response = mock_response(
            status_code=500,
            body="Internal Server Error"
        )
        
        result = scanner.analyze_response(response, "' OR 1=1--")
        
        # 500 alone should increase confidence
        assert result['confidence'] >= 0.2
    
    def test_analyze_response_time_based_sqli(self, scanner, mock_response):
        """Test detection of time-based SQLi."""
        response = mock_response(
            status_code=200,
            body="Loading...",
            response_time=5.5  # Delayed response
        )
        
        result = scanner.analyze_response(response, "'; WAITFOR DELAY '00:00:05'--")
        
        assert result['detected'] is True
        assert result['confidence'] > 0.4
        assert 'Time delay detected' in result['evidence']
    
    def test_analyze_response_no_detection(self, scanner, mock_response):
        """Test no detection on clean response."""
        response = mock_response(
            status_code=200,
            body="Normal response content"
        )
        
        result = scanner.analyze_response(response, "' OR 1=1--")
        
        assert result['detected'] is False
        assert result['confidence'] == 0.0
        assert 'No SQL injection indicators' in result['evidence']
    
    def test_analyze_response_multiple_patterns(self, scanner, mock_response):
        """Test detection with multiple error patterns."""
        response = mock_response(
            status_code=500,
            body="MySQL error: You have an error in your SQL syntax near 'OR 1=1'"
        )
        
        result = scanner.analyze_response(response, "' OR 1=1--")
        
        # Multiple indicators should increase confidence
        assert result['detected'] is True
        assert result['confidence'] > 0.5
        assert len(result['matched_patterns']) > 0
    
    def test_analyze_response_confidence_cap(self, scanner, mock_response):
        """Test confidence is capped at 0.95 for heuristic."""
        response = mock_response(
            status_code=500,
            body="MySQL error: syntax error SQL database error"
        )
        
        result = scanner.analyze_response(response, "'; WAITFOR DELAY '00:00:05'--")
        
        # Even with multiple strong indicators, cap at 0.95
        assert result['confidence'] <= 0.95
    
    # Scan Tests
    
    @pytest.mark.asyncio
    async def test_scan_basic(self, scanner):
        """Test basic scan execution."""
        with patch.object(scanner, 'execute_scan', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = []
            
            findings = await scanner.scan('https://example.com/page?id=1')
            
            assert mock_exec.called
            assert isinstance(findings, list)
            assert scanner.stats['payloads_generated'] > 0
    
    @pytest.mark.asyncio
    async def test_scan_with_parameter(self, scanner):
        """Test scan with specific parameter."""
        with patch.object(scanner, 'execute_scan', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = []
            
            await scanner.scan('https://example.com/api', parameter='search')
            
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert call_args[0] == 'https://example.com/api'
            assert call_args[1] == 'search'
    
    # Edge Cases
    
    def test_analyze_response_empty_body(self, scanner, mock_response):
        """Test analysis with empty response body."""
        response = mock_response(status_code=200, body="")
        
        result = scanner.analyze_response(response, "' OR 1=1--")
        
        assert result['detected'] is False
        assert isinstance(result, dict)
    
    def test_analyze_response_large_body(self, scanner, mock_response):
        """Test analysis with large response body."""
        large_body = "x" * 1000000 + "MySQL error" + "y" * 1000000
        response = mock_response(status_code=200, body=large_body)
        
        result = scanner.analyze_response(response, "' OR 1=1--")
        
        # Should still detect even in large body
        assert result['detected'] is True
    
    def test_analyze_response_unicode_characters(self, scanner, mock_response):
        """Test analysis with unicode in response."""
        response = mock_response(
            status_code=200,
            body="MySQL错误: You have an error in your SQL syntax"
        )
        
        result = scanner.analyze_response(response, "' OR 1=1--")
        
        assert result['detected'] is True
    
    def test_error_patterns_case_insensitive(self, scanner, mock_response):
        """Test error patterns are case insensitive."""
        response_lower = mock_response(
            status_code=200,
            body="mysql error in your sql syntax"
        )
        response_upper = mock_response(
            status_code=200,
            body="MYSQL ERROR IN YOUR SQL SYNTAX"
        )
        
        result_lower = scanner.analyze_response(response_lower, "' OR 1=1--")
        result_upper = scanner.analyze_response(response_upper, "' OR 1=1--")
        
        assert result_lower['detected'] is True
        assert result_upper['detected'] is True
    
    # Statistics Tests
    
    @pytest.mark.asyncio
    async def test_stats_update_on_scan(self, scanner):
        """Test statistics are updated during scan."""
        initial_scans = scanner.stats['scans_performed']
        
        with patch.object(scanner, 'execute_scan', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = []
            await scanner.scan('https://example.com/test')
        
        # Scans should not be incremented (execute_scan does that)
        # But payloads should be generated
        assert scanner.stats['payloads_generated'] > 0
    
    def test_response_analysis_fields(self, scanner, mock_response):
        """Test response analysis includes all required fields."""
        response = mock_response(status_code=200, body="test")
        
        result = scanner.analyze_response(response, "test")
        
        assert 'detected' in result
        assert 'confidence' in result
        assert 'evidence' in result
        assert 'matched_patterns' in result
        assert 'response_analysis' in result
        assert 'status_code' in result['response_analysis']
        assert 'response_time' in result['response_analysis']
        assert 'body_length' in result['response_analysis']


class TestSQLInjectionScannerIntegration:
    """Integration tests for SQLInjectionScanner."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner instance."""
        return SQLInjectionScanner()
    
    @pytest.mark.asyncio
    async def test_full_scan_workflow(self, scanner):
        """Test complete scan workflow."""
        # Mock network client
        with patch.object(scanner, 'network_client') as mock_client:
            mock_response = Mock(spec=HTTPResponse)
            mock_response.status_code = 500
            mock_response.body = "MySQL error: syntax error"
            mock_response.response_time = 0.1
            mock_response.headers = {}
            
            mock_client.request = AsyncMock(return_value=mock_response)
            
            findings = await scanner.scan('https://example.com/test?id=1')
            
            # Should have made requests
            assert mock_client.request.called
    
    def test_pattern_compilation(self, scanner):
        """Test all patterns compile correctly."""
        assert len(scanner.error_patterns) == len(scanner.compiled_patterns)
        
        for pattern in scanner.compiled_patterns:
            assert pattern.pattern is not None
            # Test pattern can match
            test_string = "mysql error in sql syntax"
            try:
                pattern.search(test_string)
            except Exception as e:
                pytest.fail(f"Pattern compilation failed: {e}")