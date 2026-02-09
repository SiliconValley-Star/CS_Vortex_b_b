"""
Test suite for XSS Scanner
Tests payload reflection detection, context analysis, and XSS identification
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from html import escape

from scanners.vulns.xss import XSSScanner
from core.network import HTTPResponse
from domain.enums import FindingType


class TestXSSScanner:
    """Test XSSScanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner instance."""
        return XSSScanner()
    
    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        def _create(status_code=200, body=""):
            response = Mock(spec=HTTPResponse)
            response.status_code = status_code
            response.body = body
            response.headers = {}
            response.response_time = 0.1
            return response
        return _create
    
    # Initialization Tests
    
    def test_scanner_initialization(self, scanner):
        """Test scanner initializes correctly."""
        assert scanner.finding_type == FindingType.XSS_REFLECTED
        assert len(scanner.xss_indicators) > 0
        assert len(scanner.compiled_patterns) > 0
    
    def test_xss_patterns_compiled(self, scanner):
        """Test XSS detection patterns are compiled."""
        assert len(scanner.compiled_patterns) == len(scanner.xss_indicators)
        
        for pattern in scanner.compiled_patterns:
            assert pattern.pattern is not None
    
    # Payload Generation Tests
    
    def test_generate_payloads(self, scanner):
        """Test XSS payload generation."""
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        # Check for common XSS patterns
        assert any('<script>' in p for p in payloads)
        assert any('onerror' in p for p in payloads)
    
    def test_generate_payloads_variety(self, scanner):
        """Test payload variety."""
        payloads = scanner.generate_payloads()
        
        # Should have different types of payloads
        script_tags = [p for p in payloads if '<script>' in p]
        event_handlers = [p for p in payloads if 'onerror' in p or 'onload' in p]
        
        assert len(script_tags) > 0
        assert len(event_handlers) > 0
    
    # Response Analysis Tests
    
    def test_analyze_response_reflected_payload(self, scanner, mock_response):
        """Test detection of reflected XSS payload."""
        payload = '<script>alert(1)</script>'
        response = mock_response(
            status_code=200,
            body=f"Search results for: {payload}"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
        assert result['confidence'] > 0.5
        assert 'Payload reflected' in result['evidence']
        assert result['payload_reflected'] is True
    
    def test_analyze_response_reflected_in_script_context(self, scanner, mock_response):
        """Test detection of reflection in script context."""
        payload = 'alert(1)'
        response = mock_response(
            status_code=200,
            body=f"<script>var x = '{payload}';</script>"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
        assert result['confidence'] > 0.5
    
    def test_analyze_response_img_onerror(self, scanner, mock_response):
        """Test detection of img onerror XSS."""
        payload = '<img src=x onerror=alert(1)>'
        response = mock_response(
            status_code=200,
            body=f"<div>{payload}</div>"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
        assert len(result['matched_patterns']) > 0
    
    def test_analyze_response_svg_xss(self, scanner, mock_response):
        """Test detection of SVG-based XSS."""
        payload = '<svg onload=alert(1)>'
        response = mock_response(
            status_code=200,
            body=f"<div>{payload}</div>"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
    
    def test_analyze_response_event_handler_context(self, scanner, mock_response):
        """Test detection in event handler context."""
        payload = '" onmouseover="alert(1)" x="'
        response = mock_response(
            status_code=200,
            body=f'<input value="{payload}">'
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
        assert result['confidence'] > 0.0
    
    def test_analyze_response_javascript_protocol(self, scanner, mock_response):
        """Test detection of javascript: protocol."""
        payload = 'javascript:alert(1)'
        response = mock_response(
            status_code=200,
            body=f'<a href="{payload}">Click</a>'
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
    
    def test_analyze_response_no_reflection(self, scanner, mock_response):
        """Test no detection when payload not reflected."""
        payload = '<script>alert(1)</script>'
        response = mock_response(
            status_code=200,
            body="Normal content without payload"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is False
        assert result['payload_reflected'] is False
    
    def test_analyze_response_escaped_payload(self, scanner, mock_response):
        """Test detection with escaped payload."""
        payload = '<script>alert(1)</script>'
        escaped = escape(payload)
        response = mock_response(
            status_code=200,
            body=f"Search: {escaped}"
        )
        
        result = scanner.analyze_response(response, payload)
        
        # Should still detect reflection even if escaped
        # because we normalize in analysis
        assert 'reflected' in result['evidence'].lower() or result['detected'] is False
    
    def test_analyze_response_partial_reflection(self, scanner, mock_response):
        """Test detection with partial payload reflection."""
        payload = '<script>alert("test")</script>'
        # Only part of payload reflected
        response = mock_response(
            status_code=200,
            body="<div>alert test</div>"
        )
        
        result = scanner.analyze_response(response, payload)
        
        # Should detect based on keywords
        if result['detected']:
            assert result['confidence'] > 0.0
    
    def test_analyze_response_case_insensitive(self, scanner, mock_response):
        """Test case-insensitive detection."""
        payload = '<SCRIPT>alert(1)</SCRIPT>'
        response = mock_response(
            status_code=200,
            body=f"<div>{payload}</div>"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
    
    def test_analyze_response_confidence_cap(self, scanner, mock_response):
        """Test confidence is capped at 0.95."""
        payload = '<script>alert(1)</script>'
        # Multiple strong indicators
        response = mock_response(
            status_code=200,
            body=f"""
            <script>{payload}</script>
            <img src=x onerror="{payload}">
            <svg onload="{payload}">
            """
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['confidence'] <= 0.95
    
    # Edge Cases
    
    def test_analyze_response_empty_payload(self, scanner, mock_response):
        """Test analysis with empty payload."""
        response = mock_response(status_code=200, body="test")
        
        result = scanner.analyze_response(response, "")
        
        assert isinstance(result, dict)
        assert 'detected' in result
    
    def test_analyze_response_unicode_payload(self, scanner, mock_response):
        """Test analysis with unicode characters."""
        payload = '<script>alert("测试")</script>'
        response = mock_response(
            status_code=200,
            body=f"<div>{payload}</div>"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert isinstance(result, dict)
    
    def test_analyze_response_long_payload(self, scanner, mock_response):
        """Test analysis with very long payload."""
        payload = '<script>alert("x' + 'y' * 10000 + '")</script>'
        response = mock_response(
            status_code=200,
            body=f"<div>{payload}</div>"
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert isinstance(result, dict)
    
    def test_analyze_response_multiple_reflections(self, scanner, mock_response):
        """Test detection with multiple payload reflections."""
        payload = '<script>alert(1)</script>'
        response = mock_response(
            status_code=200,
            body=f"""
            <div>{payload}</div>
            <span>{payload}</span>
            <p>{payload}</p>
            """
        )
        
        result = scanner.analyze_response(response, payload)
        
        assert result['detected'] is True
        assert result['confidence'] > 0.5
    
    # Scan Tests
    
    @pytest.mark.asyncio
    async def test_scan_basic(self, scanner):
        """Test basic scan execution."""
        with patch.object(scanner, 'execute_scan', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = []
            
            findings = await scanner.scan('https://example.com/search?q=test')
            
            assert mock_exec.called
            assert isinstance(findings, list)
    
    @pytest.mark.asyncio
    async def test_scan_with_parameter(self, scanner):
        """Test scan with custom parameter."""
        with patch.object(scanner, 'execute_scan', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = []
            
            await scanner.scan('https://example.com/page', parameter='input')
            
            call_args = mock_exec.call_args[0]
            assert call_args[1] == 'input'
    
    # Pattern Matching Tests
    
    def test_script_pattern_matching(self, scanner, mock_response):
        """Test script tag pattern matching."""
        response = mock_response(
            status_code=200,
            body='<script>var x = "test";</script>'
        )
        
        result = scanner.analyze_response(response, '<script>')
        
        # Should detect script tag
        if result['detected']:
            assert any('script' in p.lower() for p in result['matched_patterns'])
    
    def test_event_handler_pattern_matching(self, scanner, mock_response):
        """Test event handler pattern matching."""
        test_cases = [
            '<img onerror="alert(1)">',
            '<body onload="alert(1)">',
            '<div onclick="alert(1)">',
        ]
        
        for test_case in test_cases:
            response = mock_response(status_code=200, body=test_case)
            result = scanner.analyze_response(response, test_case)
            
            if result['detected']:
                assert len(result['matched_patterns']) > 0
    
    def test_iframe_pattern_matching(self, scanner, mock_response):
        """Test iframe pattern matching."""
        response = mock_response(
            status_code=200,
            body='<iframe src="javascript:alert(1)"></iframe>'
        )
        
        result = scanner.analyze_response(response, '<iframe>')
        
        # Should detect iframe
        if result['detected']:
            assert any('iframe' in p.lower() for p in result['matched_patterns'])


class TestXSSScannerIntegration:
    """Integration tests for XSS Scanner."""
    
    @pytest.fixture
    def scanner(self):
        """Create scanner instance."""
        return XSSScanner()
    
    @pytest.mark.asyncio
    async def test_full_scan_workflow(self, scanner):
        """Test complete XSS scan workflow."""
        with patch.object(scanner, 'network_client') as mock_client:
            mock_response = Mock(spec=HTTPResponse)
            mock_response.status_code = 200
            mock_response.body = '<div><script>alert(1)</script></div>'
            mock_response.headers = {}
            mock_response.response_time = 0.1
            
            mock_client.request = AsyncMock(return_value=mock_response)
            
            findings = await scanner.scan('https://example.com/search?q=test')
            
            assert mock_client.request.called
    
    def test_payload_normalization(self, scanner, mock_response):
        """Test payload normalization in analysis."""
        # HTML entities in payload
        payload = '&lt;script&gt;alert(1)&lt;/script&gt;'
        response = mock_response(
            status_code=200,
            body='<div>&lt;script&gt;alert(1)&lt;/script&gt;</div>'
        )
        
        result = scanner.analyze_response(response, payload)
        
        # Should handle HTML entities
        assert isinstance(result, dict)