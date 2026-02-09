"""
Test suite for SSTI (Server-Side Template Injection) Scanner
Tests template engine detection and expression injection
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from scanners.vulns.ssti import SSTIScanner, get_ssti_scanner
from core.network import HTTPResponse
from domain.enums import FindingType, FindingSeverity


class TestSSTIScanner:
    """Test SSTI Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return SSTIScanner()
    
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
        assert scanner.finding_type == FindingType.SSTI
        assert len(SSTIScanner.TEMPLATE_ENGINES) > 0
        assert len(SSTIScanner.MATH_EXPRESSIONS) > 0
    
    def test_template_engines_defined(self, scanner):
        """Test template engine configurations."""
        engines = ['jinja2', 'twig', 'freemarker', 'velocity', 'erb', 'smarty']
        
        for engine in engines:
            assert engine in SSTIScanner.TEMPLATE_ENGINES
            config = SSTIScanner.TEMPLATE_ENGINES[engine]
            assert 'math' in config
            assert 'detection' in config
            assert 'rce' in config
    
    def test_generate_payloads(self, scanner):
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        # Should include math expressions and engine-specific payloads
        assert any('7*7' in p for p in payloads)
        assert any('{{' in p for p in payloads)
    
    def test_analyze_response_math_evaluation(self, scanner, mock_response):
        """Test detection of mathematical expression evaluation."""
        response = mock_response(
            status_code=200,
            body="Result: 49"
        )
        
        result = scanner.analyze_response(response, '{{7*7}}')
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.85
        assert 'Mathematical expression evaluated' in result['evidence']
    
    def test_analyze_response_string_multiplication(self, scanner, mock_response):
        """Test detection of string multiplication."""
        response = mock_response(
            status_code=200,
            body="Output: 7777777"
        )
        
        result = scanner.analyze_response(response, "{{7*'7'}}")
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.85
    
    def test_analyze_response_template_error(self, scanner, mock_response):
        """Test detection of template errors."""
        response = mock_response(
            status_code=500,
            body="Jinja2 TemplateSyntaxError: unexpected char"
        )
        
        result = scanner.analyze_response(response, '{{7*7')
        
        assert result['detected'] is True
        assert result['confidence'] > 0.0
        assert 'Template error' in result['evidence']
    
    def test_analyze_response_rce_indicators(self, scanner, mock_response):
        """Test detection of RCE indicators."""
        response = mock_response(
            status_code=200,
            body="uid=0(root) gid=0(root) groups=0(root)"
        )
        
        result = scanner.analyze_response(response, '{{config.__class__.__init__.__globals__}}')
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.95
        assert 'RCE indicators' in result['evidence']
    
    def test_analyze_response_no_detection(self, scanner, mock_response):
        """Test no detection on safe response."""
        response = mock_response(
            status_code=200,
            body="Normal content"
        )
        
        result = scanner.analyze_response(response, '{{7*7}}')
        
        assert result['detected'] is False
    
    def test_detect_engine_from_syntax(self, scanner):
        """Test template engine detection from syntax."""
        assert 'jinja' in scanner._detect_engine_from_syntax('{{7*7}}').lower()
        assert 'freemarker' in scanner._detect_engine_from_syntax('${7*7}').lower()
        assert 'erb' in scanner._detect_engine_from_syntax('<%= 7*7 %>').lower()
        assert 'smarty' in scanner._detect_engine_from_syntax('{7*7}').lower()
    
    def test_has_template_error(self, scanner):
        """Test template error detection."""
        assert scanner._has_template_error("Jinja2 TemplateSyntaxError")
        assert scanner._has_template_error("Twig syntax error")
        assert scanner._has_template_error("UndefinedError: variable not found")
        assert not scanner._has_template_error("Normal response")
    
    @pytest.mark.asyncio
    async def test_test_math_expressions(self, scanner, mock_response):
        """Test math expression injection testing."""
        scanner.network_client = Mock()
        
        # Baseline response
        baseline = mock_response(status_code=200, body="Input: test")
        # Response with evaluation
        eval_response = mock_response(status_code=200, body="Result: 49")
        
        scanner.network_client.request = AsyncMock(side_effect=[baseline, eval_response])
        
        findings = await scanner._test_math_expressions(
            'https://example.com/render',
            {'template': 'test'},
            {},
            'GET'
        )
        
        assert len(findings) > 0
        assert findings[0].severity == FindingSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_scan_full_workflow(self, scanner):
        """Test complete scan workflow."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=Mock(
            status_code=200,
            body="49",
            headers={}
        ))
        
        findings = await scanner.scan(
            'https://example.com/template',
            params={'input': 'test'}
        )
        
        assert isinstance(findings, list)
        assert scanner.stats['scans_performed'] > 0
    
    def test_get_ssti_scanner_singleton(self):
        """Test global scanner instance."""
        scanner1 = get_ssti_scanner()
        scanner2 = get_ssti_scanner()
        
        assert scanner1 is scanner2