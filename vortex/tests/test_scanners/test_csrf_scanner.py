"""
Test suite for CSRF (Cross-Site Request Forgery) Scanner - V19.1
Tests token validation, SameSite cookies, and origin validation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from scanners.vulns.csrf import CSRFScanner, get_csrf_scanner
from core.network import HTTPResponse
from domain.enums import FindingType, FindingSeverity
from domain.models import AssessmentResult


class TestCSRFScanner:
    """Test CSRF Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return CSRFScanner()
    
    @pytest.fixture
    def mock_response(self):
        def _create(status_code=200, body="", headers=None):
            response = Mock(spec=HTTPResponse)
            response.status_code = status_code
            response.body = body
            response.headers = headers or {}
            response.response_time = 0.1
            return response
        return _create
    
    def test_scanner_initialization(self, scanner):
        assert scanner.finding_type == FindingType.CSRF
        assert len(CSRFScanner.CSRF_TOKEN_NAMES) > 0
        assert len(CSRFScanner.STATE_CHANGING_METHODS) > 0
    
    def test_csrf_token_names_defined(self, scanner):
        """Test CSRF token names are properly defined."""
        expected_names = ['csrf_token', 'csrftoken', '_csrf', 'authenticity_token']
        
        for name in expected_names:
            assert name in CSRFScanner.CSRF_TOKEN_NAMES
    
    def test_generate_payloads(self, scanner):
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        # Should include invalid tokens and malicious origins
        assert any('evil.com' in p for p in payloads)
        assert any('invalid' in p for p in payloads)
    
    def test_analyze_response_successful_attack(self, scanner, mock_response):
        """Test detection when state-changing request succeeds."""
        response = mock_response(status_code=200, body="Success")
        
        result = scanner.analyze_response(response, 'test_payload')
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.70
        assert 'CSRF protection' in result['evidence']
    
    def test_analyze_response_csrf_error(self, scanner, mock_response):
        """Test when CSRF protection properly rejects request."""
        response = mock_response(
            status_code=403,
            body="CSRF token verification failed"
        )
        
        result = scanner.analyze_response(response, 'invalid_token')
        
        # Should not detect vulnerability if properly rejected
        assert 'properly rejected' in result['evidence'].lower()
    
    @pytest.mark.asyncio
    async def test_test_csrf_token_presence_missing(self, scanner, mock_response):
        """Test detection of missing CSRF token in POST request."""
        scanner.network_client = Mock()
        
        # POST without CSRF token
        finding = await scanner._test_csrf_token_presence(
            'https://example.com/submit',
            'POST',
            {'username': 'test', 'password': 'test'}
        )
        
        assert finding is not None
        assert finding.severity == FindingSeverity.HIGH
        assert 'No CSRF token found' in finding.evidence
    
    @pytest.mark.asyncio
    async def test_test_csrf_token_presence_present(self, scanner):
        """Test no finding when CSRF token is present."""
        finding = await scanner._test_csrf_token_presence(
            'https://example.com/submit',
            'POST',
            {'username': 'test', 'csrf_token': 'valid_token_123'}
        )
        
        assert finding is None
    
    @pytest.mark.asyncio
    async def test_test_samesite_cookies_missing(self, scanner, mock_response):
        """Test detection of missing SameSite attribute."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="OK",
            headers={'set-cookie': 'session=abc123; HttpOnly'}
        ))
        
        finding = await scanner._test_samesite_cookies(
            'https://example.com',
            {'session': 'test'}
        )
        
        assert finding is not None
        assert 'SameSite' in finding.evidence
    
    @pytest.mark.asyncio
    async def test_test_origin_validation_bypass(self, scanner, mock_response):
        """Test detection of origin validation bypass."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="Success"
        ))
        
        finding = await scanner._test_origin_validation(
            'https://example.com/api/update',
            'POST',
            {'data': 'test'}
        )
        
        assert finding is not None
        assert finding.severity == FindingSeverity.HIGH
        assert 'arbitrary origins' in finding.description
    
    @pytest.mark.asyncio
    async def test_test_token_validation_weak(self, scanner, mock_response):
        """Test detection of weak token validation."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body="Update successful"
        ))
        
        finding = await scanner._test_token_validation(
            'https://example.com/update',
            'POST',
            {'csrf_token': 'valid_token', 'data': 'test'}
        )
        
        assert finding is not None
        assert 'invalid CSRF token' in finding.evidence
    
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
            'https://example.com/submit',
            method='POST',
            params={'username': 'test'}
        )
        
        assert isinstance(findings, list)
        assert scanner.stats['scans_performed'] > 0
    
    def test_get_csrf_scanner_singleton(self):
        """Test global scanner instance."""
        scanner1 = get_csrf_scanner()
        scanner2 = get_csrf_scanner()
        
        assert scanner1 is scanner2