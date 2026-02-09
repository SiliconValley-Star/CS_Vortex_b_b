"""
Test suite for JWT Security Scanner
Tests JWT algorithm confusion, weak secrets, and token manipulation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
import base64

from scanners.api.jwt_scanner import JWTScanner, get_jwt_scanner
from core.network import HTTPResponse
from domain.enums import FindingType, FindingSeverity


class TestJWTScanner:
    """Test JWT Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return JWTScanner()
    
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
    
    @pytest.fixture
    def valid_jwt(self):
        """Create a valid JWT token for testing."""
        header = {'alg': 'HS256', 'typ': 'JWT'}
        payload = {'sub': '1234567890', 'name': 'Test User', 'iat': 1516239022}
        
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
        signature = 'signature'
        
        return f"{header_b64}.{payload_b64}.{signature}"
    
    def test_scanner_initialization(self, scanner):
        assert scanner.finding_type == FindingType.AUTH_BYPASS
        assert len(JWTScanner.WEAK_SECRETS) > 0
        assert len(JWTScanner.ALGORITHM_CONFUSION) > 0
    
    def test_weak_secrets_defined(self, scanner):
        """Test weak secret list."""
        weak_secrets = JWTScanner.WEAK_SECRETS
        
        assert 'secret' in weak_secrets
        assert 'password' in weak_secrets
        assert '123456' in weak_secrets
    
    def test_generate_payloads(self, scanner):
        payloads = scanner.generate_payloads()
        
        assert isinstance(payloads, list)
        assert len(payloads) > 0
        # Should include algorithm confusion
        assert 'none' in payloads or 'None' in payloads
        # Should include weak secrets
        assert 'secret' in payloads
    
    def test_extract_jwt_from_headers(self, scanner):
        """Test JWT extraction from Authorization header."""
        headers = {'Authorization': 'Bearer test.token.here'}
        
        token = scanner._extract_jwt_from_headers(headers)
        
        assert token == 'test.token.here'
    
    def test_extract_jwt_from_other_headers(self, scanner):
        """Test JWT extraction from alternative headers."""
        headers = {'X-Auth-Token': 'test.token.here'}
        
        token = scanner._extract_jwt_from_headers(headers)
        
        assert token == 'test.token.here'
    
    def test_parse_jwt_valid(self, scanner, valid_jwt):
        """Test JWT parsing with valid token."""
        result = scanner._parse_jwt(valid_jwt)
        
        assert result is not None
        assert 'header' in result
        assert 'payload' in result
        assert 'signature' in result
        assert result['header']['alg'] == 'HS256'
    
    def test_parse_jwt_invalid(self, scanner):
        """Test JWT parsing with invalid token."""
        result = scanner._parse_jwt('invalid.token')
        
        assert result is None
    
    def test_base64_encode_decode(self, scanner):
        """Test base64url encoding/decoding."""
        test_string = "test data"
        
        encoded = scanner._base64_encode(test_string)
        decoded = scanner._base64_decode(encoded)
        
        assert decoded == test_string
    
    def test_analyze_response_token_accepted(self, scanner, mock_response):
        """Test detection when modified token is accepted."""
        response = mock_response(status_code=200, body='{"user": "admin"}')
        
        result = scanner.analyze_response(response, 'modified.token.here')
        
        assert result['detected'] is True
        assert result['confidence'] >= 0.85
        assert 'Modified JWT token accepted' in result['evidence']
    
    def test_analyze_response_token_rejected(self, scanner, mock_response):
        """Test when token is properly rejected."""
        response = mock_response(
            status_code=401,
            body='{"error": "Invalid token signature"}'
        )
        
        result = scanner.analyze_response(response, 'modified.token.here')
        
        assert result['detected'] is False
    
    @pytest.mark.asyncio
    async def test_test_none_algorithm(self, scanner, valid_jwt, mock_response):
        """Test none algorithm attack detection."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body='{"success": true}'
        ))
        
        jwt_parts = scanner._parse_jwt(valid_jwt)
        findings = await scanner._test_none_algorithm(
            'https://example.com/api',
            valid_jwt,
            jwt_parts,
            test_endpoint=True
        )
        
        assert isinstance(findings, list)
        if findings:
            assert findings[0].severity == FindingSeverity.CRITICAL
            assert 'none' in findings[0].evidence.lower()
    
    @pytest.mark.asyncio
    async def test_test_weak_secret(self, scanner, valid_jwt):
        """Test weak secret detection."""
        jwt_parts = scanner._parse_jwt(valid_jwt)
        
        findings = await scanner._test_weak_secret(
            'https://example.com/api',
            valid_jwt,
            jwt_parts,
            test_endpoint=False
        )
        
        assert isinstance(findings, list)
    
    @pytest.mark.asyncio
    async def test_scan_no_token(self, scanner):
        """Test scan when no JWT token present."""
        findings = await scanner.scan(
            'https://example.com/api',
            headers={}
        )
        
        assert findings == []
    
    @pytest.mark.asyncio
    async def test_scan_with_token(self, scanner, valid_jwt):
        """Test scan with JWT token."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=Mock(
            status_code=200,
            body='{}',
            headers={}
        ))
        
        findings = await scanner.scan(
            'https://example.com/api',
            token=valid_jwt,
            test_endpoint=True
        )
        
        assert isinstance(findings, list)
        assert scanner.stats['scans_performed'] > 0
    
    def test_get_jwt_scanner_singleton(self):
        """Test global scanner instance."""
        scanner1 = get_jwt_scanner()
        scanner2 = get_jwt_scanner()
        
        assert scanner1 is scanner2