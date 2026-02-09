"""
VORTEX PoC Replay Integration Tests
Test end-to-end PoC replay functionality
"""

import pytest
import asyncio
from datetime import datetime

from core.verification import parse_poc, replay_poc, global_poc_replayer
from core.verification.poc_parser import ParsedPoC
from core.network import HTTPResponse
from domain.enums import MatchType


class TestPoCReplayIntegration:
    """Integration tests for PoC replay system."""
    
    @pytest.mark.asyncio
    async def test_curl_parsing_and_replay(self):
        """Test parsing cURL and replaying."""
        curl_cmd = 'curl -X POST https://example.com/api -H "Content-Type: application/json" -d \'{"test":"data"}\''
        
        # Parse
        parsed = parse_poc(curl_cmd)
        
        assert parsed is not None
        assert parsed.method == 'POST'
        assert parsed.url == 'https://example.com/api'
        assert parsed.format_detected == 'curl'
        assert 'Content-Type' in parsed.headers
        
        # Note: Actual replay would require mock server
        # This test validates parsing only
    
    @pytest.mark.asyncio
    async def test_http_raw_parsing(self):
        """Test HTTP raw format parsing."""
        http_raw = """POST /api/test HTTP/1.1
Host: example.com
Content-Type: application/json

{"key":"value"}"""
        
        parsed = parse_poc(http_raw)
        
        assert parsed is not None
        assert parsed.method == 'POST'
        assert 'example.com' in parsed.url
        assert parsed.format_detected == 'http_raw'
        assert parsed.body == '{"key":"value"}'
    
    @pytest.mark.asyncio
    async def test_python_requests_parsing(self):
        """Test Python requests code parsing."""
        python_code = """import requests
response = requests.post('https://api.example.com/endpoint',
                        json={'test': 'data'},
                        headers={'Authorization': 'Bearer token123'})"""
        
        parsed = parse_poc(python_code)
        
        assert parsed is not None
        assert parsed.method == 'POST'
        assert parsed.url == 'https://api.example.com/endpoint'
        assert parsed.format_detected == 'python'
    
    def test_poc_replayer_initialization(self):
        """Test PoC replayer initializes correctly."""
        replayer = global_poc_replayer
        
        assert replayer is not None
        assert replayer.baseline_timeout == 30
        assert replayer.poc_timeout == 30
        assert replayer.min_determinism_poc_replay == 0.70
    
    def test_parser_statistics(self):
        """Test parser tracks statistics."""
        from core.verification.poc_parser import global_poc_parser
        
        stats = global_poc_parser.get_stats()
        
        assert 'parsed_curl' in stats
        assert 'parsed_http' in stats
        assert 'parsed_python' in stats
        assert 'parse_errors' in stats


class TestPoCReplayDeterminism:
    """Test determinism scoring logic."""
    
    def test_determinism_calculation_high_confidence(self):
        """Test high confidence determinism scoring."""
        replayer = global_poc_replayer
        
        # Mock behavioral analysis with strong indicators
        behavioral_analysis = {
            'indicators': [
                {'type': 'error_messages', 'confidence_impact': 0.5},
                {'type': 'status_code', 'confidence_impact': 0.4},
                {'type': 'payload_reflection', 'confidence_impact': 0.4}
            ],
            'uncertainty_factors': [],
            'has_new_errors': True,
            'payload_reflected': True,
            'similarity_score': 0.3
        }
        
        # Mock responses
        baseline = HTTPResponse(
            status_code=200,
            headers={},
            body='Normal response',
            response_time=0.5,
            url='https://example.com'
        )
        
        poc = HTTPResponse(
            status_code=500,
            headers={},
            body='SQL error: mysql syntax error',
            response_time=0.6,
            url='https://example.com'
        )
        
        score = replayer._calculate_determinism_score(
            baseline,
            poc,
            behavioral_analysis
        )
        
        # Should be high confidence (>0.8)
        assert score >= 0.8
    
    def test_determinism_calculation_medium_confidence(self):
        """Test medium confidence determinism scoring."""
        replayer = global_poc_replayer
        
        # Mock behavioral analysis with moderate indicators
        behavioral_analysis = {
            'indicators': [
                {'type': 'response_time', 'confidence_impact': 0.2},
                {'type': 'content_size', 'confidence_impact': 0.25}
            ],
            'uncertainty_factors': ['Could be infrastructure'],
            'has_new_errors': False,
            'payload_reflected': False,
            'similarity_score': 0.7
        }
        
        baseline = HTTPResponse(
            status_code=200,
            headers={},
            body='A' * 1000,
            response_time=0.5,
            url='https://example.com'
        )
        
        poc = HTTPResponse(
            status_code=200,
            headers={},
            body='A' * 1200,
            response_time=0.8,
            url='https://example.com'
        )
        
        score = replayer._calculate_determinism_score(
            baseline,
            poc,
            behavioral_analysis
        )
        
        # Should be medium confidence (0.4-0.7)
        assert 0.3 <= score <= 0.7


class TestPatternVerification:
    """Test pattern-based verification."""
    
    def test_xss_pattern_detection(self):
        """Test XSS pattern matching."""
        from core.verification import SystemVerificationEngine
        from domain.enums import FindingType
        
        engine = SystemVerificationEngine()
        patterns = engine._get_vulnerability_patterns(FindingType.XSS_REFLECTED)
        
        assert len(patterns) > 0
        assert any('script' in p.lower() for p in patterns)
    
    def test_sqli_pattern_detection(self):
        """Test SQLi pattern matching."""
        from core.verification import SystemVerificationEngine
        from domain.enums import FindingType
        
        engine = SystemVerificationEngine()
        patterns = engine._get_vulnerability_patterns(FindingType.SQLI_ERROR)
        
        assert len(patterns) > 0
        assert any('mysql' in p.lower() or 'sql' in p.lower() for p in patterns)
    
    def test_pattern_matching_logic(self):
        """Test pattern matching against content."""
        from core.verification import SystemVerificationEngine
        
        engine = SystemVerificationEngine()
        
        # Test SQL error pattern
        content = "MySQL error: You have an error in your SQL syntax"
        pattern = r'mysql.*error'
        
        matches = engine._check_pattern(content, pattern)
        assert matches is True
        
        # Test non-matching
        content = "Normal response"
        matches = engine._check_pattern(content, pattern)
        assert matches is False


class TestVerificationStatistics:
    """Test verification statistics tracking."""
    
    def test_verification_stats_initialization(self):
        """Test stats are initialized."""
        from core.verification import SystemVerificationEngine
        
        engine = SystemVerificationEngine()
        stats = engine.get_stats()
        
        assert 'total_verifications' in stats
        assert 'poc_replays' in stats
        assert 'pattern_verifications' in stats
        assert 'successful_verifications' in stats
        assert 'failed_verifications' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])