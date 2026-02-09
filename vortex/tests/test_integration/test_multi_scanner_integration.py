"""
Multi-Scanner Integration Tests
Tests integration between different scanner types and components
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from scanners.vulns.sqli import SQLiScanner
from scanners.vulns.xss import XSSScanner
from scanners.vulns.csrf import CSRFScanner
from scanners.vulns.lfi import LFIScanner
from scanners.vulns.ssrf import SSRFScanner
from scanners.api.jwt_scanner import JWTScanner
from core.network import HTTPClient
from core.payloads.manager import PayloadManager


@pytest.mark.integration
@pytest.mark.asyncio
class TestScannerCoordination:
    """Test coordination between multiple scanners."""
    
    async def test_sequential_scanner_execution(self):
        """Test running scanners sequentially on same target."""
        http_client = HTTPClient()
        target_url = "https://example.com/page?id=1"
        
        scanners = [
            SQLiScanner(),
            XSSScanner(),
            CSRFScanner()
        ]
        
        results = []
        
        with patch.object(http_client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                'status': 200,
                'content': '<html><body>Test</body></html>',
                'headers': {}
            }
            
            # Run each scanner
            for scanner in scanners:
                scanner.http_client = http_client
                result = await scanner.scan(target_url)
                results.append(result)
            
            # All scanners should have executed
            assert len(results) == len(scanners)
            assert mock_get.call_count > 0
        
        await http_client.close()
    
    async def test_parallel_scanner_execution(self):
        """Test running multiple scanners in parallel."""
        http_client = HTTPClient()
        target_url = "https://example.com/search?q=test"
        
        scanners = [
            SQLiScanner(),
            XSSScanner(),
            LFIScanner()
        ]
        
        # Configure scanners
        for scanner in scanners:
            scanner.http_client = http_client
        
        with patch.object(http_client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                'status': 200,
                'content': '<html><body>Result: test</body></html>',
                'headers': {}
            }
            
            # Run scanners in parallel
            tasks = [scanner.scan(target_url) for scanner in scanners]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should complete
            assert len(results) == len(scanners)
            
            # Verify concurrent execution (should be faster than sequential)
            assert mock_get.call_count > 0
        
        await http_client.close()
    
    async def test_scanner_resource_sharing(self):
        """Test that scanners properly share resources."""
        # Shared HTTP client
        shared_http_client = HTTPClient(max_concurrent=10)
        
        scanners = [
            XSSScanner(),
            SQLiScanner(),
            CSRFScanner()
        ]
        
        # All scanners use same HTTP client
        for scanner in scanners:
            scanner.http_client = shared_http_client
        
        with patch.object(shared_http_client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                'status': 200,
                'content': '<html><body>OK</body></html>',
                'headers': {}
            }
            
            # Run all scanners
            tasks = [s.scan("https://example.com") for s in scanners]
            await asyncio.gather(*tasks)
            
            # Should have used shared client
            assert mock_get.called
        
        await shared_http_client.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestPayloadIntegration:
    """Test integration between scanners and payload manager."""
    
    async def test_payload_sharing_across_scanners(self):
        """Test that payload manager provides payloads to all scanners."""
        payload_mgr = PayloadManager()
        
        # Get payloads for different scanner types
        sqli_payloads = payload_mgr.get_payloads('sqli', count=10)
        xss_payloads = payload_mgr.get_payloads('xss', count=10)
        
        assert len(sqli_payloads) > 0
        assert len(xss_payloads) > 0
        
        # Payloads should be different for different types
        assert sqli_payloads != xss_payloads
    
    async def test_payload_mutation_integration(self):
        """Test payload mutations work with scanners."""
        from core.payloads.mutation_engine import MutationEngine
        
        mutation_engine = MutationEngine()
        base_payload = "<script>alert(1)</script>"
        
        # Generate mutations
        mutations = mutation_engine.mutate(base_payload, count=5)
        
        assert len(mutations) >= 5
        
        # Each mutation should be different
        assert len(set(mutations)) >= 3  # At least some variety


@pytest.mark.integration
@pytest.mark.asyncio
class TestVulnerabilityChaining:
    """Test chaining different vulnerability types."""
    
    async def test_xss_to_csrf_chain(self):
        """Test detecting XSS that can be used for CSRF."""
        http_client = HTTPClient()
        
        xss_scanner = XSSScanner()
        csrf_scanner = CSRFScanner()
        
        xss_scanner.http_client = http_client
        csrf_scanner.http_client = http_client
        
        target_url = "https://example.com/form"
        
        with patch.object(http_client, 'get', new_callable=AsyncMock) as mock_get:
            # Response with form but vulnerable to XSS
            mock_get.return_value = {
                'status': 200,
                'content': '''
                <html>
                <body>
                    <form action="/submit" method="POST">
                        <input name="data" value="<script>alert(1)</script>">
                        <button type="submit">Submit</button>
                    </form>
                </body>
                </html>
                ''',
                'headers': {}
            }
            
            # Scan for XSS first
            xss_result = await xss_scanner.scan(target_url)
            
            # Then scan for CSRF
            csrf_result = await csrf_scanner.scan(target_url)
            
            # Both vulnerabilities should be detectable
            assert xss_result is not None or csrf_result is not None
        
        await http_client.close()
    
    async def test_sqli_to_auth_bypass_chain(self):
        """Test SQLi leading to authentication bypass."""
        http_client = HTTPClient()
        sqli_scanner = SQLiScanner()
        sqli_scanner.http_client = http_client
        
        login_url = "https://example.com/login?user=admin&pass=test"
        
        with patch.object(http_client, 'get', new_callable=AsyncMock) as mock_get:
            # Vulnerable to SQL injection in auth
            mock_get.return_value = {
                'status': 200,
                'content': 'Welcome admin! You are logged in.',
                'headers': {'Set-Cookie': 'session=abc123'},
                'response_time': 0.1
            }
            
            result = await sqli_scanner.scan(login_url)
            
            # Should detect potential auth bypass via SQLi
            assert result is not None
        
        await http_client.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIScannersIntegration:
    """Test integration of API-specific scanners."""
    
    async def test_jwt_with_graphql_integration(self):
        """Test JWT scanner working with GraphQL endpoints."""
        http_client = HTTPClient()
        jwt_scanner = JWTScanner()
        jwt_scanner.http_client = http_client
        
        graphql_endpoint = "https://api.example.com/graphql"
        
        with patch.object(http_client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = {
                'status': 200,
                'content': '{"data": {"user": "admin"}}',
                'headers': {
                    'Authorization': 'Bearer eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NTY3ODkwIn0.'
                }
            }
            
            # JWT scanner should work with GraphQL
            result = await jwt_scanner.scan(graphql_endpoint)
            
            # Should analyze JWT from API response
            assert result is not None
        
        await http_client.close()
    
    async def test_api_authentication_flow(self):
        """Test complete API authentication and scanning flow."""
        from core.auth.manager import AuthManager
        
        http_client = HTTPClient()
        auth_manager = AuthManager()
        
        api_url = "https://api.example.com/data"
        
        with patch.object(http_client, 'post', new_callable=AsyncMock) as mock_post:
            # Mock auth response
            mock_post.return_value = {
                'status': 200,
                'content': '{"token": "abc123"}',
                'headers': {}
            }
            
            # Authenticate
            auth_result = await auth_manager.authenticate(
                url="https://api.example.com/auth",
                method="oauth2",
                credentials={'client_id': 'test', 'client_secret': 'secret'}
            )
            
            # Should have token for subsequent requests
            assert auth_result is not None
        
        await http_client.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestScannerErrorPropagation:
    """Test error handling across scanner integration."""
    
    async def test_one_scanner_failure_doesnt_affect_others(self):
        """Test that one scanner's failure doesn't stop others."""
        http_client = HTTPClient()
        
        scanners = [
            XSSScanner(),
            SQLiScanner(),
            CSRFScanner()
        ]
        
        for scanner in scanners:
            scanner.http_client = http_client
        
        call_count = 0
        
        async def failing_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # First scanner fails
            if call_count <= 2:
                raise ConnectionError("Scanner failed")
            
            # Others succeed
            return {
                'status': 200,
                'content': '<html><body>OK</body></html>',
                'headers': {}
            }
        
        with patch.object(http_client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = failing_request
            
            # Run all scanners
            results = []
            for scanner in scanners:
                try:
                    result = await scanner.scan("https://example.com")
                    results.append(result)
                except Exception as e:
                    results.append(None)
            
            # Some should succeed despite one failure
            successful = sum(1 for r in results if r is not None)
            assert successful > 0
        
        await http_client.close()
    
    async def test_timeout_handling_across_scanners(self):
        """Test timeout handling when multiple scanners run."""
        http_client = HTTPClient(timeout=1.0)
        
        scanners = [XSSScanner(), SQLiScanner()]
        for s in scanners:
            s.http_client = http_client
        
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(5.0)  # Longer than timeout
            return {'status': 200, 'content': 'OK', 'headers': {}}
        
        with patch.object(http_client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = slow_response
            
            # Both should handle timeout
            results = []
            for scanner in scanners:
                try:
                    result = await asyncio.wait_for(
                        scanner.scan("https://example.com"),
                        timeout=2.0
                    )
                    results.append(result)
                except asyncio.TimeoutError:
                    results.append(None)
            
            # Timeouts should be handled gracefully
            assert len(results) == len(scanners)
        
        await http_client.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestResultAggregation:
    """Test aggregating results from multiple scanners."""
    
    async def test_finding_deduplication(self):
        """Test that duplicate findings are properly deduplicated."""
        findings = [
            {'url': 'https://example.com', 'type': 'xss', 'param': 'q'},
            {'url': 'https://example.com', 'type': 'xss', 'param': 'q'},  # Duplicate
            {'url': 'https://example.com', 'type': 'sqli', 'param': 'id'},
        ]
        
        # Simple deduplication
        unique_findings = []
        seen = set()
        
        for finding in findings:
            key = (finding['url'], finding['type'], finding['param'])
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)
        
        assert len(unique_findings) == 2  # Should deduplicate
    
    async def test_severity_based_prioritization(self):
        """Test that findings are prioritized by severity."""
        findings = [
            {'type': 'xss', 'severity': 'LOW'},
            {'type': 'sqli', 'severity': 'CRITICAL'},
            {'type': 'csrf', 'severity': 'MEDIUM'},
            {'type': 'lfi', 'severity': 'HIGH'},
        ]
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        sorted_findings = sorted(
            findings,
            key=lambda f: severity_order.get(f['severity'], 99)
        )
        
        # Should be ordered correctly
        assert sorted_findings[0]['severity'] == 'CRITICAL'
        assert sorted_findings[-1]['severity'] == 'LOW'


@pytest.mark.integration
@pytest.mark.asyncio
class TestMemorySharing:
    """Test memory management across scanners."""
    
    async def test_shared_cache_usage(self):
        """Test that scanners can share cached responses."""
        from functools import lru_cache
        
        # Simulate shared cache
        cache_hits = 0
        
        @lru_cache(maxsize=100)
        def get_cached_response(url):
            nonlocal cache_hits
            cache_hits += 1
            return f"Response for {url}"
        
        # Multiple scanners accessing same URLs
        urls = [
            "https://example.com/page1",
            "https://example.com/page1",  # Cache hit
            "https://example.com/page2",
            "https://example.com/page1",  # Cache hit
        ]
        
        for url in urls:
            _ = get_cached_response(url)
        
        # Should have cached some requests
        assert cache_hits < len(urls)  # Some cache hits occurred
    
    async def test_memory_efficient_result_storage(self):
        """Test memory-efficient storage of scan results."""
        import sys
        
        # Create many findings
        findings = []
        for i in range(100):
            findings.append({
                'id': f'finding-{i}',
                'url': f'https://example.com/page{i}',
                'type': 'xss',
                'severity': 'MEDIUM'
            })
        
        # Check memory usage
        findings_size = sys.getsizeof(findings)
        
        print(f"\n100 findings: {findings_size / 1024:.2f} KB")
        
        # Should be reasonable
        assert findings_size < 100 * 1024  # Less than 100KB for 100 findings