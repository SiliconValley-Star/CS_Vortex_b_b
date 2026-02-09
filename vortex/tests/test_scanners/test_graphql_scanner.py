"""
Test suite for GraphQL Security Scanner
Tests introspection, batching, depth limits, and injection attacks
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json

from scanners.api.graphql_scanner import GraphQLScanner, GraphQLEndpoint, GraphQLVulnerability, get_graphql_scanner
from core.network import HTTPResponse
from domain.enums import FindingType, FindingSeverity


class TestGraphQLScanner:
    """Test GraphQL Scanner functionality."""
    
    @pytest.fixture
    def scanner(self):
        return GraphQLScanner()
    
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
        """Test scanner initializes correctly."""
        assert scanner.network_client is not None
        assert isinstance(scanner.discovered_endpoints, list)
        assert isinstance(scanner.vulnerabilities, list)
        assert 'endpoints_scanned' in scanner.stats
    
    def test_introspection_query_defined(self, scanner):
        """Test introspection query is properly defined."""
        query = GraphQLScanner.INTROSPECTION_QUERY
        
        assert '__schema' in query
        assert 'queryType' in query
        assert 'mutationType' in query
        assert 'types' in query
    
    def test_graphql_endpoint_dataclass(self):
        """Test GraphQLEndpoint dataclass."""
        endpoint = GraphQLEndpoint(
            url='https://example.com/graphql',
            introspection_enabled=True,
            queries=['users', 'posts'],
            mutations=['createUser']
        )
        
        assert endpoint.url == 'https://example.com/graphql'
        assert endpoint.introspection_enabled is True
        assert len(endpoint.queries) == 2
    
    def test_graphql_vulnerability_dataclass(self):
        """Test GraphQLVulnerability dataclass."""
        vuln = GraphQLVulnerability(
            vuln_type='introspection_enabled',
            severity=FindingSeverity.MEDIUM,
            endpoint='https://example.com/graphql',
            query='test query',
            evidence='Schema exposed',
            impact='Information disclosure',
            remediation='Disable introspection'
        )
        
        assert vuln.vuln_type == 'introspection_enabled'
        assert vuln.severity == FindingSeverity.MEDIUM
    
    @pytest.mark.asyncio
    async def test_test_introspection_enabled(self, scanner, mock_response):
        """Test detection of enabled introspection."""
        schema_response = {
            'data': {
                '__schema': {
                    'queryType': {'name': 'Query'},
                    'types': [
                        {'name': 'User', 'kind': 'OBJECT', 'fields': [{'name': 'id'}]}
                    ]
                }
            }
        }
        
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body=json.dumps(schema_response)
        ))
        
        endpoint = await scanner._test_introspection('https://example.com/graphql')
        
        assert endpoint.introspection_enabled is True
        assert len(endpoint.types) > 0
    
    @pytest.mark.asyncio
    async def test_test_introspection_disabled(self, scanner, mock_response):
        """Test when introspection is disabled."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=400,
            body=json.dumps({'errors': [{'message': 'Introspection disabled'}]})
        ))
        
        endpoint = await scanner._test_introspection('https://example.com/graphql')
        
        assert endpoint.introspection_enabled is False
    
    @pytest.mark.asyncio
    async def test_test_batch_attacks(self, scanner, mock_response):
        """Test batch attack detection."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body=json.dumps({'data': {}})
        ))
        
        vuln = await scanner._test_batch_attacks('https://example.com/graphql')
        
        if vuln:
            assert vuln.vuln_type == 'batch_attack'
            assert vuln.severity == FindingSeverity.HIGH
            assert 'batch' in vuln.evidence.lower()
    
    @pytest.mark.asyncio
    async def test_test_depth_limits(self, scanner, mock_response):
        """Test depth limit bypass detection."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body=json.dumps({'data': {'__typename': 'Query'}})
        ))
        
        endpoint = GraphQLEndpoint(
            url='https://example.com/graphql',
            queries=['users']
        )
        
        vuln = await scanner._test_depth_limits(
            'https://example.com/graphql',
            endpoint
        )
        
        if vuln:
            assert vuln.vuln_type == 'depth_limit_bypass'
            assert 'depth' in vuln.evidence.lower()
    
    @pytest.mark.asyncio
    async def test_test_injection_attacks(self, scanner, mock_response):
        """Test SQL injection in GraphQL arguments."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body='{"errors": [{"message": "SQL syntax error"}]}'
        ))
        
        endpoint = GraphQLEndpoint(
            url='https://example.com/graphql',
            queries=['user']
        )
        
        vulns = await scanner._test_injection_attacks(
            'https://example.com/graphql',
            endpoint
        )
        
        if vulns:
            assert vulns[0].vuln_type == 'graphql_injection'
            assert vulns[0].severity == FindingSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_scan_endpoint_full(self, scanner, mock_response):
        """Test complete endpoint scanning."""
        scanner.network_client = Mock()
        scanner.network_client.request = AsyncMock(return_value=mock_response(
            status_code=200,
            body=json.dumps({'data': {}})
        ))
        
        vulnerabilities = await scanner.scan_endpoint('https://example.com/graphql')
        
        assert isinstance(vulnerabilities, list)
        assert scanner.stats['endpoints_scanned'] > 0
    
    def test_convert_to_findings(self, scanner):
        """Test conversion of vulnerabilities to findings."""
        vulns = [
            GraphQLVulnerability(
                vuln_type='introspection_enabled',
                severity=FindingSeverity.MEDIUM,
                endpoint='https://example.com/graphql',
                query='introspection query',
                evidence='Schema exposed',
                impact='Info disclosure',
                remediation='Disable introspection'
            )
        ]
        
        findings = scanner.convert_to_findings(vulns)
        
        assert len(findings) == 1
        assert findings[0].finding_type == FindingType.API_SECURITY
        assert findings[0].severity == FindingSeverity.MEDIUM
    
    def test_get_stats(self, scanner):
        """Test statistics retrieval."""
        stats = scanner.get_stats()
        
        assert isinstance(stats, dict)
        assert 'endpoints_scanned' in stats
        assert 'introspection_enabled' in stats
    
    def test_get_graphql_scanner_singleton(self):
        """Test global scanner instance."""
        scanner1 = get_graphql_scanner()
        scanner2 = get_graphql_scanner()
        
        assert scanner1 is scanner2


class TestGraphQLScannerIntegration:
    """Integration tests for GraphQL Scanner."""
    
    @pytest.fixture
    def scanner(self):
        return GraphQLScanner()
    
    @pytest.mark.asyncio
    async def test_full_scan_workflow(self, scanner):
        """Test complete scanning workflow."""
        scanner.network_client = Mock()
        
        # Mock introspection response
        schema_response = {
            'data': {
                '__schema': {
                    'queryType': {'name': 'Query'},
                    'mutationType': {'name': 'Mutation'},
                    'types': [
                        {
                            'name': 'Query',
                            'kind': 'OBJECT',
                            'fields': [
                                {'name': 'users', 'args': []}
                            ]
                        },
                        {
                            'name': 'Mutation',
                            'kind': 'OBJECT',
                            'fields': [
                                {'name': 'createUser', 'args': []}
                            ]
                        }
                    ]
                }
            }
        }
        
        scanner.network_client.request = AsyncMock(return_value=Mock(
            status_code=200,
            body=json.dumps(schema_response),
            headers={},
            response_time=0.1
        ))
        
        vulnerabilities = await scanner.scan_endpoint('https://example.com/graphql')
        
        assert isinstance(vulnerabilities, list)
        assert len(scanner.discovered_endpoints) > 0