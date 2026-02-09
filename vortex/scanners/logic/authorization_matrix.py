#!/usr/bin/env python3
"""
Authorization Matrix Testing Module (PHASE 4.2)
Tests for authorization bypass and IDOR vulnerabilities
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import structlog

logger = structlog.get_logger()


class UserRole(Enum):
    """User role levels"""
    ANONYMOUS = "anonymous"
    AUTHENTICATED = "authenticated"
    USER = "user"
    PREMIUM_USER = "premium"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


class AccessType(Enum):
    """Types of access"""
    READ = "read"
    CREATE = "create"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"


@dataclass
class Endpoint:
    """API endpoint definition"""
    url: str
    method: str
    description: str
    required_role: UserRole
    access_type: AccessType
    resource_pattern: Optional[str] = None  # e.g., /users/{id}


@dataclass
class AuthTest:
    """Authorization test case"""
    name: str
    endpoint: Endpoint
    test_role: UserRole
    expected_allowed: bool
    test_resource_id: Optional[str] = None
    description: str = ""


@dataclass
class AuthVulnerability:
    """Detected authorization vulnerability"""
    vuln_type: str
    endpoint: str
    method: str
    test_role: UserRole
    expected_role: UserRole
    resource_id: Optional[str]
    evidence: str
    severity: str = "HIGH"


class AuthorizationMatrixTester:
    """
    Authorization Matrix Testing (PHASE 4.2)
    
    Tests for:
    - Horizontal privilege escalation (IDOR)
    - Vertical privilege escalation
    - Missing access controls
    - Role-based access control (RBAC) bypass
    """
    
    def __init__(self):
        # Common resource ID patterns
        self.RESOURCE_PATTERNS = [
            r'/users/(\d+)',
            r'/user/(\d+)',
            r'/profile/(\d+)',
            r'/account/(\d+)',
            r'/api/v\d+/users/(\d+)',
            r'/posts/(\d+)',
            r'/documents/(\d+)',
            r'/files/(\d+)',
            r'/orders/(\d+)',
        ]
        
        # Sensitive endpoints (admin-only)
        self.ADMIN_ENDPOINTS = [
            '/admin', '/dashboard', '/users', '/settings',
            '/config', '/logs', '/system', '/management'
        ]
        
        # Role hierarchy (lower = less privilege)
        self.ROLE_HIERARCHY = {
            UserRole.ANONYMOUS: 0,
            UserRole.AUTHENTICATED: 1,
            UserRole.USER: 2,
            UserRole.PREMIUM_USER: 3,
            UserRole.MODERATOR: 4,
            UserRole.ADMIN: 5,
            UserRole.SUPERADMIN: 6
        }
        
        logger.info("Authorization Matrix Tester initialized")
    
    def generate_matrix_tests(
        self,
        endpoints: List[Endpoint]
    ) -> List[AuthTest]:
        """
        Generate authorization matrix test cases
        
        Args:
            endpoints: List of endpoints to test
            
        Returns:
            List of authorization test cases
        """
        tests = []
        
        for endpoint in endpoints:
            # Test with all role levels
            for test_role in UserRole:
                # Determine if access should be allowed
                expected_allowed = self._should_allow_access(
                    test_role,
                    endpoint.required_role
                )
                
                test = AuthTest(
                    name=f"{test_role.value} → {endpoint.method} {endpoint.url}",
                    endpoint=endpoint,
                    test_role=test_role,
                    expected_allowed=expected_allowed,
                    description=f"Test {test_role.value} access to {endpoint.description}"
                )
                tests.append(test)
        
        logger.info(
            f"Generated {len(tests)} authorization tests",
            endpoints=len(endpoints)
        )
        
        return tests
    
    def generate_idor_tests(
        self,
        url: str,
        method: str,
        user_id: str,
        other_user_ids: List[str]
    ) -> List[AuthTest]:
        """
        Generate IDOR (Insecure Direct Object Reference) tests
        
        Args:
            url: Endpoint URL with {id} placeholder
            method: HTTP method
            user_id: Current user's ID
            other_user_ids: Other users' IDs to test access
            
        Returns:
            List of IDOR test cases
        """
        tests = []
        
        for other_id in other_user_ids:
            # Test accessing other user's resource
            test_url = url.replace('{id}', other_id)
            
            endpoint = Endpoint(
                url=test_url,
                method=method,
                description=f"Access user {other_id}'s resource",
                required_role=UserRole.USER,
                access_type=AccessType.READ,
                resource_pattern=url
            )
            
            test = AuthTest(
                name=f"IDOR: User {user_id} → User {other_id}",
                endpoint=endpoint,
                test_role=UserRole.USER,
                expected_allowed=False,  # Should NOT be allowed
                test_resource_id=other_id,
                description=f"Horizontal privilege escalation test"
            )
            tests.append(test)
        
        logger.info(
            f"Generated {len(tests)} IDOR tests",
            user_id=user_id,
            targets=len(other_user_ids)
        )
        
        return tests
    
    def detect_resource_ids(self, url: str) -> List[str]:
        """
        Detect resource IDs in URL
        
        Args:
            url: URL to analyze
            
        Returns:
            List of detected resource IDs
        """
        ids = []
        
        for pattern in self.RESOURCE_PATTERNS:
            matches = re.findall(pattern, url)
            ids.extend(matches)
        
        return ids
    
    def is_admin_endpoint(self, url: str) -> bool:
        """Check if URL is admin-only endpoint"""
        url_lower = url.lower()
        return any(admin in url_lower for admin in self.ADMIN_ENDPOINTS)
    
    def _should_allow_access(
        self,
        test_role: UserRole,
        required_role: UserRole
    ) -> bool:
        """
        Determine if role should have access
        
        Args:
            test_role: Role being tested
            required_role: Required role for access
            
        Returns:
            True if access should be allowed
        """
        test_level = self.ROLE_HIERARCHY.get(test_role, 0)
        required_level = self.ROLE_HIERARCHY.get(required_role, 0)
        
        return test_level >= required_level
    
    def analyze_response(
        self,
        test: AuthTest,
        status_code: int,
        response_body: str
    ) -> Optional[AuthVulnerability]:
        """
        Analyze response to detect authorization vulnerabilities
        
        Args:
            test: The authorization test
            status_code: Response status code
            response_body: Response body
            
        Returns:
            AuthVulnerability if detected, None otherwise
        """
        # If access should be denied but was granted
        if not test.expected_allowed:
            if 200 <= status_code < 300:
                # Check if actually succeeded (not just error page)
                error_indicators = [
                    'unauthorized', 'forbidden', 'access denied',
                    'not authorized', 'permission denied', '401', '403'
                ]
                
                response_lower = response_body.lower()
                has_error = any(ind in response_lower for ind in error_indicators)
                
                if not has_error:
                    # Determine vulnerability type
                    if test.test_resource_id:
                        vuln_type = "IDOR (Horizontal Privilege Escalation)"
                    elif self._is_vertical_escalation(test):
                        vuln_type = "Vertical Privilege Escalation"
                    else:
                        vuln_type = "Missing Access Control"
                    
                    return AuthVulnerability(
                        vuln_type=vuln_type,
                        endpoint=test.endpoint.url,
                        method=test.endpoint.method,
                        test_role=test.test_role,
                        expected_role=test.endpoint.required_role,
                        resource_id=test.test_resource_id,
                        evidence=f"Status: {status_code}, Role: {test.test_role.value}",
                        severity="HIGH" if vuln_type == "IDOR" else "CRITICAL"
                    )
        
        # If access should be granted but was denied (false positive check)
        elif test.expected_allowed:
            if status_code in [401, 403]:
                logger.warning(
                    f"Possible false positive: {test.test_role.value} denied on {test.endpoint.url}"
                )
        
        return None
    
    def _is_vertical_escalation(self, test: AuthTest) -> bool:
        """Check if test represents vertical privilege escalation"""
        test_level = self.ROLE_HIERARCHY.get(test.test_role, 0)
        required_level = self.ROLE_HIERARCHY.get(test.endpoint.required_role, 0)
        
        # Vertical if trying to access higher privilege endpoint
        return test_level < required_level
    
    def get_test_summary(self, tests: List[AuthTest]) -> Dict:
        """Get summary of authorization tests"""
        summary = {
            'total_tests': len(tests),
            'by_role': {},
            'by_method': {},
            'should_deny': 0,
            'should_allow': 0
        }
        
        for test in tests:
            # By role
            role = test.test_role.value
            summary['by_role'][role] = summary['by_role'].get(role, 0) + 1
            
            # By method
            method = test.endpoint.method
            summary['by_method'][method] = summary['by_method'].get(method, 0) + 1
            
            # Expected results
            if test.expected_allowed:
                summary['should_allow'] += 1
            else:
                summary['should_deny'] += 1
        
        return summary
    
    def create_test_matrix(
        self,
        endpoints: List[Endpoint],
        roles: Optional[List[UserRole]] = None
    ) -> Dict[str, Dict[str, bool]]:
        """
        Create authorization matrix
        
        Returns:
            Matrix of endpoint → role → allowed
        """
        if roles is None:
            roles = list(UserRole)
        
        matrix = {}
        
        for endpoint in endpoints:
            endpoint_key = f"{endpoint.method} {endpoint.url}"
            matrix[endpoint_key] = {}
            
            for role in roles:
                allowed = self._should_allow_access(role, endpoint.required_role)
                matrix[endpoint_key][role.value] = allowed
        
        return matrix


# Global instance
authorization_tester = AuthorizationMatrixTester()