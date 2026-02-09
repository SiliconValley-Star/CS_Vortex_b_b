#!/usr/bin/env python3
"""
Business Logic Flaw Analyzer (PHASE 4.1)
Detects logic-based vulnerabilities in application workflows
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger()


class LogicFlawType(Enum):
    """Types of business logic flaws"""
    PRICE_MANIPULATION = "price_manipulation"
    QUANTITY_BYPASS = "quantity_bypass"
    WORKFLOW_BYPASS = "workflow_bypass"
    STATE_CONFUSION = "state_confusion"
    LIMIT_BYPASS = "limit_bypass"
    DISCOUNT_ABUSE = "discount_abuse"
    NEGATIVE_VALUE = "negative_value"
    EXCESSIVE_VALUE = "excessive_value"
    REFERRAL_ABUSE = "referral_abuse"
    COUPON_REUSE = "coupon_reuse"


@dataclass
class LogicTest:
    """A single business logic test"""
    name: str
    flaw_type: LogicFlawType
    parameter: str
    test_value: any
    expected_behavior: str
    description: str


@dataclass
class LogicFlaw:
    """Detected business logic flaw"""
    flaw_type: LogicFlawType
    url: str
    parameter: str
    test_value: any
    original_value: any
    evidence: str
    severity: str = "HIGH"
    description: str = ""


class BusinessLogicAnalyzer:
    """
    Business Logic Flaw Detector (PHASE 4.1)
    
    Detects common business logic vulnerabilities:
    - Price manipulation (negative, zero prices)
    - Quantity bypass (negative quantities, overflow)
    - Workflow bypass (skipping required steps)
    - Discount/coupon abuse
    - Limit bypass
    """
    
    def __init__(self):
        # Suspicious parameter names
        self.PRICE_PARAMS = {
            'price', 'cost', 'amount', 'total', 'subtotal',
            'value', 'fee', 'charge', 'payment'
        }
        
        self.QUANTITY_PARAMS = {
            'quantity', 'qty', 'count', 'items', 'amount',
            'number', 'num', 'units'
        }
        
        self.DISCOUNT_PARAMS = {
            'discount', 'coupon', 'promo', 'voucher', 'code',
            'promotion', 'rebate', 'offer'
        }
        
        self.LIMIT_PARAMS = {
            'limit', 'max', 'maximum', 'quota', 'threshold'
        }
        
        # Test values for different scenarios
        self.NEGATIVE_TESTS = [-1, -100, -999999]
        self.ZERO_TESTS = [0, 0.0, "0"]
        self.OVERFLOW_TESTS = [999999999, 2147483647, 9999999999]
        self.BOUNDARY_TESTS = [-1, 0, 1, 100, 999, 1000, 9999]
        
        logger.info("Business Logic Analyzer initialized")
    
    def analyze_endpoint(
        self,
        url: str,
        params: Dict[str, any],
        method: str = "GET"
    ) -> List[LogicTest]:
        """
        Analyze endpoint for business logic vulnerabilities
        
        Args:
            url: Target URL
            params: Current parameters
            method: HTTP method
            
        Returns:
            List of logic tests to perform
        """
        tests = []
        
        # Analyze each parameter
        for param_name, param_value in params.items():
            param_lower = param_name.lower()
            
            # Price manipulation tests
            if any(p in param_lower for p in self.PRICE_PARAMS):
                tests.extend(self._generate_price_tests(param_name, param_value))
            
            # Quantity manipulation tests
            if any(q in param_lower for q in self.QUANTITY_PARAMS):
                tests.extend(self._generate_quantity_tests(param_name, param_value))
            
            # Discount/coupon tests
            if any(d in param_lower for d in self.DISCOUNT_PARAMS):
                tests.extend(self._generate_discount_tests(param_name, param_value))
            
            # Limit bypass tests
            if any(l in param_lower for l in self.LIMIT_PARAMS):
                tests.extend(self._generate_limit_tests(param_name, param_value))
        
        logger.info(
            f"Generated {len(tests)} business logic tests",
            url=url,
            params=len(params)
        )
        
        return tests
    
    def _generate_price_tests(
        self,
        param_name: str,
        original_value: any
    ) -> List[LogicTest]:
        """Generate price manipulation tests"""
        tests = []
        
        # Negative price test
        tests.append(LogicTest(
            name=f"Negative Price - {param_name}",
            flaw_type=LogicFlawType.PRICE_MANIPULATION,
            parameter=param_name,
            test_value=-1,
            expected_behavior="Reject negative price",
            description="Testing if negative price is accepted"
        ))
        
        # Zero price test
        tests.append(LogicTest(
            name=f"Zero Price - {param_name}",
            flaw_type=LogicFlawType.PRICE_MANIPULATION,
            parameter=param_name,
            test_value=0,
            expected_behavior="Reject zero price",
            description="Testing if zero price is accepted"
        ))
        
        # Fractional price test
        tests.append(LogicTest(
            name=f"Fractional Price - {param_name}",
            flaw_type=LogicFlawType.PRICE_MANIPULATION,
            parameter=param_name,
            test_value=0.01,
            expected_behavior="Accept minimal valid price",
            description="Testing minimal price threshold"
        ))
        
        return tests
    
    def _generate_quantity_tests(
        self,
        param_name: str,
        original_value: any
    ) -> List[LogicTest]:
        """Generate quantity manipulation tests"""
        tests = []
        
        # Negative quantity
        tests.append(LogicTest(
            name=f"Negative Quantity - {param_name}",
            flaw_type=LogicFlawType.QUANTITY_BYPASS,
            parameter=param_name,
            test_value=-1,
            expected_behavior="Reject negative quantity",
            description="Testing if negative quantity is accepted"
        ))
        
        # Zero quantity
        tests.append(LogicTest(
            name=f"Zero Quantity - {param_name}",
            flaw_type=LogicFlawType.QUANTITY_BYPASS,
            parameter=param_name,
            test_value=0,
            expected_behavior="Reject zero quantity",
            description="Testing if zero quantity is accepted"
        ))
        
        # Overflow quantity
        tests.append(LogicTest(
            name=f"Overflow Quantity - {param_name}",
            flaw_type=LogicFlawType.QUANTITY_BYPASS,
            parameter=param_name,
            test_value=999999999,
            expected_behavior="Reject excessive quantity",
            description="Testing quantity overflow handling"
        ))
        
        return tests
    
    def _generate_discount_tests(
        self,
        param_name: str,
        original_value: any
    ) -> List[LogicTest]:
        """Generate discount/coupon abuse tests"""
        tests = []
        
        # Excessive discount
        tests.append(LogicTest(
            name=f"Excessive Discount - {param_name}",
            flaw_type=LogicFlawType.DISCOUNT_ABUSE,
            parameter=param_name,
            test_value=100,
            expected_behavior="Limit discount to valid range",
            description="Testing 100% discount acceptance"
        ))
        
        # Negative discount (credit)
        tests.append(LogicTest(
            name=f"Negative Discount - {param_name}",
            flaw_type=LogicFlawType.DISCOUNT_ABUSE,
            parameter=param_name,
            test_value=-50,
            expected_behavior="Reject negative discount",
            description="Testing negative discount (potential credit)"
        ))
        
        return tests
    
    def _generate_limit_tests(
        self,
        param_name: str,
        original_value: any
    ) -> List[LogicTest]:
        """Generate limit bypass tests"""
        tests = []
        
        # Exceed limit
        tests.append(LogicTest(
            name=f"Exceed Limit - {param_name}",
            flaw_type=LogicFlawType.LIMIT_BYPASS,
            parameter=param_name,
            test_value=999999,
            expected_behavior="Enforce limit",
            description="Testing limit enforcement"
        ))
        
        # Negative limit (bypass)
        tests.append(LogicTest(
            name=f"Negative Limit - {param_name}",
            flaw_type=LogicFlawType.LIMIT_BYPASS,
            parameter=param_name,
            test_value=-1,
            expected_behavior="Reject negative limit",
            description="Testing negative limit bypass"
        ))
        
        return tests
    
    def detect_workflow_endpoints(self, url: str) -> List[str]:
        """
        Detect if URL is part of multi-step workflow
        
        Returns:
            List of workflow-related endpoint patterns
        """
        workflow_patterns = [
            r'/checkout',
            r'/order',
            r'/payment',
            r'/confirm',
            r'/cart',
            r'/step\d+',
            r'/wizard',
            r'/process'
        ]
        
        detected = []
        for pattern in workflow_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                detected.append(pattern)
        
        return detected
    
    def analyze_response_for_logic_flaw(
        self,
        test: LogicTest,
        status_code: int,
        response_body: str,
        original_response: Optional[str] = None
    ) -> Optional[LogicFlaw]:
        """
        Analyze response to detect if logic flaw exists
        
        Args:
            test: The logic test that was performed
            status_code: Response status code
            response_body: Response body
            original_response: Original response for comparison
            
        Returns:
            LogicFlaw if detected, None otherwise
        """
        # Success indicators (bad - means flaw accepted)
        success_indicators = [
            'success', 'confirmed', 'accepted', 'approved',
            'completed', 'processed', 'thank you', 'order placed',
            'payment successful'
        ]
        
        # Rejection indicators (good - means flaw rejected)
        rejection_indicators = [
            'error', 'invalid', 'rejected', 'denied', 'failed',
            'not allowed', 'forbidden', 'must be positive',
            'must be greater', 'exceeds limit'
        ]
        
        response_lower = response_body.lower()
        
        # If status code is 2xx and no rejection indicators
        if 200 <= status_code < 300:
            # Check for rejection in body
            has_rejection = any(ind in response_lower for ind in rejection_indicators)
            has_success = any(ind in response_lower for ind in success_indicators)
            
            if has_success or not has_rejection:
                return LogicFlaw(
                    flaw_type=test.flaw_type,
                    url="",  # Will be filled by caller
                    parameter=test.parameter,
                    test_value=test.test_value,
                    original_value=None,  # Will be filled by caller
                    evidence=f"Status: {status_code}, Test: {test.name}",
                    severity="HIGH",
                    description=test.description
                )
        
        return None
    
    def get_test_summary(self, tests: List[LogicTest]) -> Dict:
        """Get summary of generated tests"""
        summary = {
            'total_tests': len(tests),
            'by_type': {}
        }
        
        for test in tests:
            flaw_type = test.flaw_type.value
            if flaw_type not in summary['by_type']:
                summary['by_type'][flaw_type] = 0
            summary['by_type'][flaw_type] += 1
        
        return summary


# Global instance
business_logic_analyzer = BusinessLogicAnalyzer()