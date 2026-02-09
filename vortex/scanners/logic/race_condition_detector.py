#!/usr/bin/env python3
"""
Race Condition Detector Module (PHASE 4.3)
Detects race condition vulnerabilities through concurrent testing
"""

from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog

logger = structlog.get_logger()


class RaceType(Enum):
    """Types of race conditions"""
    TOCTOU = "time_of_check_to_time_of_use"
    DOUBLE_SPEND = "double_spend"
    INVENTORY_MANIPULATION = "inventory_manipulation"
    COUPON_ABUSE = "coupon_abuse"
    RATE_LIMIT_BYPASS = "rate_limit_bypass"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_CORRUPTION = "data_corruption"


class ConcurrencyMode(Enum):
    """Concurrency testing modes"""
    ASYNC = "async"  # asyncio-based
    THREADED = "threaded"  # thread-based
    BURST = "burst"  # rapid fire


@dataclass
class RaceTest:
    """Race condition test case"""
    name: str
    url: str
    method: str
    race_type: RaceType
    concurrent_requests: int
    delay_between_ms: float
    params: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict] = None
    description: str = ""


@dataclass
class RaceResult:
    """Result of race condition test"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    status_codes: List[int]
    responses: List[Dict[str, Any]]
    avg_response_time: float
    race_window_ms: float
    detected_vulnerability: bool = False
    vulnerability_evidence: str = ""


@dataclass
class RaceVulnerability:
    """Detected race condition vulnerability"""
    vuln_type: RaceType
    endpoint: str
    method: str
    evidence: str
    impact: str
    severity: str = "HIGH"
    exploitation_rate: float = 0.0  # Success rate (0.0 to 1.0)
    race_window_ms: float = 0.0


class RaceConditionDetector:
    """
    Race Condition Detection System (PHASE 4.3)
    
    Detects:
    - TOCTOU vulnerabilities
    - Double-spend attacks
    - Inventory manipulation
    - Coupon/promo code abuse
    - Rate limit bypass via race conditions
    - Resource exhaustion
    - Privilege escalation via race
    - Data corruption
    """
    
    def __init__(self, max_concurrent: int = 50):
        """
        Initialize race condition detector
        
        Args:
            max_concurrent: Maximum concurrent requests
        """
        self.max_concurrent = max_concurrent
        self.default_concurrency = 10
        self.default_timeout = 30.0
        
        # Race window thresholds (in milliseconds)
        self.RACE_WINDOW_THRESHOLDS = {
            RaceType.TOCTOU: 100,  # 100ms window
            RaceType.DOUBLE_SPEND: 50,  # 50ms for payment
            RaceType.INVENTORY_MANIPULATION: 200,  # 200ms for inventory
            RaceType.COUPON_ABUSE: 150,  # 150ms for coupon
            RaceType.RATE_LIMIT_BYPASS: 500,  # 500ms for rate limit
            RaceType.RESOURCE_EXHAUSTION: 100,  # 100ms for resources
            RaceType.PRIVILEGE_ESCALATION: 100,  # 100ms for privilege
            RaceType.DATA_CORRUPTION: 50,  # 50ms for data integrity
        }
        
        logger.info("Race Condition Detector initialized", max_concurrent=max_concurrent)
    
    def generate_race_tests(
        self,
        url: str,
        method: str,
        params: Dict[str, Any]
    ) -> List[RaceTest]:
        """
        Generate race condition test cases
        
        Args:
            url: Target URL
            method: HTTP method
            params: Request parameters
            
        Returns:
            List of race condition tests
        """
        tests = []
        
        # Detect vulnerable parameters
        vulnerable_params = self._detect_vulnerable_params(params)
        
        # Generate tests for each race type
        if 'balance' in vulnerable_params or 'amount' in vulnerable_params:
            tests.append(self._create_double_spend_test(url, method, params))
        
        if 'quantity' in vulnerable_params or 'stock' in vulnerable_params:
            tests.append(self._create_inventory_test(url, method, params))
        
        if 'coupon' in vulnerable_params or 'promo' in vulnerable_params or 'discount' in vulnerable_params:
            tests.append(self._create_coupon_abuse_test(url, method, params))
        
        # Generic TOCTOU test for checkout/payment endpoints
        if any(keyword in url.lower() for keyword in ['checkout', 'payment', 'order', 'purchase']):
            tests.append(self._create_toctou_test(url, method, params))
        
        # Rate limit bypass test (always applicable)
        tests.append(self._create_rate_limit_test(url, method, params))
        
        logger.info(
            f"Generated {len(tests)} race condition tests",
            url=url,
            vulnerable_params=list(vulnerable_params)
        )
        
        return tests
    
    def _detect_vulnerable_params(self, params: Dict[str, Any]) -> set:
        """Detect parameters vulnerable to race conditions"""
        vulnerable = set()
        
        vulnerable_keywords = [
            'balance', 'amount', 'quantity', 'qty', 'stock',
            'coupon', 'promo', 'discount', 'credit', 'points',
            'limit', 'count', 'total', 'price'
        ]
        
        for key in params.keys():
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in vulnerable_keywords):
                vulnerable.add(key_lower)
        
        return vulnerable
    
    def _create_double_spend_test(
        self,
        url: str,
        method: str,
        params: Dict[str, Any]
    ) -> RaceTest:
        """Create double-spend attack test"""
        return RaceTest(
            name="Double Spend Attack",
            url=url,
            method=method,
            race_type=RaceType.DOUBLE_SPEND,
            concurrent_requests=20,
            delay_between_ms=0,  # No delay - burst mode
            params=params,
            description="Attempt to spend same balance/credit multiple times"
        )
    
    def _create_inventory_test(
        self,
        url: str,
        method: str,
        params: Dict[str, Any]
    ) -> RaceTest:
        """Create inventory manipulation test"""
        # Try to buy more than available
        test_params = params.copy()
        if 'quantity' in test_params:
            test_params['quantity'] = 999
        if 'qty' in test_params:
            test_params['qty'] = 999
        
        return RaceTest(
            name="Inventory Manipulation",
            url=url,
            method=method,
            race_type=RaceType.INVENTORY_MANIPULATION,
            concurrent_requests=15,
            delay_between_ms=5,
            params=test_params,
            description="Bypass inventory limits via concurrent requests"
        )
    
    def _create_coupon_abuse_test(
        self,
        url: str,
        method: str,
        params: Dict[str, Any]
    ) -> RaceTest:
        """Create coupon abuse test"""
        return RaceTest(
            name="Coupon/Promo Code Abuse",
            url=url,
            method=method,
            race_type=RaceType.COUPON_ABUSE,
            concurrent_requests=10,
            delay_between_ms=10,
            params=params,
            description="Use single-use coupon multiple times via race"
        )
    
    def _create_toctou_test(
        self,
        url: str,
        method: str,
        params: Dict[str, Any]
    ) -> RaceTest:
        """Create TOCTOU (Time-of-Check to Time-of-Use) test"""
        return RaceTest(
            name="TOCTOU Attack",
            url=url,
            method=method,
            race_type=RaceType.TOCTOU,
            concurrent_requests=12,
            delay_between_ms=8,
            params=params,
            description="Exploit time gap between check and use"
        )
    
    def _create_rate_limit_test(
        self,
        url: str,
        method: str,
        params: Dict[str, Any]
    ) -> RaceTest:
        """Create rate limit bypass test"""
        return RaceTest(
            name="Rate Limit Bypass",
            url=url,
            method=method,
            race_type=RaceType.RATE_LIMIT_BYPASS,
            concurrent_requests=30,
            delay_between_ms=0,
            params=params,
            description="Bypass rate limiting via concurrent requests"
        )
    
    async def execute_race_test_async(
        self,
        test: RaceTest,
        request_func: Callable
    ) -> RaceResult:
        """
        Execute race condition test (async mode)
        
        Args:
            test: Race test to execute
            request_func: Async function to make requests
            
        Returns:
            Race test result
        """
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for i in range(test.concurrent_requests):
            if test.delay_between_ms > 0:
                await asyncio.sleep(test.delay_between_ms / 1000)
            
            task = asyncio.create_task(
                self._make_request_async(request_func, test, i)
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate race window
        race_window = (time.time() - start_time) * 1000  # Convert to ms
        
        return self._analyze_results(test, results, race_window)
    
    def execute_race_test_threaded(
        self,
        test: RaceTest,
        request_func: Callable
    ) -> RaceResult:
        """
        Execute race condition test (threaded mode)
        
        Args:
            test: Race test to execute
            request_func: Function to make requests
            
        Returns:
            Race test result
        """
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=test.concurrent_requests) as executor:
            futures = []
            
            for i in range(test.concurrent_requests):
                if test.delay_between_ms > 0:
                    time.sleep(test.delay_between_ms / 1000)
                
                future = executor.submit(
                    self._make_request_sync,
                    request_func,
                    test,
                    i
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.default_timeout)
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e), 'status_code': 0})
        
        race_window = (time.time() - start_time) * 1000
        
        return self._analyze_results(test, results, race_window)
    
    async def _make_request_async(
        self,
        request_func: Callable,
        test: RaceTest,
        index: int
    ) -> Dict[str, Any]:
        """Make a single async request"""
        try:
            start = time.time()
            response = await request_func(
                url=test.url,
                method=test.method,
                params=test.params,
                headers=test.headers,
                body=test.body
            )
            elapsed = (time.time() - start) * 1000
            
            return {
                'index': index,
                'status_code': response.get('status_code', 0),
                'response_time': elapsed,
                'body': response.get('body', ''),
                'headers': response.get('headers', {}),
                'success': 200 <= response.get('status_code', 0) < 300
            }
        except Exception as e:
            return {
                'index': index,
                'error': str(e),
                'status_code': 0,
                'success': False
            }
    
    def _make_request_sync(
        self,
        request_func: Callable,
        test: RaceTest,
        index: int
    ) -> Dict[str, Any]:
        """Make a single sync request"""
        try:
            start = time.time()
            response = request_func(
                url=test.url,
                method=test.method,
                params=test.params,
                headers=test.headers,
                body=test.body
            )
            elapsed = (time.time() - start) * 1000
            
            return {
                'index': index,
                'status_code': response.get('status_code', 0),
                'response_time': elapsed,
                'body': response.get('body', ''),
                'headers': response.get('headers', {}),
                'success': 200 <= response.get('status_code', 0) < 300
            }
        except Exception as e:
            return {
                'index': index,
                'error': str(e),
                'status_code': 0,
                'success': False
            }
    
    def _analyze_results(
        self,
        test: RaceTest,
        results: List[Dict[str, Any]],
        race_window_ms: float
    ) -> RaceResult:
        """Analyze race test results"""
        # Extract metrics
        response_times = [r.get('response_time', 0) for r in results if 'response_time' in r]
        status_codes = [r.get('status_code', 0) for r in results]
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Detect vulnerability
        detected, evidence = self._detect_race_vulnerability(
            test,
            results,
            race_window_ms
        )
        
        return RaceResult(
            test_name=test.name,
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            response_times=response_times,
            status_codes=status_codes,
            responses=results,
            avg_response_time=avg_response_time,
            race_window_ms=race_window_ms,
            detected_vulnerability=detected,
            vulnerability_evidence=evidence
        )
    
    def _detect_race_vulnerability(
        self,
        test: RaceTest,
        results: List[Dict[str, Any]],
        race_window_ms: float
    ) -> Tuple[bool, str]:
        """
        Detect if race condition vulnerability exists
        
        Returns:
            (detected, evidence)
        """
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)
        success_rate = successful / total if total > 0 else 0
        
        # Check race window
        expected_window = self.RACE_WINDOW_THRESHOLDS.get(test.race_type, 100)
        
        # Vulnerability indicators
        if test.race_type == RaceType.DOUBLE_SPEND:
            # Multiple successful transactions = vulnerability
            if successful > 1:
                return True, f"Multiple successful charges: {successful}/{total}"
        
        elif test.race_type == RaceType.INVENTORY_MANIPULATION:
            # High success rate on over-quantity = vulnerability
            if success_rate > 0.5:
                return True, f"High inventory bypass rate: {success_rate:.1%}"
        
        elif test.race_type == RaceType.COUPON_ABUSE:
            # Multiple coupon uses = vulnerability
            if successful > 1:
                return True, f"Coupon used {successful} times (should be 1)"
        
        elif test.race_type == RaceType.RATE_LIMIT_BYPASS:
            # High success rate = weak rate limiting
            if success_rate > 0.7:
                return True, f"Rate limit bypassed: {success_rate:.1%} success"
        
        elif test.race_type == RaceType.TOCTOU:
            # Check for inconsistent states
            unique_responses = len(set(str(r.get('body', '')) for r in results))
            if unique_responses < len(results) * 0.5:
                return True, f"Inconsistent states detected: {unique_responses} unique responses"
        
        # Check race window
        if race_window_ms < expected_window:
            return True, f"Narrow race window exploited: {race_window_ms:.1f}ms"
        
        return False, "No race condition detected"
    
    def create_vulnerability_report(
        self,
        test: RaceTest,
        result: RaceResult
    ) -> Optional[RaceVulnerability]:
        """Create vulnerability report if detected"""
        if not result.detected_vulnerability:
            return None
        
        exploitation_rate = result.successful_requests / result.total_requests
        
        # Determine severity
        if exploitation_rate > 0.8:
            severity = "CRITICAL"
        elif exploitation_rate > 0.5:
            severity = "HIGH"
        else:
            severity = "MEDIUM"
        
        # Determine impact
        impact_map = {
            RaceType.DOUBLE_SPEND: "Financial loss, payment fraud",
            RaceType.INVENTORY_MANIPULATION: "Stock depletion, business logic bypass",
            RaceType.COUPON_ABUSE: "Revenue loss, promotion abuse",
            RaceType.RATE_LIMIT_BYPASS: "Resource exhaustion, DoS potential",
            RaceType.TOCTOU: "Data inconsistency, authorization bypass",
            RaceType.RESOURCE_EXHAUSTION: "Service degradation, DoS",
            RaceType.PRIVILEGE_ESCALATION: "Unauthorized access, data breach",
            RaceType.DATA_CORRUPTION: "Data integrity loss, system instability",
        }
        
        return RaceVulnerability(
            vuln_type=test.race_type,
            endpoint=test.url,
            method=test.method,
            evidence=result.vulnerability_evidence,
            impact=impact_map.get(test.race_type, "Business logic bypass"),
            severity=severity,
            exploitation_rate=exploitation_rate,
            race_window_ms=result.race_window_ms
        )
    
    def get_test_summary(self, tests: List[RaceTest]) -> Dict:
        """Get summary of race condition tests"""
        summary = {
            'total_tests': len(tests),
            'by_type': {},
            'total_requests': 0,
            'avg_concurrent': 0
        }
        
        for test in tests:
            race_type = test.race_type.value
            summary['by_type'][race_type] = summary['by_type'].get(race_type, 0) + 1
            summary['total_requests'] += test.concurrent_requests
        
        summary['avg_concurrent'] = (
            summary['total_requests'] / len(tests) if tests else 0
        )
        
        return summary


# Global instance
race_detector = RaceConditionDetector()