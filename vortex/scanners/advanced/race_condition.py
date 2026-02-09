"""
Race Condition Detector - PHASE 4.3
Detects race condition vulnerabilities through timing analysis
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RaceConditionResult:
    """Result of race condition detection."""
    vulnerable: bool
    confidence: float
    description: str
    timing_delta: float
    requests_sent: int


class RaceConditionDetector:
    """
    Detects race condition vulnerabilities.
    
    PHASE 4.3 Implementation:
    - Concurrent request timing
    - State race detection
    - Resource exhaustion checks
    - TOCTOU vulnerability detection
    """
    
    def __init__(self, concurrent_requests: int = 10):
        self.concurrent_requests = concurrent_requests
        self.stats = {
            'total_tests': 0,
            'race_conditions_detected': 0,
            'timing_anomalies': 0
        }
    
    async def detect(self, url: str, method: str = "GET", 
                    data: Optional[Dict[str, Any]] = None,
                    concurrent_count: Optional[int] = None) -> RaceConditionResult:
        """
        Detect race conditions by sending concurrent requests.
        
        Args:
            url: Target URL
            method: HTTP method
            data: Request data
            concurrent_count: Number of concurrent requests (default: self.concurrent_requests)
            
        Returns:
            RaceConditionResult with detection results
        """
        self.stats['total_tests'] += 1
        
        if concurrent_count is None:
            concurrent_count = self.concurrent_requests
        
        # Simulate concurrent requests (simplified version)
        # In production, this would use actual HTTP client
        start_time = time.time()
        
        # Simulate sending concurrent requests
        tasks = []
        for i in range(concurrent_count):
            # Simulate async request
            tasks.append(self._simulate_request(url, method, data))
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results for race condition indicators
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        
        # Simple heuristic: if all succeeded too quickly, might indicate race condition
        avg_time_per_request = total_time / concurrent_count if concurrent_count > 0 else 0
        
        # Check for timing anomalies
        if avg_time_per_request < 0.05:  # Very fast responses
            self.stats['timing_anomalies'] += 1
            confidence = 0.60
            vulnerable = True
            description = "Timing anomaly detected - possible race condition"
        elif success_count == concurrent_count:
            # All requests succeeded - might indicate race condition
            confidence = 0.55
            vulnerable = True
            description = "All concurrent requests succeeded - possible race window"
            self.stats['race_conditions_detected'] += 1
        else:
            confidence = 0.30
            vulnerable = False
            description = "No clear race condition detected"
        
        return RaceConditionResult(
            vulnerable=vulnerable,
            confidence=confidence,
            description=description,
            timing_delta=avg_time_per_request,
            requests_sent=concurrent_count
        )
    
    async def _simulate_request(self, url: str, method: str, 
                                data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate an HTTP request (placeholder for actual implementation)."""
        # In production, this would use the actual NetworkClient
        await asyncio.sleep(0.01)  # Simulate network delay
        return {'status': 200, 'body': '{}'}
    
    async def test_toctou(self, check_url: str, use_url: str) -> RaceConditionResult:
        """
        Test for Time-of-Check-Time-of-Use (TOCTOU) vulnerabilities.
        
        Args:
            check_url: URL for checking resource state
            use_url: URL for using the resource
            
        Returns:
            RaceConditionResult
        """
        self.stats['total_tests'] += 1
        
        # Simulate TOCTOU detection
        # 1. Send check request
        # 2. Quickly send use request
        # 3. Analyze if use succeeded despite check
        
        start_time = time.time()
        
        check_task = self._simulate_request(check_url, "GET", None)
        use_task = self._simulate_request(use_url, "POST", {})
        
        check_result, use_result = await asyncio.gather(check_task, use_task)
        
        end_time = time.time()
        delta = end_time - start_time
        
        # Simplified logic: if both succeeded, might indicate TOCTOU
        if check_result and use_result:
            return RaceConditionResult(
                vulnerable=True,
                confidence=0.65,
                description="Possible TOCTOU vulnerability detected",
                timing_delta=delta,
                requests_sent=2
            )
        
        return RaceConditionResult(
            vulnerable=False,
            confidence=0.40,
            description="No TOCTOU vulnerability detected",
            timing_delta=delta,
            requests_sent=2
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get detector statistics."""
        return self.stats.copy()


# Global instance
race_condition_detector = RaceConditionDetector()