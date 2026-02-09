#!/usr/bin/env python3
"""
ML-Based Request Timing Module (PHASE 3.3)
Adaptive request timing based on response analysis
NO REAL ML - Using statistical analysis for simplicity
"""

import time
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from statistics import mean, stdev
from collections import deque
import structlog

logger = structlog.get_logger()


@dataclass
class TimingProfile:
    """Request timing profile for a target"""
    target: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_times: deque = field(default_factory=lambda: deque(maxlen=50))
    failed_times: deque = field(default_factory=lambda: deque(maxlen=50))
    rate_limit_detected: bool = False
    last_rate_limit_time: float = 0.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    
    # Adaptive timing parameters
    min_delay: float = 0.5
    max_delay: float = 10.0
    current_delay: float = 1.0
    
    def __post_init__(self):
        """Initialize timing profile"""
        self.response_times = deque(maxlen=100)
        self.success_times = deque(maxlen=50)
        self.failed_times = deque(maxlen=50)


class RequestTimingAnalyzer:
    """
    Adaptive request timing analyzer (PHASE 3.3)
    Uses statistical analysis (NOT real ML) to optimize request delays
    """
    
    def __init__(self):
        self.profiles: Dict[str, TimingProfile] = {}
        
        # Rate limiting detection patterns
        self.RATE_LIMIT_STATUS_CODES = {429, 503, 509}
        self.RATE_LIMIT_KEYWORDS = [
            'rate limit', 'too many requests', 'throttle',
            'slow down', 'quota exceeded', 'retry after'
        ]
        
        # Timing thresholds
        self.VERY_FAST_THRESHOLD = 0.1  # < 100ms = very fast
        self.FAST_THRESHOLD = 0.5  # < 500ms = fast
        self.SLOW_THRESHOLD = 2.0  # > 2s = slow
        self.TIMEOUT_THRESHOLD = 10.0  # > 10s = timeout
        
        # Adaptation rates
        self.SPEEDUP_FACTOR = 0.9  # Decrease delay by 10%
        self.SLOWDOWN_FACTOR = 1.5  # Increase delay by 50%
        self.RATE_LIMIT_BACKOFF = 30.0  # Back off for 30 seconds
        
        logger.info("Request timing analyzer initialized (statistical, not ML)")
    
    def get_profile(self, target: str) -> TimingProfile:
        """Get or create timing profile for target"""
        if target not in self.profiles:
            self.profiles[target] = TimingProfile(target=target)
        return self.profiles[target]
    
    def record_request(
        self,
        target: str,
        response_time: float,
        status_code: int,
        response_body: Optional[str] = None,
        success: bool = True
    ):
        """
        Record request timing and adapt delays
        
        Args:
            target: Target URL
            response_time: Response time in seconds
            status_code: HTTP status code
            response_body: Response body (for rate limit detection)
            success: Whether request was successful
        """
        profile = self.get_profile(target)
        
        # Record timing
        profile.response_times.append(response_time)
        if success:
            profile.success_times.append(response_time)
            profile.consecutive_successes += 1
            profile.consecutive_failures = 0
        else:
            profile.failed_times.append(response_time)
            profile.consecutive_failures += 1
            profile.consecutive_successes = 0
        
        # Detect rate limiting
        rate_limited = self._detect_rate_limiting(
            status_code, response_body, profile
        )
        
        if rate_limited:
            self._handle_rate_limit(profile)
        else:
            # Adapt timing based on response pattern
            self._adapt_timing(profile, response_time, success)
    
    def _detect_rate_limiting(
        self,
        status_code: int,
        response_body: Optional[str],
        profile: TimingProfile
    ) -> bool:
        """
        Detect rate limiting from response
        
        Returns:
            True if rate limiting detected
        """
        # Check status code
        if status_code in self.RATE_LIMIT_STATUS_CODES:
            logger.warning(
                "Rate limit detected (status code)",
                target=profile.target,
                status_code=status_code
            )
            return True
        
        # Check response body for keywords
        if response_body:
            body_lower = response_body.lower()
            for keyword in self.RATE_LIMIT_KEYWORDS:
                if keyword in body_lower:
                    logger.warning(
                        "Rate limit detected (keyword)",
                        target=profile.target,
                        keyword=keyword
                    )
                    return True
        
        # Detect pattern: multiple slow responses
        if len(profile.response_times) >= 5:
            recent_times = list(profile.response_times)[-5:]
            if all(t > self.SLOW_THRESHOLD for t in recent_times):
                logger.warning(
                    "Possible rate limit (slow responses)",
                    target=profile.target,
                    avg_time=mean(recent_times)
                )
                return True
        
        return False
    
    def _handle_rate_limit(self, profile: TimingProfile):
        """Handle rate limiting detection"""
        profile.rate_limit_detected = True
        profile.last_rate_limit_time = time.time()
        
        # Exponential backoff
        profile.current_delay = min(
            profile.current_delay * 2.0,
            self.RATE_LIMIT_BACKOFF
        )
        
        logger.warning(
            "Rate limit handled - backing off",
            target=profile.target,
            new_delay=profile.current_delay
        )
    
    def _adapt_timing(
        self,
        profile: TimingProfile,
        response_time: float,
        success: bool
    ):
        """
        Adapt request timing based on response patterns
        Statistical approach (NOT real ML)
        """
        # Reset rate limit flag after cooldown
        if profile.rate_limit_detected:
            if time.time() - profile.last_rate_limit_time > 60.0:
                profile.rate_limit_detected = False
                logger.info("Rate limit cooldown complete", target=profile.target)
        
        # Don't adapt during rate limit cooldown
        if profile.rate_limit_detected:
            return
        
        # Need enough data for statistical analysis
        if len(profile.response_times) < 5:
            return
        
        # Calculate statistics
        avg_time = mean(profile.response_times)
        
        try:
            std_time = stdev(profile.response_times) if len(profile.response_times) > 1 else 0
        except:
            std_time = 0
        
        # Speed up if consistently fast and successful
        if profile.consecutive_successes >= 5 and avg_time < self.FAST_THRESHOLD:
            profile.current_delay = max(
                profile.min_delay,
                profile.current_delay * self.SPEEDUP_FACTOR
            )
            logger.debug(
                "Speeding up requests",
                target=profile.target,
                new_delay=profile.current_delay,
                avg_time=avg_time
            )
        
        # Slow down if responses are getting slower
        elif avg_time > self.SLOW_THRESHOLD:
            profile.current_delay = min(
                profile.max_delay,
                profile.current_delay * self.SLOWDOWN_FACTOR
            )
            logger.debug(
                "Slowing down requests",
                target=profile.target,
                new_delay=profile.current_delay,
                avg_time=avg_time
            )
        
        # Slow down significantly after failures
        if profile.consecutive_failures >= 3:
            profile.current_delay = min(
                profile.max_delay,
                profile.current_delay * 2.0
            )
            logger.warning(
                "Multiple failures - slowing down",
                target=profile.target,
                new_delay=profile.current_delay,
                failures=profile.consecutive_failures
            )
    
    def get_recommended_delay(self, target: str) -> float:
        """
        Get recommended delay for next request
        
        Args:
            target: Target URL
            
        Returns:
            Recommended delay in seconds
        """
        profile = self.get_profile(target)
        
        # Extra delay if rate limited
        if profile.rate_limit_detected:
            time_since_limit = time.time() - profile.last_rate_limit_time
            if time_since_limit < self.RATE_LIMIT_BACKOFF:
                return self.RATE_LIMIT_BACKOFF - time_since_limit
        
        return profile.current_delay
    
    async def smart_delay(self, target: str):
        """
        Apply smart delay before next request
        
        Args:
            target: Target URL
        """
        delay = self.get_recommended_delay(target)
        
        if delay > 0:
            logger.debug(
                "Applying smart delay",
                target=target,
                delay=delay
            )
            await asyncio.sleep(delay)
    
    def get_statistics(self, target: str) -> Dict:
        """Get timing statistics for target"""
        profile = self.get_profile(target)
        
        stats = {
            'target': target,
            'total_requests': len(profile.response_times),
            'current_delay': profile.current_delay,
            'rate_limit_detected': profile.rate_limit_detected,
            'consecutive_successes': profile.consecutive_successes,
            'consecutive_failures': profile.consecutive_failures
        }
        
        if profile.response_times:
            stats['avg_response_time'] = mean(profile.response_times)
            if len(profile.response_times) > 1:
                stats['std_response_time'] = stdev(profile.response_times)
            stats['min_response_time'] = min(profile.response_times)
            stats['max_response_time'] = max(profile.response_times)
        
        if profile.success_times:
            stats['avg_success_time'] = mean(profile.success_times)
        
        if profile.failed_times:
            stats['avg_failed_time'] = mean(profile.failed_times)
        
        return stats


# Global instance
timing_analyzer = RequestTimingAnalyzer()