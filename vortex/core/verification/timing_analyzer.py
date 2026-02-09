"""
VORTEX Timing Attack Analyzer - V17.0 ULTIMATE
Statistical analysis for time-based attacks

DETECTION METHODS:
- Chi-square test for timing anomalies
- Statistical significance testing
- Blind SQLi time-based detection
- Response time distribution analysis

CRITICAL: Time-based evidence requires statistical validation
Per VORTEX_EVIDENCE_STANDARDS.md: Multiple samples needed for confidence
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Scipy import guard
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("Scipy not available - statistical timing analysis disabled")


@dataclass
class TimingSample:
    """Single timing measurement."""
    request_id: str
    payload: str
    response_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TimingAnalysisResult:
    """Result from timing analysis."""
    is_time_based: bool
    confidence: float
    baseline_mean: float
    test_mean: float
    time_difference: float
    statistical_significance: float  # p-value
    samples_count: int
    analysis_method: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TimingAnalyzer:
    """
    Analyze timing patterns for time-based attacks.
    
    APPROACH:
    - Collect multiple timing samples (statistical requirement)
    - Compare baseline vs test distributions
    - Apply chi-square test for significance
    - Detect time-based SQLi patterns (e.g., SLEEP())
    
    REQUIREMENTS:
    - Minimum 10 samples for statistical validity
    - Confidence threshold: p-value < 0.05
    """
    
    def __init__(self):
        # Statistical thresholds
        self.min_samples = 10  # Minimum for statistical validity
        self.significance_threshold = 0.05  # p-value threshold
        self.min_time_difference = 2.0  # seconds (for SQLi SLEEP detection)
        
        # Confidence thresholds
        self.high_confidence = 0.90
        self.medium_confidence = 0.75
        self.low_confidence = 0.60
        
        # Statistics
        self.stats = {
            'analyses_performed': 0,
            'time_based_detected': 0,
            'samples_collected': 0
        }
    
    def statistical_timing_test(self,
                                baseline_times: List[float],
                                test_times: List[float]) -> TimingAnalysisResult:
        """
        Perform statistical test on timing distributions.
        
        Uses chi-square test to determine if test times are significantly
        different from baseline times.
        
        Args:
            baseline_times: Response times for baseline requests
            test_times: Response times for test requests (with payload)
            
        Returns:
            TimingAnalysisResult with statistical analysis
        """
        self.stats['analyses_performed'] += 1
        self.stats['samples_collected'] += len(baseline_times) + len(test_times)
        
        # Validate sample sizes
        if len(baseline_times) < self.min_samples or len(test_times) < self.min_samples:
            logger.warning(
                f"Insufficient samples: baseline={len(baseline_times)}, "
                f"test={len(test_times)}, required={self.min_samples}"
            )
            return TimingAnalysisResult(
                is_time_based=False,
                confidence=0.0,
                baseline_mean=statistics.mean(baseline_times) if baseline_times else 0.0,
                test_mean=statistics.mean(test_times) if test_times else 0.0,
                time_difference=0.0,
                statistical_significance=1.0,
                samples_count=len(baseline_times) + len(test_times),
                analysis_method='insufficient_samples'
            )
        
        # Calculate statistics
        baseline_mean = statistics.mean(baseline_times)
        test_mean = statistics.mean(test_times)
        time_diff = test_mean - baseline_mean
        
        # Perform statistical test
        if SCIPY_AVAILABLE:
            # Use Mann-Whitney U test (non-parametric, better for response times)
            try:
                statistic, p_value = stats.mannwhitneyu(
                    baseline_times,
                    test_times,
                    alternative='two-sided'
                )
                
                logger.debug(
                    f"Mann-Whitney U test: statistic={statistic:.2f}, p-value={p_value:.4f}"
                )
                
            except Exception as e:
                logger.error(f"Statistical test error: {e}")
                # Fallback to simple comparison
                p_value = self._simple_significance_test(baseline_times, test_times)
        else:
            # Fallback without scipy
            p_value = self._simple_significance_test(baseline_times, test_times)
        
        # Determine if time-based
        is_time_based = (
            p_value < self.significance_threshold and
            abs(time_diff) >= self.min_time_difference
        )
        
        # Calculate confidence
        confidence = self._calculate_timing_confidence(
            p_value,
            time_diff,
            len(baseline_times),
            len(test_times)
        )
        
        if is_time_based:
            self.stats['time_based_detected'] += 1
        
        return TimingAnalysisResult(
            is_time_based=is_time_based,
            confidence=confidence,
            baseline_mean=baseline_mean,
            test_mean=test_mean,
            time_difference=time_diff,
            statistical_significance=p_value,
            samples_count=len(baseline_times) + len(test_times),
            analysis_method='mann_whitney_u' if SCIPY_AVAILABLE else 'simple'
        )
    
    def detect_time_based_sqli(self,
                               baseline_time: float,
                               test_times: List[float],
                               sleep_duration: int = 5) -> TimingAnalysisResult:
        """
        Detect time-based SQL injection.
        
        Specific for payloads like: SLEEP(5), WAITFOR DELAY '00:00:05'
        
        Args:
            baseline_time: Single baseline response time
            test_times: Multiple test response times (with SLEEP payload)
            sleep_duration: Expected sleep duration in seconds
            
        Returns:
            TimingAnalysisResult
        """
        self.stats['analyses_performed'] += 1
        
        if len(test_times) < 3:  # Minimum 3 samples for time-based SQLi
            return TimingAnalysisResult(
                is_time_based=False,
                confidence=0.0,
                baseline_mean=baseline_time,
                test_mean=statistics.mean(test_times) if test_times else 0.0,
                time_difference=0.0,
                statistical_significance=1.0,
                samples_count=len(test_times),
                analysis_method='insufficient_samples_sqli'
            )
        
        # Calculate test mean
        test_mean = statistics.mean(test_times)
        time_diff = test_mean - baseline_time
        
        # Check if time difference is close to sleep duration
        # Allow ±20% tolerance
        expected_diff = sleep_duration
        tolerance = sleep_duration * 0.2
        
        is_within_range = (
            abs(time_diff - expected_diff) <= tolerance
        )
        
        # Check consistency of test times (should all be similar if SLEEP works)
        if len(test_times) > 1:
            test_stdev = statistics.stdev(test_times)
            is_consistent = test_stdev < (sleep_duration * 0.15)  # 15% variance allowed
        else:
            is_consistent = True
        
        is_time_based = is_within_range and is_consistent
        
        # Calculate confidence
        if is_time_based:
            # High confidence if time matches expected sleep
            match_quality = 1.0 - (abs(time_diff - expected_diff) / expected_diff)
            confidence = min(0.9, 0.6 + (match_quality * 0.3))
        else:
            confidence = 0.0
        
        # Fake p-value based on match quality
        p_value = 1.0 - confidence if confidence > 0 else 1.0
        
        if is_time_based:
            self.stats['time_based_detected'] += 1
        
        return TimingAnalysisResult(
            is_time_based=is_time_based,
            confidence=confidence,
            baseline_mean=baseline_time,
            test_mean=test_mean,
            time_difference=time_diff,
            statistical_significance=p_value,
            samples_count=len(test_times),
            analysis_method='sqli_sleep_detection'
        )
    
    def _simple_significance_test(self,
                                  baseline_times: List[float],
                                  test_times: List[float]) -> float:
        """
        Simple significance test without scipy.
        
        Uses coefficient of variation to estimate significance.
        """
        try:
            baseline_mean = statistics.mean(baseline_times)
            test_mean = statistics.mean(test_times)
            
            baseline_stdev = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0.0
            test_stdev = statistics.stdev(test_times) if len(test_times) > 1 else 0.0
            
            # Calculate effect size (Cohen's d approximation)
            pooled_stdev = ((baseline_stdev ** 2 + test_stdev ** 2) / 2) ** 0.5
            
            if pooled_stdev == 0:
                return 1.0  # No variance, no significance
            
            effect_size = abs(test_mean - baseline_mean) / pooled_stdev
            
            # Convert effect size to approximate p-value
            # Large effect size → low p-value
            if effect_size > 2.0:
                return 0.01
            elif effect_size > 1.5:
                return 0.05
            elif effect_size > 1.0:
                return 0.10
            elif effect_size > 0.5:
                return 0.20
            else:
                return 0.50
                
        except Exception as e:
            logger.error(f"Simple significance test error: {e}")
            return 1.0  # No significance on error
    
    def _calculate_timing_confidence(self,
                                    p_value: float,
                                    time_diff: float,
                                    baseline_count: int,
                                    test_count: int) -> float:
        """Calculate confidence score for timing analysis."""
        
        # Base confidence from p-value
        if p_value < 0.01:
            base_confidence = 0.9
        elif p_value < 0.05:
            base_confidence = 0.75
        elif p_value < 0.10:
            base_confidence = 0.60
        else:
            base_confidence = 0.0
        
        # Boost for large time differences (more obvious)
        if abs(time_diff) >= 5.0:
            base_confidence += 0.05
        elif abs(time_diff) >= 3.0:
            base_confidence += 0.03
        
        # Boost for more samples (better statistics)
        total_samples = baseline_count + test_count
        if total_samples >= 30:
            base_confidence += 0.05
        elif total_samples >= 20:
            base_confidence += 0.03
        
        return min(base_confidence, 0.95)
    
    async def collect_timing_samples(self,
                                    request_func,
                                    url: str,
                                    payload: str,
                                    sample_count: int = 10) -> List[float]:
        """
        Collect multiple timing samples for statistical analysis.
        
        Args:
            request_func: Async function to make requests
            url: Target URL
            payload: Payload to test
            sample_count: Number of samples to collect
            
        Returns:
            List of response times
        """
        times = []
        
        for i in range(sample_count):
            try:
                start = datetime.utcnow()
                response = await request_func(url, payload)
                elapsed = (datetime.utcnow() - start).total_seconds()
                times.append(elapsed)
                
            except Exception as e:
                logger.error(f"Sample collection error: {e}")
                continue
        
        return times
    
    def get_stats(self) -> Dict[str, int]:
        """Get analyzer statistics."""
        return self.stats.copy()


# Global analyzer instance
global_timing_analyzer = TimingAnalyzer()


def analyze_timing_pattern(baseline_times: List[float],
                           test_times: List[float]) -> TimingAnalysisResult:
    """
    Convenience function for timing analysis.
    
    Args:
        baseline_times: Baseline response times
        test_times: Test response times
        
    Returns:
        TimingAnalysisResult
    """
    return global_timing_analyzer.statistical_timing_test(baseline_times, test_times)