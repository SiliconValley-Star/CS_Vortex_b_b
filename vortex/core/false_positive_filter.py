"""
VORTEX False Positive Filter - V17.0 ULTIMATE
CDN/WAF false positive detection and filtering

Per .clinerules:
- Detect CDN switching, load balancing, cache variations
- Filter non-security behavioral changes
- Acknowledge uncertainty in causation
- Prevent false positives from reaching SUBMIT_READY

FEATURES:
- CDN behavior detection
- WAF response pattern recognition
- A/B testing detection
- Cache variation identification
- Infrastructure change filtering
"""

import re
import logging
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FalsePositiveIndicator:
    """False positive indicator with confidence."""
    indicator_type: str
    confidence: float  # 0.0-1.0
    description: str
    evidence: List[str]
    
    # Context
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FilterResult:
    """False positive filtering result."""
    is_false_positive: bool
    confidence: float  # Confidence it's a false positive
    indicators: List[FalsePositiveIndicator]
    
    # Classification
    likely_cause: Optional[str] = None
    recommendation: str = "proceed"  # 'proceed', 'investigate', 'reject'
    
    # Uncertainty acknowledgment (per .clinerules)
    causation_certainty: str = "UNCERTAIN"  # System can't definitively determine remote causes
    requires_human_analysis: bool = False


class CDNDetector:
    """
    CDN and load balancer behavior detector.
    
    Detects infrastructure-level changes that aren't security issues.
    """
    
    # Common CDN/Load Balancer headers
    CDN_HEADERS = {
        'cf-ray', 'cf-cache-status', 'cloudflare',  # Cloudflare
        'x-amz-cf-id', 'x-amz-cf-pop',  # AWS CloudFront
        'x-cache', 'x-cache-hits', 'age',  # Generic cache
        'x-served-by', 'x-backend',  # Load balancer
        'via',  # Proxy/CDN
    }
    
    # CDN/Cache status values
    CACHE_INDICATORS = [
        'HIT', 'MISS', 'EXPIRED', 'STALE', 'UPDATING',
        'REVALIDATED', 'BYPASS', 'DYNAMIC'
    ]
    
    def detect_cdn_switch(self, 
                         original_headers: Dict[str, str],
                         replay_headers: Dict[str, str]) -> Optional[FalsePositiveIndicator]:
        """
        Detect CDN node switching.
        
        Args:
            original_headers: Original response headers
            replay_headers: Replay response headers
            
        Returns:
            False positive indicator if CDN switch detected
        """
        evidence = []
        
        # Normalize header keys
        orig_lower = {k.lower(): v for k, v in original_headers.items()}
        replay_lower = {k.lower(): v for k, v in replay_headers.items()}
        
        # Check for CDN header changes
        for cdn_header in self.CDN_HEADERS:
            if cdn_header in orig_lower and cdn_header in replay_lower:
                if orig_lower[cdn_header] != replay_lower[cdn_header]:
                    evidence.append(f"{cdn_header}: {orig_lower[cdn_header]} → {replay_lower[cdn_header]}")
        
        # Check cache status changes
        for header in ['x-cache', 'cf-cache-status']:
            if header in orig_lower and header in replay_lower:
                orig_val = orig_lower[header].upper()
                replay_val = replay_lower[header].upper()
                
                if orig_val != replay_val:
                    # Check if both are valid cache statuses
                    if any(status in orig_val for status in self.CACHE_INDICATORS):
                        evidence.append(f"Cache status change: {orig_val} → {replay_val}")
        
        if evidence:
            return FalsePositiveIndicator(
                indicator_type="CDN_SWITCH",
                confidence=0.7,  # High confidence it's infrastructure
                description="CDN node switching or cache behavior change detected",
                evidence=evidence
            )
        
        return None
    
    def detect_load_balancer_rotation(self,
                                     original_headers: Dict[str, str],
                                     replay_headers: Dict[str, str]) -> Optional[FalsePositiveIndicator]:
        """Detect load balancer backend rotation."""
        evidence = []
        
        orig_lower = {k.lower(): v for k, v in original_headers.items()}
        replay_lower = {k.lower(): v for k, v in replay_headers.items()}
        
        # Server changes
        if 'server' in orig_lower and 'server' in replay_lower:
            if orig_lower['server'] != replay_lower['server']:
                evidence.append(f"Server header change: {orig_lower['server']} → {replay_lower['server']}")
        
        # Backend identifiers
        backend_headers = ['x-served-by', 'x-backend', 'x-upstream']
        for header in backend_headers:
            if header in orig_lower and header in replay_lower:
                if orig_lower[header] != replay_lower[header]:
                    evidence.append(f"{header} change: load balancer rotation")
        
        if evidence:
            return FalsePositiveIndicator(
                indicator_type="LOAD_BALANCER_ROTATION",
                confidence=0.8,
                description="Load balancer backend rotation detected",
                evidence=evidence
            )
        
        return None


class WAFDetector:
    """
    WAF (Web Application Firewall) response detector.
    
    Identifies WAF responses that may vary based on request patterns.
    """
    
    # Common WAF signatures
    WAF_SIGNATURES = {
        'cloudflare': ['cloudflare', 'cf-ray', 'attention required'],
        'aws_waf': ['aws-waf', 'x-amzn-requestid', 'x-amzn-waf'],
        'akamai': ['akamai', 'reference #'],
        'imperva': ['imperva', 'incapsula'],
        'f5': ['f5', 'bigip'],
        'mod_security': ['mod_security', 'modsecurity'],
    }
    
    # WAF block page indicators
    BLOCK_INDICATORS = [
        'access denied',
        'forbidden',
        'blocked',
        'security policy',
        'firewall',
        'request id:',
        'reference #',
    ]
    
    def detect_waf_response(self, 
                           response_body: str,
                           response_headers: Dict[str, str],
                           status_code: int) -> Optional[FalsePositiveIndicator]:
        """
        Detect WAF-generated response.
        
        Args:
            response_body: Response body text
            response_headers: Response headers
            status_code: HTTP status code
            
        Returns:
            False positive indicator if WAF detected
        """
        evidence = []
        waf_type = "UNKNOWN"
        
        body_lower = response_body.lower()
        headers_lower = {k.lower(): v.lower() for k, v in response_headers.items()}
        
        # Check for WAF signatures in headers
        for waf_name, signatures in self.WAF_SIGNATURES.items():
            for signature in signatures:
                for header_value in headers_lower.values():
                    if signature in header_value:
                        evidence.append(f"{waf_name.upper()} signature in headers")
                        waf_type = waf_name.upper()
        
        # Check for block indicators in body
        for indicator in self.BLOCK_INDICATORS:
            if indicator in body_lower:
                evidence.append(f"Block indicator: '{indicator}'")
        
        # WAF typically returns 403, 406, 429
        if status_code in [403, 406, 429] and evidence:
            evidence.append(f"WAF-typical status code: {status_code}")
        
        if evidence:
            return FalsePositiveIndicator(
                indicator_type="WAF_RESPONSE",
                confidence=0.6,  # Moderate confidence
                description=f"WAF response detected ({waf_type})",
                evidence=evidence
            )
        
        return None
    
    def detect_rate_limiting(self,
                           status_code: int,
                           response_headers: Dict[str, str],
                           response_body: str) -> Optional[FalsePositiveIndicator]:
        """Detect rate limiting responses."""
        evidence = []
        
        # Status code 429 (Too Many Requests)
        if status_code == 429:
            evidence.append("HTTP 429 Too Many Requests")
        
        # Rate limit headers
        headers_lower = {k.lower(): v for k, v in response_headers.items()}
        rate_limit_headers = ['x-ratelimit-', 'ratelimit-', 'retry-after']
        
        for header in headers_lower.keys():
            if any(rl_header in header for rl_header in rate_limit_headers):
                evidence.append(f"Rate limit header: {header}")
        
        # Rate limit in body
        body_lower = response_body.lower()
        if 'rate limit' in body_lower or 'too many requests' in body_lower:
            evidence.append("Rate limit message in body")
        
        if evidence:
            return FalsePositiveIndicator(
                indicator_type="RATE_LIMITING",
                confidence=0.9,  # Very high confidence
                description="Rate limiting detected",
                evidence=evidence
            )
        
        return None


class DynamicContentDetector:
    """
    Dynamic content and A/B testing detector.
    
    Identifies legitimate dynamic content changes.
    """
    
    # Dynamic content indicators
    DYNAMIC_PATTERNS = [
        r'timestamp["\']?\s*:\s*["\']?\d+',
        r'nonce["\']?\s*:\s*["\'][a-f0-9]{32}',
        r'csrf[_-]?token',
        r'session[_-]?id',
        r'_ga\d*=',  # Google Analytics
        r'fbclid=',  # Facebook click ID
    ]
    
    def detect_dynamic_content(self,
                               original_body: str,
                               replay_body: str) -> Optional[FalsePositiveIndicator]:
        """Detect legitimate dynamic content changes."""
        evidence = []
        
        # Check for dynamic patterns
        for pattern in self.DYNAMIC_PATTERNS:
            orig_matches = re.findall(pattern, original_body)
            replay_matches = re.findall(pattern, replay_body)
            
            if orig_matches and replay_matches:
                if orig_matches != replay_matches:
                    evidence.append(f"Dynamic content: {pattern[:30]}...")
        
        # Timestamp differences
        if 'timestamp' in original_body.lower() and 'timestamp' in replay_body.lower():
            evidence.append("Timestamp variations detected")
        
        if evidence:
            return FalsePositiveIndicator(
                indicator_type="DYNAMIC_CONTENT",
                confidence=0.5,  # Moderate confidence
                description="Legitimate dynamic content changes",
                evidence=evidence
            )
        
        return None


class FalsePositiveFilter:
    """
    False positive filtering engine.
    
    RESPONSIBILITIES:
    - Detect CDN/infrastructure changes
    - Identify WAF responses
    - Filter non-security behavioral changes
    - Acknowledge causation uncertainty
    
    Per .clinerules VORTEX_EVIDENCE_STANDARDS.md:
    "Behavioral differences can result from:
    - SECURITY-RELEVANT: Backend errors, logic changes, validation failures
    - NON-SECURITY: CDN switching, load balancing, cache variations, A/B testing
    
    System CANNOT definitively distinguish causes remotely."
    """
    
    def __init__(self):
        self.cdn_detector = CDNDetector()
        self.waf_detector = WAFDetector()
        self.dynamic_detector = DynamicContentDetector()
        
        # Statistics
        self.total_checks = 0
        self.false_positives_detected = 0
        
        logger.info("False Positive Filter initialized")
    
    def filter_finding(self,
                      original_response: Dict[str, Any],
                      replay_response: Dict[str, Any]) -> FilterResult:
        """
        Filter finding for false positives.
        
        Args:
            original_response: Original response data
            replay_response: Replay response data
            
        Returns:
            Filter result with uncertainty acknowledgment
        """
        self.total_checks += 1
        
        indicators: List[FalsePositiveIndicator] = []
        
        # Extract data
        orig_headers = original_response.get('headers', {})
        replay_headers = replay_response.get('headers', {})
        orig_body = original_response.get('body', '')
        replay_body = replay_response.get('body', '')
        orig_status = original_response.get('status_code', 200)
        replay_status = replay_response.get('status_code', 200)
        
        # Run all detectors
        
        # CDN detection
        cdn_indicator = self.cdn_detector.detect_cdn_switch(orig_headers, replay_headers)
        if cdn_indicator:
            indicators.append(cdn_indicator)
        
        lb_indicator = self.cdn_detector.detect_load_balancer_rotation(orig_headers, replay_headers)
        if lb_indicator:
            indicators.append(lb_indicator)
        
        # WAF detection
        waf_indicator = self.waf_detector.detect_waf_response(replay_body, replay_headers, replay_status)
        if waf_indicator:
            indicators.append(waf_indicator)
        
        rate_limit = self.waf_detector.detect_rate_limiting(replay_status, replay_headers, replay_body)
        if rate_limit:
            indicators.append(rate_limit)
        
        # Dynamic content
        dynamic = self.dynamic_detector.detect_dynamic_content(orig_body, replay_body)
        if dynamic:
            indicators.append(dynamic)
        
        # Calculate overall false positive confidence
        if indicators:
            # Weighted average by confidence
            total_confidence = sum(ind.confidence for ind in indicators)
            weighted_confidence = total_confidence / len(indicators)
        else:
            weighted_confidence = 0.0
        
        # Determine if likely false positive
        is_false_positive = weighted_confidence >= 0.6
        
        if is_false_positive:
            self.false_positives_detected += 1
        
        # Determine likely cause
        likely_cause = None
        if indicators:
            # Highest confidence indicator
            top_indicator = max(indicators, key=lambda x: x.confidence)
            likely_cause = top_indicator.indicator_type
        
        # Recommendation
        if weighted_confidence >= 0.8:
            recommendation = "reject"  # Very likely false positive
        elif weighted_confidence >= 0.5:
            recommendation = "investigate"  # Uncertain
        else:
            recommendation = "proceed"  # Likely real
        
        # CRITICAL per .clinerules: Acknowledge causation uncertainty
        # System cannot definitively determine if behavioral changes are security-relevant
        causation_certainty = "UNCERTAIN"
        requires_human = weighted_confidence >= 0.4  # Human analysis for uncertain cases
        
        result = FilterResult(
            is_false_positive=is_false_positive,
            confidence=weighted_confidence,
            indicators=indicators,
            likely_cause=likely_cause,
            recommendation=recommendation,
            causation_certainty=causation_certainty,  # ALWAYS uncertain per .clinerules
            requires_human_analysis=requires_human
        )
        
        if is_false_positive:
            logger.info(f"False positive detected: {likely_cause} (confidence: {weighted_confidence:.2f})")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics."""
        fp_rate = self.false_positives_detected / self.total_checks if self.total_checks > 0 else 0.0
        
        return {
            'total_checks': self.total_checks,
            'false_positives_detected': self.false_positives_detected,
            'false_positive_rate': fp_rate,
            'causation_certainty': 'UNCERTAIN'  # Always per .clinerules
        }


# Global false positive filter instance
global_false_positive_filter = FalsePositiveFilter()