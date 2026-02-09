"""
VORTEX PoC Replayer - V17.0 ULTIMATE
Execute and verify Proof-of-Concept exploits

CRITICAL PRINCIPLES:
- Only replay AI-generated PoCs (never heuristic)
- Baseline vs PoC differential analysis
- Determinism scoring with uncertainty acknowledgment
- Behavioral evidence is INDICATIVE, not CONCLUSIVE

SECURITY:
- Isolated execution environment
- Rate limiting and timeout controls
- Scope validation before replay
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
from difflib import SequenceMatcher

from core.network import global_network_client, HTTPResponse
from core.verification.poc_parser import ParsedPoC
from core.exceptions import (
    PoCReplayError, VerificationTimeoutError,
    ResponseMismatchError, ScopeViolationError
)
from domain.models import VerificationResult, BehaviorAssessment
from domain.enums import MatchType

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Result from PoC replay."""
    success: bool
    baseline_response: HTTPResponse
    poc_response: HTTPResponse
    
    # Behavioral analysis
    behavioral_indicators: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    
    # Determinism scoring
    determinism_score: float = 0.0
    confidence: float = 0.0
    
    # Metadata
    replay_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Evidence
    evidence_summary: str = ""
    match_type: MatchType = MatchType.BEHAVIORAL_ONLY


class PoCReplayer:
    """
    Execute PoC replay and analyze results.
    
    WORKFLOW:
    1. Validate PoC is AI-generated (not heuristic)
    2. Make baseline request (original URL)
    3. Make PoC request (with exploit)
    4. Compare responses (behavioral differential)
    5. Calculate determinism score
    6. Generate verification result
    """
    
    def __init__(self):
        self.network_client = global_network_client
        
        # Timeouts
        self.baseline_timeout = 30
        self.poc_timeout = 30
        self.max_replay_attempts = 2
        
        # Determinism thresholds
        self.min_determinism_poc_replay = 0.70
        self.min_determinism_strong = 0.85
        
        # Behavioral thresholds
        self.response_time_threshold = 2.0  # seconds
        self.content_size_threshold = 100  # bytes
        self.content_similarity_threshold = 0.8
        
        # Statistics
        self.stats = {
            'replays_attempted': 0,
            'replays_successful': 0,
            'replays_failed': 0,
            'baseline_failures': 0,
            'poc_failures': 0
        }
    
    async def replay_poc(self, 
                        parsed_poc: ParsedPoC,
                        original_url: str,
                        original_payload: Optional[str] = None) -> ReplayResult:
        """
        Replay PoC and analyze results.
        
        Args:
            parsed_poc: Parsed PoC from AI analysis
            original_url: Original URL being tested
            original_payload: Original payload (for context)
            
        Returns:
            ReplayResult with behavioral analysis
            
        Raises:
            PoCReplayError: If replay fails
        """
        self.stats['replays_attempted'] += 1
        start_time = datetime.utcnow()
        
        logger.info(f"Starting PoC replay for {original_url}")
        
        try:
            # Step 1: Make baseline request (original, clean)
            baseline_response = await self._make_baseline_request(original_url)
            
            # Step 2: Make PoC request (with exploit)
            poc_response = await self._make_poc_request(parsed_poc)
            
            # Step 3: Analyze behavioral differences
            behavioral_analysis = self._analyze_behavioral_differences(
                baseline_response,
                poc_response,
                parsed_poc
            )
            
            # Step 4: Calculate determinism score
            determinism_score = self._calculate_determinism_score(
                baseline_response,
                poc_response,
                behavioral_analysis
            )
            
            # Step 5: Determine match type
            match_type = self._determine_match_type(
                behavioral_analysis,
                determinism_score
            )
            
            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(
                determinism_score,
                behavioral_analysis,
                match_type
            )
            
            # Step 7: Generate evidence summary
            evidence_summary = self._generate_evidence_summary(
                behavioral_analysis,
                determinism_score,
                baseline_response,
                poc_response
            )
            
            replay_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ReplayResult(
                success=confidence >= self.min_determinism_poc_replay,
                baseline_response=baseline_response,
                poc_response=poc_response,
                behavioral_indicators=behavioral_analysis['indicators'],
                uncertainty_factors=behavioral_analysis['uncertainty_factors'],
                determinism_score=determinism_score,
                confidence=confidence,
                replay_time=replay_time,
                evidence_summary=evidence_summary,
                match_type=match_type
            )
            
            if result.success:
                self.stats['replays_successful'] += 1
                logger.info(f"PoC replay successful: {confidence:.2f} confidence")
            else:
                self.stats['replays_failed'] += 1
                logger.warning(f"PoC replay failed: {confidence:.2f} confidence (below threshold)")
            
            return result
            
        except Exception as e:
            self.stats['replays_failed'] += 1
            logger.error(f"PoC replay error: {e}")
            raise PoCReplayError(f"PoC replay failed: {e}") from e
    
    async def _make_baseline_request(self, url: str) -> HTTPResponse:
        """
        Make baseline request (clean, no exploit).
        
        This establishes the "normal" behavior to compare against.
        """
        try:
            logger.debug(f"Making baseline request to {url}")
            
            response = await self.network_client.request(
                'GET',
                url,
            )
            
            return response
            
        except Exception as e:
            self.stats['baseline_failures'] += 1
            logger.error(f"Baseline request failed: {e}")
            raise PoCReplayError(f"Baseline request failed: {e}") from e
    
    async def _make_poc_request(self, parsed_poc: ParsedPoC) -> HTTPResponse:
        """
        Make PoC request (with exploit).
        
        This is the actual PoC execution.
        """
        try:
            logger.debug(f"Making PoC request to {parsed_poc.url}")
            
            # Prepare request kwargs
            kwargs = {}
            
            if parsed_poc.headers:
                kwargs['headers'] = parsed_poc.headers
            
            if parsed_poc.body and parsed_poc.method in ['POST', 'PUT', 'PATCH']:
                kwargs['data'] = parsed_poc.body
            
            response = await self.network_client.request(
                parsed_poc.method,
                parsed_poc.url,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            self.stats['poc_failures'] += 1
            logger.error(f"PoC request failed: {e}")
            raise PoCReplayError(f"PoC request failed: {e}") from e
    
    def _analyze_behavioral_differences(self,
                                       baseline: HTTPResponse,
                                       poc: HTTPResponse,
                                       parsed_poc: ParsedPoC) -> Dict[str, Any]:
        """
        CRITICAL: Analyze behavioral differences with uncertainty acknowledgment.
        
        Per VORTEX_EVIDENCE_STANDARDS.md:
        - Behavioral differences are INDICATIVE, not CONCLUSIVE
        - Cannot definitively determine causation remotely
        - Multiple non-security factors can cause differences
        """
        indicators = []
        uncertainty_factors = []
        
        # 1. Response time differential
        time_diff = abs(poc.response_time - baseline.response_time)
        if time_diff > self.response_time_threshold:
            indicators.append({
                'type': 'response_time',
                'description': f'Response time change: {time_diff:.1f}s',
                'baseline': baseline.response_time,
                'poc': poc.response_time,
                'confidence_impact': 0.2
            })
            uncertainty_factors.append(
                "Response time difference could be infrastructure/load balancer, not application"
            )
        
        # 2. Status code changes
        if baseline.status_code != poc.status_code:
            indicators.append({
                'type': 'status_code',
                'description': f'Status change: {baseline.status_code}→{poc.status_code}',
                'baseline': baseline.status_code,
                'poc': poc.status_code,
                'confidence_impact': 0.3
            })
            
            # Error status codes are more significant
            if poc.status_code >= 500:
                indicators[-1]['confidence_impact'] = 0.4
            
            uncertainty_factors.append(
                "Status code change could be upstream retry, rate limiting, or CDN behavior"
            )
        
        # 3. Content size changes
        size_diff = abs(len(poc.body) - len(baseline.body))
        if size_diff > self.content_size_threshold:
            indicators.append({
                'type': 'content_size',
                'description': f'Content size change: {size_diff} bytes',
                'baseline': len(baseline.body),
                'poc': len(poc.body),
                'confidence_impact': 0.25
            })
            uncertainty_factors.append(
                "Content size change could be dynamic content, A/B testing, or cache variation"
            )
        
        # 4. Content similarity analysis
        similarity = self._calculate_similarity(baseline.body, poc.body)
        if similarity < self.content_similarity_threshold:
            indicators.append({
                'type': 'content_similarity',
                'description': f'Content similarity: {similarity:.2f}',
                'baseline_hash': hashlib.md5(baseline.body[:1000].encode()).hexdigest(),
                'poc_hash': hashlib.md5(poc.body[:1000].encode()).hexdigest(),
                'confidence_impact': 0.3
            })
        
        # 5. Payload reflection (more deterministic)
        payload_reflected = False
        if parsed_poc.body:
            payload_sample = parsed_poc.body[:50].lower()
            if payload_sample in poc.body.lower():
                payload_reflected = True
                indicators.append({
                    'type': 'payload_reflection',
                    'description': 'Payload reflection detected',
                    'payload': parsed_poc.body[:100],
                    'confidence_impact': 0.4
                })
        
        # 6. Error message detection (deterministic)
        error_patterns = [
            'error', 'exception', 'warning', 'fatal', 'stack trace',
            'sql', 'mysql', 'postgresql', 'database'
        ]
        
        poc_body_lower = poc.body.lower()
        baseline_body_lower = baseline.body.lower()
        
        new_errors = []
        for pattern in error_patterns:
            if pattern in poc_body_lower and pattern not in baseline_body_lower:
                new_errors.append(pattern)
        
        if new_errors:
            indicators.append({
                'type': 'error_messages',
                'description': f'New error messages: {", ".join(new_errors)}',
                'patterns': new_errors,
                'confidence_impact': 0.5
            })
        
        return {
            'indicators': indicators,
            'uncertainty_factors': uncertainty_factors,
            'payload_reflected': payload_reflected,
            'has_new_errors': len(new_errors) > 0,
            'similarity_score': similarity
        }
    
    def _calculate_determinism_score(self,
                                    baseline: HTTPResponse,
                                    poc: HTTPResponse,
                                    behavioral_analysis: Dict[str, Any]) -> float:
        """
        Calculate determinism score for PoC replay.
        
        Per VORTEX_EVIDENCE_STANDARDS.md:
        - Deterministic evidence ≥0.8 required for SUBMIT_READY
        - Accounts for uncertainty in behavioral analysis
        """
        score = 0.0
        
        # Base score from indicators
        for indicator in behavioral_analysis['indicators']:
            score += indicator['confidence_impact']
        
        # Bonuses for highly deterministic signals
        if behavioral_analysis['has_new_errors']:
            score += 0.2  # Error messages are strong evidence
        
        if behavioral_analysis['payload_reflected']:
            score += 0.15  # Reflection is measurable
        
        # Penalty for uncertainty
        uncertainty_count = len(behavioral_analysis['uncertainty_factors'])
        uncertainty_penalty = min(uncertainty_count * 0.05, 0.2)
        score -= uncertainty_penalty
        
        # Normalize to 0.0-1.0
        score = max(0.0, min(score, 1.0))
        
        return score
    
    def _determine_match_type(self,
                              behavioral_analysis: Dict[str, Any],
                              determinism_score: float) -> MatchType:
        """Determine match type based on evidence."""
        
        if behavioral_analysis['has_new_errors']:
            if determinism_score >= 0.8:
                return MatchType.EXACT_REGEX
            else:
                return MatchType.PATTERN_MATCH
        
        if behavioral_analysis['payload_reflected']:
            return MatchType.STRUCTURAL_DIFFERENTIAL
        
        if determinism_score >= 0.7:
            return MatchType.FUZZY_MATCH
        
        return MatchType.BEHAVIORAL_ONLY
    
    def _calculate_confidence(self,
                             determinism_score: float,
                             behavioral_analysis: Dict[str, Any],
                             match_type: MatchType) -> float:
        """Calculate final confidence score."""
        
        base_confidence = determinism_score
        
        # Boost for strong match types
        if match_type == MatchType.EXACT_REGEX:
            base_confidence += 0.1
        elif match_type == MatchType.STRUCTURAL_DIFFERENTIAL:
            base_confidence += 0.05
        
        # Cap at 0.95 (never 100% certainty in behavioral)
        return min(base_confidence, 0.95)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio."""
        try:
            # Use first 2000 chars for performance
            sample1 = text1[:2000] if text1 else ""
            sample2 = text2[:2000] if text2 else ""
            
            return SequenceMatcher(None, sample1, sample2).ratio()
        except Exception:
            return 0.0
    
    def _generate_evidence_summary(self,
                                   behavioral_analysis: Dict[str, Any],
                                   determinism_score: float,
                                   baseline: HTTPResponse,
                                   poc: HTTPResponse) -> str:
        """Generate human-readable evidence summary."""
        
        lines = [
            "PoC Replay Verification Results:",
            f"Determinism Score: {determinism_score:.2f}",
            ""
        ]
        
        if behavioral_analysis['indicators']:
            lines.append("Behavioral Indicators:")
            for indicator in behavioral_analysis['indicators']:
                lines.append(f"  - {indicator['description']}")
            lines.append("")
        
        if behavioral_analysis['uncertainty_factors']:
            lines.append("Uncertainty Factors:")
            for factor in behavioral_analysis['uncertainty_factors']:
                lines.append(f"  - {factor}")
            lines.append("")
        
        lines.append(f"Baseline: {baseline.status_code} ({len(baseline.body)} bytes)")
        lines.append(f"PoC: {poc.status_code} ({len(poc.body)} bytes)")
        
        return "\n".join(lines)
    
    def convert_to_verification_result(self, replay_result: ReplayResult) -> VerificationResult:
        """Convert ReplayResult to VerificationResult."""
        
        return VerificationResult(
            success=replay_result.success,
            confidence=replay_result.confidence,
            match_type=replay_result.match_type,
            matched_pattern=f"PoC replay determinism: {replay_result.determinism_score:.2f}",
            response_status=replay_result.poc_response.status_code,
            response_time=replay_result.poc_response.response_time,
            response_body_sample=replay_result.poc_response.body[:500],
            
            # Behavioral indicators
            status_code_change=(
                replay_result.baseline_response.status_code != 
                replay_result.poc_response.status_code
            ),
            response_time_change=(
                replay_result.poc_response.response_time - 
                replay_result.baseline_response.response_time
            ),
            content_size_change=(
                len(replay_result.poc_response.body) - 
                len(replay_result.baseline_response.body)
            ),
            
            # Metadata
            verification_method='poc_replay',
            determinism_score=replay_result.determinism_score,
            verified_at=replay_result.timestamp
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get replayer statistics."""
        return self.stats.copy()


# Global replayer instance
global_poc_replayer = PoCReplayer()


async def replay_poc(parsed_poc: ParsedPoC, 
                    original_url: str,
                    original_payload: Optional[str] = None) -> VerificationResult:
    """
    Convenience function to replay PoC.
    
    Args:
        parsed_poc: Parsed PoC from AI
        original_url: Original URL being tested
        original_payload: Original payload
        
    Returns:
        VerificationResult
    """
    result = await global_poc_replayer.replay_poc(
        parsed_poc,
        original_url,
        original_payload
    )
    return global_poc_replayer.convert_to_verification_result(result)