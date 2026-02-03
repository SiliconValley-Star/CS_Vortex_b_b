"""
VORTEX System Verification Engine - V17.0 ULTIMATE
Authoritative system verification per VORTEX_EVIDENCE_STANDARDS.md

CRITICAL: System verification is THE authoritative evidence source.
Text matching alone does NOT prove vulnerability.

VERIFICATION TYPES:
1. PoC Replay Verification (AI-generated PoCs only)
2. Pattern-based Verification (regex, structural)
3. Behavioral Differential Analysis (with uncertainty)

AUTHORITY LEVEL: HIGHEST (Level 1)
"""

import asyncio
import logging
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

from domain.models import VerificationResult, AssessmentResult
from domain.enums import FindingType
from core.network import NetworkClient, HTTPResponse
from core.exceptions import (
    VerificationError, PoCReplayError, VerificationTimeoutError,
    ResponseMismatchError, HeuristicPoCReplayAttemptError
)

# Initialize logger first (before any conditional imports)
logger = logging.getLogger(__name__)

# V17.0 ULTIMATE - PoC Replay System
from core.verification.poc_parser import global_poc_parser, ParsedPoC
from core.verification.poc_replayer import global_poc_replayer

# PHASE 5 - AI Optimization (conditional imports)
try:
    from core.ai.triage_mode import AITriageMode, TriageDecision
    from core.analysis.deterministic_analyzer import DeterministicAnalyzer
    PHASE5_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PHASE 5 components not available: {e}")
    PHASE5_AVAILABLE = False
    AITriageMode = None
    TriageDecision = None
    DeterministicAnalyzer = None


@dataclass
class BehavioralIndicator:
    """Behavioral difference indicator."""
    type: str
    description: str
    confidence_impact: float


class SystemVerificationEngine:
    """
    Authoritative system verification engine.
    
    CRITICAL PRINCIPLES:
    - AI-generated PoCs ONLY (never heuristic)
    - Behavioral differences are INDICATIVE not CONCLUSIVE
    - Text matching is NOT proof of vulnerability
    - Causation analysis requires human expert
    
    VERIFICATION HIERARCHY:
    1. PoC Replay (highest confidence)
    2. Structural Differential (high confidence)
    3. Pattern Matching (medium confidence)
    4. Heuristic Indicators (lowest - NOT verified)
    """
    
    def __init__(self):
        self.network_client = NetworkClient()
        
        # PHASE 5 - AI Optimization (conditional initialization)
        if PHASE5_AVAILABLE:
            self.ai_triage = AITriageMode()
            self.deterministic_analyzer = DeterministicAnalyzer()
            logger.info("PHASE 5 AI Optimization enabled")
        else:
            self.ai_triage = None
            self.deterministic_analyzer = None
            logger.info("PHASE 5 AI Optimization not available")
        
        # Verification timeouts
        self.poc_replay_timeout = 30
        self.pattern_verification_timeout = 15
        
        # Confidence thresholds
        self.min_confidence_poc_replay = 0.85
        self.min_confidence_structural = 0.75
        self.min_confidence_pattern = 0.60
        
        # Behavioral analysis configuration
        self.behavioral_indicators_config = {
            'response_time_threshold': 2.0,  # seconds
            'status_code_weight': 0.3,
            'content_size_threshold': 100,  # bytes
            'content_similarity_threshold': 0.8
        }
        
        # Statistics
        self.stats = {
            'total_verifications': 0,
            'poc_replays': 0,
            'pattern_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'behavioral_analyses': 0
        }
    
    async def verify_finding(self, finding: AssessmentResult) -> VerificationResult:
        """
        Perform authoritative system verification with PHASE 5 AI optimization.
        
        PHASE 5 INTEGRATED VERIFICATION STRATEGY:
        1. Run Deterministic Analyzer first (PHASE 5.3)
        2. If high deterministic confidence → Skip AI completely
        3. Else → Use AI Triage Mode to decide (PHASE 5.1)
        4. If AI needed → Check if AI PoC available → PoC Replay or Pattern
        5. If AI skipped → Use pattern-based verification
        
        Args:
            finding: Assessment result to verify
            
        Returns:
            VerificationResult with authority-compliant evidence
        """
        self.stats['total_verifications'] += 1
        
        try:
            # PHASE 5 Integration - Only if available
            if PHASE5_AVAILABLE and self.ai_triage and self.deterministic_analyzer:
                # PHASE 5.3 - Run Deterministic Analysis First
                det_result = self.deterministic_analyzer.analyze(finding)
                logger.info(
                    f"Deterministic analysis: {det_result['confidence_level']} "
                    f"({det_result['confidence_score']:.2f}), "
                    f"needs_ai={det_result['needs_ai']}"
                )
                
                # PHASE 5.1 - AI Triage Decision
                context = {
                    'deterministic_confidence': det_result['confidence_score'],
                    'deterministic_matches': det_result['match_count'],
                    'deterministic_checks_passed': det_result['match_count']
                }
                
                triage_decision = self.ai_triage.should_use_ai(finding, context)
                logger.info(f"AI Triage decision: {triage_decision.value}")
                
                # Execute based on triage decision
                if triage_decision == TriageDecision.AUTO_ACCEPT:
                    # High deterministic confidence - skip all verification
                    # Get patterns safely
                    patterns = det_result.get('patterns_matched', det_result.get('matched_patterns', []))
                    pattern_str = ', '.join(patterns[:3]) if patterns else 'multiple checks passed'
                    
                    result = VerificationResult(
                        success=True,
                        match_type="deterministic_auto_accept",
                        confidence=det_result['confidence_score'],
                        matched_pattern=f"Deterministic: {pattern_str}",
                        determinism_score=1.0  # Pure deterministic
                    )
                    logger.info(f"Auto-accepted via deterministic analysis (AI completely skipped)")
                    
                elif triage_decision == TriageDecision.AUTO_REJECT:
                    # Clear false positive
                    result = VerificationResult(
                        success=False,
                        match_type="deterministic_auto_reject",
                        confidence=0.0,
                        error="Auto-rejected: clear false positive pattern"
                    )
                    logger.info(f"Auto-rejected via deterministic analysis")
                    
                elif triage_decision == TriageDecision.USE_AI:
                    # AI needed - check if AI PoC available
                    if self._should_replay_poc(finding):
                        result = await self._verify_with_poc_replay(finding)
                    else:
                        result = await self._verify_with_pattern_analysis(finding)
                    logger.info(f"AI verification used (triage recommended)")
                    
                else:  # SKIP_AI
                    # Use pattern-based only (no AI)
                    result = await self._verify_with_pattern_analysis(finding)
                    logger.info(f"Pattern-only verification (AI skipped by triage)")
            else:
                # PHASE 5 not available - use original logic
                if self._should_replay_poc(finding):
                    result = await self._verify_with_poc_replay(finding)
                else:
                    result = await self._verify_with_pattern_analysis(finding)
            
            # Record statistics
            if result.success:
                self.stats['successful_verifications'] += 1
            else:
                self.stats['failed_verifications'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed for {finding.id}: {e}", exc_info=True)
            self.stats['failed_verifications'] += 1
            
            return VerificationResult(
                success=False,
                match_type="error",
                confidence=0.0,
                error=str(e)
            )
    
    def _should_replay_poc(self, finding: AssessmentResult) -> bool:
        """
        CRITICAL: Determine if PoC should be replayed.
        
        RULES:
        - NEVER replay heuristic-only PoCs
        - Only replay AI-generated PoCs from successful analysis
        - Must have confidence_source != HEURISTIC_ONLY
        """
        # Check confidence source
        if hasattr(finding, 'confidence_source'):
            if finding.confidence_source == 'HEURISTIC_ONLY':
                logger.debug(f"Blocking heuristic PoC replay for {finding.id}")
                return False
        
        # Check AI analysis
        if not finding.ai_analysis:
            return False
        
        if not finding.ai_analysis.success:
            return False
        
        if finding.ai_analysis.is_fallback_result:
            return False
        
        # Check if PoC exists
        if not hasattr(finding.ai_analysis, 'poc') or not finding.ai_analysis.poc:
            return False
        
        return True
    
    async def _verify_with_poc_replay(self, finding: AssessmentResult) -> VerificationResult:
        """
        Verify by replaying AI-generated PoC.
        
        This is the highest confidence verification method.
        V17.0 ULTIMATE - Full PoC Replay Implementation
        """
        self.stats['poc_replays'] += 1
        logger.info(f"PoC replay verification for {finding.id}")
        
        try:
            # Extract PoC from AI analysis
            poc_text = finding.ai_analysis.poc
            
            if not poc_text:
                logger.error("No PoC text found in AI analysis")
                raise PoCReplayError("No PoC available for replay")
            
            # Parse PoC using PoCParser
            parsed_poc = global_poc_parser.parse(poc_text)
            
            if not parsed_poc:
                logger.error(f"Failed to parse PoC: {poc_text[:100]}")
                raise PoCReplayError("PoC parsing failed")
            
            logger.info(f"PoC parsed successfully: {parsed_poc.format_detected} format")
            
            # Execute PoC replay using PoCReplayer
            replay_result = await global_poc_replayer.replay_poc(
                parsed_poc=parsed_poc,
                original_url=finding.url,
                original_payload=finding.payload
            )
            
            # Convert ReplayResult to VerificationResult
            verification_result = global_poc_replayer.convert_to_verification_result(replay_result)
            
            logger.info(
                f"PoC replay complete: success={verification_result.success}, "
                f"confidence={verification_result.confidence:.2f}, "
                f"determinism={verification_result.determinism_score:.2f}"
            )
            
            return verification_result
            
        except PoCReplayError:
            raise
        except Exception as e:
            logger.error(f"PoC replay failed: {e}")
            raise PoCReplayError(f"PoC replay failed: {e}") from e
    
    async def _verify_with_pattern_analysis(self, finding: AssessmentResult) -> VerificationResult:
        """
        Verify using pattern matching and behavioral analysis.
        
        Lower confidence than PoC replay but still valuable.
        """
        self.stats['pattern_verifications'] += 1
        logger.info(f"Pattern verification for {finding.id}")
        
        try:
            # Get vulnerability-specific patterns
            patterns = self._get_vulnerability_patterns(finding.finding_type)
            
            # Make request with payload
            response = await self._make_safe_request(
                finding.url,
                payload=finding.payload,
                timeout=self.pattern_verification_timeout
            )
            
            # Check patterns
            matched_pattern = None
            max_confidence = 0.0
            
            for pattern in patterns:
                if self._check_pattern(response.body, pattern):
                    matched_pattern = pattern
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(
                        pattern,
                        response,
                        finding.finding_type
                    )
                    max_confidence = max(max_confidence, confidence)
            
            if matched_pattern and max_confidence >= self.min_confidence_pattern:
                return VerificationResult(
                    success=True,
                    match_type="pattern_match",
                    confidence=max_confidence,
                    matched_pattern=matched_pattern,
                    response_time=response.response_time,
                    response_status=response.status_code
                )
            else:
                return VerificationResult(
                    success=False,
                    match_type="pattern_not_found",
                    confidence=max_confidence,
                    error="No vulnerability patterns matched"
                )
            
        except Exception as e:
            logger.error(f"Pattern verification failed: {e}")
            raise VerificationError(f"Pattern verification failed: {e}") from e
    
    def _analyze_behavioral_differences(self,
                                       baseline: HTTPResponse,
                                       test: HTTPResponse,
                                       payload: str) -> Dict[str, Any]:
        """
        CRITICAL: Analyze behavioral differences with uncertainty acknowledgment.
        
        Differences can result from:
        - SECURITY-RELEVANT: Backend errors, logic changes, validation failures
        - NON-SECURITY: CDN switching, load balancing, cache variations, A/B testing
        
        System CANNOT definitively distinguish causes remotely.
        """
        self.stats['behavioral_analyses'] += 1
        
        indicators = []
        uncertainty_factors = []
        
        # Response time differential
        time_diff = abs(test.response_time - baseline.response_time)
        if time_diff > self.behavioral_indicators_config['response_time_threshold']:
            indicators.append(BehavioralIndicator(
                type='response_time',
                description=f'Response time change: {time_diff:.1f}s',
                confidence_impact=0.2
            ))
            uncertainty_factors.append("Could be infrastructure/load balancer, not application")
        
        # Status code changes
        if baseline.status_code != test.status_code:
            indicators.append(BehavioralIndicator(
                type='status_code',
                description=f'Status change: {baseline.status_code}→{test.status_code}',
                confidence_impact=0.3
            ))
            uncertainty_factors.append("Could be upstream retry, rate limiting, or CDN")
        
        # Content size changes
        size_diff = abs(len(test.body) - len(baseline.body))
        if size_diff > self.behavioral_indicators_config['content_size_threshold']:
            indicators.append(BehavioralIndicator(
                type='content_size',
                description=f'Content size change: {size_diff} bytes',
                confidence_impact=0.25
            ))
            uncertainty_factors.append("Could be dynamic content, A/B testing, or cache variation")
        
        # Content similarity analysis
        similarity = self._calculate_similarity(baseline.body, test.body)
        if similarity < self.behavioral_indicators_config['content_similarity_threshold']:
            indicators.append(BehavioralIndicator(
                type='content_similarity',
                description=f'Content similarity: {similarity:.2f}',
                confidence_impact=0.3
            ))
        
        # Payload reflection (more deterministic)
        payload_reflected = payload and payload.lower() in test.body.lower()
        if payload_reflected:
            indicators.append(BehavioralIndicator(
                type='payload_reflection',
                description='Payload reflection detected',
                confidence_impact=0.4
            ))
        
        # Calculate confidence with uncertainty penalty
        base_confidence = sum(ind.confidence_impact for ind in indicators)
        base_confidence = min(base_confidence, 0.9)
        
        uncertainty_penalty = len(uncertainty_factors) * 0.1
        final_confidence = max(0.0, base_confidence - uncertainty_penalty)
        
        return {
            'indicators': [
                {
                    'type': ind.type,
                    'description': ind.description,
                    'confidence_impact': ind.confidence_impact
                }
                for ind in indicators
            ],
            'uncertainty_factors': uncertainty_factors,
            'confidence': final_confidence,
            'causation_determination': "UNKNOWN - requires human expert analysis",
            'max_automated_status': "SYSTEM_VERIFIED",  # NOT SUBMIT_READY
            'payload_reflected': payload_reflected
        }
    
    def _get_vulnerability_patterns(self, finding_type: FindingType) -> List[str]:
        """Get vulnerability-specific verification patterns."""
        
        patterns = {
            FindingType.SQLI_ERROR: [
                r'mysql.*error',
                r'postgresql.*error',
                r'sql.*syntax',
                r'ora-\d+',
                r'database.*error'
            ],
            FindingType.XSS_REFLECTED: [
                r'<script[^>]*>',
                r'javascript:',
                r'onerror\s*=',
                r'onload\s*='
            ],
            FindingType.SSRF: [
                r'192\.168\.',
                r'10\.\d+\.',
                r'172\.(1[6-9]|2\d|3[01])\.',
                r'localhost',
                r'127\.0\.0\.1'
            ],
            FindingType.LFI: [
                r'root:.*:0:0:',
                r'\[boot loader\]',
                r'<\?php',
                r'/etc/passwd'
            ]
        }
        
        return patterns.get(finding_type, [])
    
    def _check_pattern(self, content: str, pattern: str) -> bool:
        """Check if pattern matches content."""
        try:
            return re.search(pattern, content, re.IGNORECASE) is not None
        except Exception as e:
            logger.error(f"Pattern matching error: {e}")
            return False
    
    def _calculate_pattern_confidence(self,
                                     pattern: str,
                                     response: HTTPResponse,
                                     finding_type: FindingType) -> float:
        """Calculate confidence based on pattern match."""
        
        # Base confidence by pattern specificity
        base_confidence = 0.6
        
        # Bonus for specific vulnerability indicators
        if finding_type == FindingType.SQLI_ERROR:
            if 'mysql' in pattern.lower() or 'postgresql' in pattern.lower():
                base_confidence += 0.15
        elif finding_type == FindingType.XSS_REFLECTED:
            if 'script' in pattern.lower():
                base_confidence += 0.20
        
        # Bonus for error status codes (500, 503, etc.)
        if response.status_code >= 500:
            base_confidence += 0.1
        
        return min(base_confidence, 0.90)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio."""
        try:
            # Use first 1000 chars for performance
            sample1 = text1[:1000] if text1 else ""
            sample2 = text2[:1000] if text2 else ""
            
            return SequenceMatcher(None, sample1, sample2).ratio()
        except Exception:
            return 0.0
    
    async def _make_safe_request(self,
                                 url: str,
                                 payload: Optional[str] = None,
                                 timeout: int = 30) -> HTTPResponse:
        """Make safe HTTP request with error handling."""
        try:
            # Prepare request
            request_url = url
            if payload:
                # Append payload (simplified - would need proper handling)
                separator = '&' if '?' in url else '?'
                request_url = f"{url}{separator}test={payload}"
            
            # Make request
            response = await self.network_client.request(
                'GET',
                request_url
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Safe request failed: {e}")
            raise VerificationError(f"Request failed: {e}") from e
    
    # REMOVED - No longer needed (handled by PoCParser and PoCReplayer)
    # _extract_poc_url() and _extract_poc_payload() replaced by:
    # - core.verification.poc_parser.PoCParser (parsing)
    # - core.verification.poc_replayer.PoCReplayer (execution)
    
    def get_stats(self) -> Dict[str, int]:
        """Get verification statistics."""
        return self.stats.copy()


# Global verification engine instance
try:
    global_verification_engine = SystemVerificationEngine()
    logger.info("SystemVerificationEngine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SystemVerificationEngine: {e}")
    global_verification_engine = None