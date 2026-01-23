"""
VORTEX AI Fallback Manager - V17.0 ULTIMATE
AI fallback strategies per VORTEX_AI_INTEGRATION.md

CRITICAL: Fallback results are NON-AUTHORITATIVE
"""

import structlog
from typing import Dict, Optional
from datetime import datetime

from domain.enums import AIVerdict, AuthorityLevel, VerificationStatus, ConfidenceSource
from domain.models import AssessmentResult, AIAnalysisResult
from config.constants import AI_INTEGRATION_LIMITS

logger = structlog.get_logger()


class AIFallbackManager:
    """
    Manages AI fallback strategies
    Per VORTEX_AI_INTEGRATION.md: Comprehensive fallback chain
    """
    
    def __init__(self):
        self.fallback_history = []
        
        # Heuristic confidence penalty per VORTEX_AI_INTEGRATION.md
        self.heuristic_confidence_penalty = AI_INTEGRATION_LIMITS.fallback_confidence_penalty
    
    def create_heuristic_fallback_result(
        self,
        finding: AssessmentResult,
        reason: str = "AI models unavailable"
    ) -> AIAnalysisResult:
        """
        Create heuristic-only fallback when all AI models fail
        Per VORTEX_AI_INTEGRATION.md: Always route to NEEDS_MANUAL
        
        Args:
            finding: Assessment result with heuristic data
            reason: Reason for fallback
        
        Returns: Non-authoritative fallback result
        """
        # Apply conservative confidence penalty
        original_confidence = finding.heuristic_score
        adjusted_confidence = original_confidence * (1 - self.heuristic_confidence_penalty)
        
        # Conservative reportability with penalty
        reportability = max(0.0, adjusted_confidence - 0.4)
        
        result = AIAnalysisResult(
            model_used="heuristic_fallback",
            verdict=AIVerdict.NEEDS_MANUAL,  # Always require manual review
            confidence=adjusted_confidence,
            
            # Cannot determine without AI
            exploitability=None,
            impact="UNKNOWN",
            reportability=reportability,
            
            # Reasoning explaining fallback
            reasoning=(
                f"Heuristic-only analysis due to AI unavailability.\n\n"
                f"REASON: {reason}\n\n"
                f"Original heuristic confidence: {original_confidence:.2f}\n"
                f"Adjusted confidence (with penalty): {adjusted_confidence:.2f}\n"
                f"Confidence penalty: {self.heuristic_confidence_penalty:.1%}\n\n"
                f"CRITICAL: This is a FALLBACK result and is NON-AUTHORITATIVE.\n"
                f"Manual expert review is REQUIRED for final determination.\n\n"
                f"Finding Type: {finding.finding_type.value if finding.finding_type else 'Unknown'}\n"
                f"URL: {finding.url}\n"
                f"Evidence: {finding.evidence[:200] if finding.evidence else 'None'}..."
            ),
            poc="",  # No PoC from heuristic fallback
            
            # Fallback metadata
            success=True,
            is_fallback_result=True,
            fallback_reason=reason,
            authority_level=AuthorityLevel.HEURISTIC,
            is_authoritative=False,
            requires_system_validation=True,
            
            # Additional context
            availability_status="unavailable"
        )
        
        logger.warning(
            "Created heuristic fallback result",
            finding_id=str(finding.id),
            reason=reason,
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence
        )
        
        self._record_fallback(finding, "heuristic", reason)
        
        return result
    
    def create_ai_unavailable_result(
        self,
        finding_data: Dict,
        reason: str = "All AI models failed"
    ) -> AIAnalysisResult:
        """
        Create result when AI is completely unavailable
        Per VORTEX_AI_INTEGRATION.md: Non-authoritative, routes to manual
        
        Args:
            finding_data: Finding information dict
            reason: Reason for unavailability
        
        Returns: AI unavailable result
        """
        result = AIAnalysisResult(
            model_used="ai_unavailable",
            verdict=AIVerdict.NEEDS_MANUAL,  # Always manual when AI unavailable
            confidence=0.0,  # No confidence without AI
            
            # Cannot determine without AI
            exploitability=None,
            impact="UNKNOWN",
            reportability=None,
            
            # Reasoning explaining unavailability
            reasoning=(
                f"AI models unavailable - requires manual expert analysis.\n\n"
                f"REASON: {reason}\n\n"
                f"All configured AI models failed to respond. This could be due to:\n"
                f"- API rate limiting\n"
                f"- Model unavailability\n"
                f"- Network connectivity issues\n"
                f"- Service outages\n\n"
                f"CRITICAL: Without AI analysis, this finding requires manual review.\n"
                f"System will attempt AI analysis retry on next scan cycle.\n\n"
                f"Finding Type: {finding_data.get('finding_type', 'Unknown')}\n"
                f"URL: {finding_data.get('url', 'Unknown')}"
            ),
            poc="",
            
            # Unavailability metadata
            success=False,
            is_fallback_result=True,
            fallback_reason=reason,
            authority_level=AuthorityLevel.NONE,
            is_authoritative=False,
            requires_system_validation=True,
            availability_status="unavailable",
            error_message=f"AI unavailable: {reason}"
        )
        
        logger.error(
            "AI completely unavailable",
            finding_id=finding_data.get('id', 'unknown'),
            reason=reason
        )
        
        self._record_fallback(finding_data, "unavailable", reason)
        
        return result
    
    def create_malformed_recovery_result(
        self,
        recovered_data: Dict,
        model: str,
        finding_data: Dict
    ) -> AIAnalysisResult:
        """
        Create result from malformed JSON recovery
        Per VORTEX_AI_INTEGRATION.md: Recovery is NON-AUTHORITATIVE
        
        Args:
            recovered_data: Recovered data from malformed JSON
            model: Model that produced malformed response
            finding_data: Finding information
        
        Returns: Non-authoritative recovered result
        """
        # Extract basic fields
        confidence = recovered_data.get('confidence', 0.3)
        verdict_str = recovered_data.get('verdict', 'NEEDS_MANUAL')
        
        # Parse verdict
        try:
            verdict = AIVerdict[verdict_str.upper()]
        except KeyError:
            verdict = AIVerdict.NEEDS_MANUAL
        
        # Apply severe penalties for malformed recovery
        recovered_confidence = min(confidence * 0.3, 0.4)  # Severe penalty
        
        result = AIAnalysisResult(
            model_used=f"{model}_malformed_recovery",
            verdict=AIVerdict.NEEDS_MANUAL,  # FORCE manual review
            confidence=recovered_confidence,
            
            # Unknown fields (recovery cannot be trusted)
            exploitability=None,
            impact="UNKNOWN",
            reportability=None,  # NOT REPORTABLE
            
            # Reasoning explaining recovery
            reasoning=(
                f"Recovered from malformed AI response - NOT AUTHORITATIVE.\n\n"
                f"Model: {model}\n"
                f"Original verdict: {verdict_str} (OVERRIDDEN to NEEDS_MANUAL)\n"
                f"Original confidence: {confidence:.2f}\n"
                f"Adjusted confidence: {recovered_confidence:.2f} (severe penalty)\n\n"
                f"CRITICAL: This result was recovered from malformed JSON and is\n"
                f"NOT AUTHORITATIVE. Manual expert review is MANDATORY.\n\n"
                f"Malformed AI responses indicate:\n"
                f"- Model instability\n"
                f"- Parsing issues\n"
                f"- Unreliable analysis\n\n"
                f"This result is for statistical tracking only and CANNOT be used\n"
                f"for automated decisions or SUBMIT_READY status."
            ),
            poc="",
            
            # Recovery metadata
            success=False,  # Mark as failed
            is_fallback_result=True,
            fallback_reason="Malformed JSON recovery",
            authority_level=AuthorityLevel.NONE,
            is_authoritative=False,
            requires_system_validation=True,
            error_message="Malformed JSON recovery - non-authoritative data"
        )
        
        logger.warning(
            "Created malformed recovery result",
            finding_id=finding_data.get('id', 'unknown'),
            model=model,
            original_confidence=confidence,
            recovered_confidence=recovered_confidence
        )
        
        self._record_fallback(finding_data, "malformed_recovery", f"Model: {model}")
        
        return result
    
    def assess_fallback_strategy(
        self,
        finding: AssessmentResult,
        ai_error: Optional[Exception] = None
    ) -> str:
        """
        Assess which fallback strategy to use
        
        Returns: "heuristic" | "unavailable" | "retry"
        """
        # If finding has strong heuristic data, use heuristic fallback
        if finding.heuristic_score >= 0.6:
            return "heuristic"
        
        # If temporary error, suggest retry
        if ai_error and self._is_temporary_error(ai_error):
            return "retry"
        
        # Otherwise, unavailable
        return "unavailable"
    
    def _is_temporary_error(self, error: Exception) -> bool:
        """Check if error is temporary and worth retrying."""
        error_str = str(error).lower()
        
        temporary_indicators = [
            "timeout",
            "rate limit",
            "503",
            "429",
            "connection",
            "network"
        ]
        
        return any(indicator in error_str for indicator in temporary_indicators)
    
    def _record_fallback(
        self,
        finding_or_data,
        fallback_type: str,
        reason: str
    ) -> None:
        """Record fallback for audit."""
        if isinstance(finding_or_data, AssessmentResult):
            finding_id = str(finding_or_data.id)
        else:
            finding_id = finding_or_data.get('id', 'unknown')
        
        record = {
            'timestamp': datetime.utcnow(),
            'finding_id': finding_id,
            'fallback_type': fallback_type,
            'reason': reason
        }
        
        self.fallback_history.append(record)
    
    def get_fallback_stats(self) -> Dict:
        """Get fallback usage statistics."""
        if not self.fallback_history:
            return {
                'total_fallbacks': 0,
                'fallback_rate': 0.0,
                'fallback_types': {}
            }
        
        from collections import Counter
        
        total = len(self.fallback_history)
        types = [f['fallback_type'] for f in self.fallback_history]
        type_counts = dict(Counter(types))
        
        return {
            'total_fallbacks': total,
            'fallback_types': type_counts,
            'recent_fallbacks': self.fallback_history[-10:]  # Last 10
        }


def create_heuristic_fallback_result(
    finding: AssessmentResult,
    reason: str = "AI models unavailable"
) -> AIAnalysisResult:
    """
    Convenience function for heuristic fallback
    """
    manager = AIFallbackManager()
    return manager.create_heuristic_fallback_result(finding, reason)


def create_ai_unavailable_result(
    finding_data: Dict,
    reason: str = "All AI models failed"
) -> AIAnalysisResult:
    """
    Convenience function for AI unavailable result
    """
    manager = AIFallbackManager()
    return manager.create_ai_unavailable_result(finding_data, reason)


# Global fallback manager
global_fallback_manager = AIFallbackManager()