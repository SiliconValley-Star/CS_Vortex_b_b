"""
VORTEX Authority Hierarchy - V17.0 ULTIMATE
IMMUTABLE authority hierarchy enforcement per VORTEX_CORE_AUTHORITY.md

CRITICAL: Authority hierarchy CANNOT be bypassed or modified
1. System Verification (Deterministic) - HIGHEST AUTHORITY
2. Human Expert Analysis (Authoritative) - SECOND
3. AI Analysis (ADVISORY ONLY) - THIRD
4. Heuristic Detection (Indicative) - LOWEST
"""

import structlog
from typing import Optional, Tuple
from datetime import datetime

from domain.enums import AuthorityLevel, VerificationStatus
from domain.models import AssessmentResult, VerificationResult, AIAnalysisResult
from config.constants import AUTHORITY_THRESHOLDS

logger = structlog.get_logger()


class AuthorityHierarchyEnforcer:
    """
    Enforces IMMUTABLE authority hierarchy
    Per VORTEX_CORE_AUTHORITY.md: NO rule can bypass this hierarchy
    """
    
    # Authority levels with numerical priority (lower = higher authority)
    AUTHORITY_PRIORITY = {
        AuthorityLevel.SYSTEM_VERIFICATION: 1,  # HIGHEST
        AuthorityLevel.HUMAN_EXPERT: 2,         # SECOND
        AuthorityLevel.AI_ADVISORY: 3,          # THIRD (NEVER authoritative)
        AuthorityLevel.HEURISTIC: 4,            # LOWEST
        AuthorityLevel.NONE: 999,               # No authority
    }
    
    def __init__(self):
        self.violation_count = 0
        self.decision_history = []
    
    def get_finding_authority_level(self, finding: AssessmentResult) -> AuthorityLevel:
        """
        Determine highest authority level achieved by finding
        Per VORTEX_CORE_AUTHORITY.md: System verification > Human > AI > Heuristic
        """
        # Check system verification (HIGHEST AUTHORITY)
        if (finding.verification_result and 
            finding.verification_result.success and
            finding.verification_result.confidence >= AUTHORITY_THRESHOLDS.submit_ready_system_confidence):
            return AuthorityLevel.SYSTEM_VERIFICATION
        
        # Check human expert (SECOND AUTHORITY - not implemented yet)
        # This would check if human analyst has reviewed
        # For now, we don't have human review system integrated
        
        # Check AI analysis (THIRD AUTHORITY - ADVISORY ONLY)
        if (finding.ai_analysis and 
            finding.ai_analysis.success and
            not finding.ai_analysis.is_fallback_result):
            return AuthorityLevel.AI_ADVISORY
        
        # Default to heuristic (LOWEST AUTHORITY)
        if finding.heuristic_score > 0:
            return AuthorityLevel.HEURISTIC
        
        return AuthorityLevel.NONE
    
    def validate_submit_ready_authority(self, finding: AssessmentResult) -> Tuple[bool, str]:
        """
        Validate finding meets ALL authority requirements for SUBMIT_READY
        Per VORTEX_CORE_AUTHORITY.md: ALL requirements must be met
        
        Returns: (is_valid, reason)
        """
        violations = []
        
        # REQUIREMENT 1: System verification (MANDATORY)
        if not finding.verification_result or not finding.verification_result.success:
            violations.append("Missing successful system verification (MANDATORY)")
            logger.warning(
                "Authority violation: No system verification",
                finding_id=str(finding.id),
                status=finding.status
            )
        
        # REQUIREMENT 2: Confidence threshold (MANDATORY)
        if (finding.verification_result and 
            finding.verification_result.confidence < AUTHORITY_THRESHOLDS.submit_ready_system_confidence):
            violations.append(
                f"System confidence {finding.verification_result.confidence:.2f} "
                f"below threshold {AUTHORITY_THRESHOLDS.submit_ready_system_confidence}"
            )
            logger.warning(
                "Authority violation: Low system confidence",
                finding_id=str(finding.id),
                confidence=finding.verification_result.confidence,
                threshold=AUTHORITY_THRESHOLDS.submit_ready_system_confidence
            )
        
        # REQUIREMENT 3: No UNKNOWN values in critical fields
        if self._has_unknown_values(finding):
            violations.append("UNKNOWN values present in critical fields (FORBIDDEN)")
            logger.warning(
                "Authority violation: UNKNOWN values present",
                finding_id=str(finding.id)
            )
        
        # REQUIREMENT 4: Deterministic evidence
        if not self._has_deterministic_evidence(finding):
            violations.append("Evidence does not meet deterministic standards")
            logger.warning(
                "Authority violation: Non-deterministic evidence",
                finding_id=str(finding.id)
            )
        
        # Track violations
        if violations:
            self.violation_count += 1
            self._record_violation(finding, violations)
            return False, "; ".join(violations)
        
        # All requirements met
        self._record_valid_decision(finding)
        return True, "All authority requirements met"
    
    def _has_unknown_values(self, finding: AssessmentResult) -> bool:
        """
        Check for UNKNOWN values in critical fields
        Per VORTEX_CORE_AUTHORITY.md: UNKNOWN ≠ LOW ≠ FALSE ≠ 0
        """
        if not finding.ai_analysis:
            return False  # No AI analysis, no UNKNOWN values to check
        
        ai = finding.ai_analysis
        
        # Check critical fields
        unknown_checks = [
            ai.impact == "UNKNOWN",
            ai.exploitability is None,
            ai.reportability is None,
        ]
        
        return any(unknown_checks)
    
    def _has_deterministic_evidence(self, finding: AssessmentResult) -> bool:
        """
        Check if evidence meets deterministic standards
        Per VORTEX_EVIDENCE_STANDARDS.md: Deterministic evidence required for SUBMIT_READY
        """
        if not finding.verification_result:
            return False
        
        # High determinism match types
        deterministic_types = ["exact_regex", "structural_differential"]
        
        return (
            finding.verification_result.match_type.value in deterministic_types or
            finding.verification_result.determinism_score >= AUTHORITY_THRESHOLDS.submit_ready_evidence_determinism
        )
    
    def make_final_determination(self, finding: AssessmentResult) -> VerificationStatus:
        """
        Make final authoritative determination following strict hierarchy
        Per VORTEX_CORE_AUTHORITY.md: System authority + AI advisory
        
        Returns: Final VerificationStatus
        """
        logger.info(
            "Making final determination with authority enforcement",
            finding_id=str(finding.id),
            current_status=finding.status
        )
        
        # HARD STOP 1: No system verification = manual review
        if not finding.verification_result or not finding.verification_result.success:
            logger.info(
                "Routing to manual: No system verification",
                finding_id=str(finding.id)
            )
            return VerificationStatus.NEEDS_MANUAL
        
        # HARD STOP 2: Low confidence system verification = manual review
        if finding.verification_result.confidence < AUTHORITY_THRESHOLDS.submit_ready_system_confidence:
            logger.info(
                "Routing to manual: Low system confidence",
                finding_id=str(finding.id),
                confidence=finding.verification_result.confidence
            )
            return VerificationStatus.NEEDS_MANUAL
        
        # HARD STOP 3: UNKNOWN values = manual review (NEVER proceed with unknowns)
        if self._has_unknown_values(finding):
            logger.info(
                "Routing to manual: UNKNOWN values present",
                finding_id=str(finding.id)
            )
            return VerificationStatus.NEEDS_MANUAL
        
        # V11.1 FASTPATH: Strong system verification + AI advisory support
        if (finding.verification_result.confidence >= AUTHORITY_THRESHOLDS.fastpath_system_confidence and
            self._has_deterministic_evidence(finding)):
            
            # AI can provide supportive evidence (ADVISORY role only)
            if (finding.ai_analysis and 
                finding.ai_analysis.success and
                finding.ai_analysis.verdict.value == "CONFIRMED" and
                finding.ai_analysis.confidence >= AUTHORITY_THRESHOLDS.submit_ready_ai_support_confidence):
                
                logger.info(
                    "SUBMIT_READY: Strong system evidence + AI advisory support",
                    finding_id=str(finding.id),
                    system_confidence=finding.verification_result.confidence,
                    ai_confidence=finding.ai_analysis.confidence
                )
                return VerificationStatus.SUBMIT_READY
            
            # Very strong system evidence can proceed without AI (system authority)
            if finding.verification_result.confidence >= AUTHORITY_THRESHOLDS.very_strong_system_confidence:
                logger.info(
                    "SUBMIT_READY: Very strong system evidence (AI-independent)",
                    finding_id=str(finding.id),
                    confidence=finding.verification_result.confidence
                )
                return VerificationStatus.SUBMIT_READY
        
        # AI_FAILED doesn't penalize strong system evidence
        if (finding.status == VerificationStatus.AI_FAILED and
            finding.verification_result.confidence >= AUTHORITY_THRESHOLDS.fastpath_system_confidence and
            self._has_deterministic_evidence(finding)):
            
            logger.info(
                "SUBMIT_READY: Strong system evidence compensates for AI failure",
                finding_id=str(finding.id),
                confidence=finding.verification_result.confidence
            )
            return VerificationStatus.SUBMIT_READY
        
        # Default: Route to manual review for safety
        logger.info(
            "Routing to manual: Safety default",
            finding_id=str(finding.id),
            system_confidence=finding.verification_result.confidence if finding.verification_result else 0.0
        )
        return VerificationStatus.NEEDS_MANUAL
    
    def _record_violation(self, finding: AssessmentResult, violations: list) -> None:
        """Record authority violation for audit."""
        self.decision_history.append({
            'timestamp': datetime.utcnow(),
            'finding_id': str(finding.id),
            'decision': 'VIOLATION',
            'violations': violations,
            'attempted_status': finding.status
        })
    
    def _record_valid_decision(self, finding: AssessmentResult) -> None:
        """Record valid authority decision."""
        self.decision_history.append({
            'timestamp': datetime.utcnow(),
            'finding_id': str(finding.id),
            'decision': 'VALID',
            'authority_level': self.get_finding_authority_level(finding),
            'status': finding.status
        })
    
    def get_violation_rate(self) -> float:
        """Get authority violation rate."""
        if not self.decision_history:
            return 0.0
        
        violations = sum(1 for d in self.decision_history if d['decision'] == 'VIOLATION')
        return violations / len(self.decision_history)
    
    def reset_metrics(self) -> None:
        """Reset tracking metrics."""
        self.violation_count = 0
        self.decision_history = []


def get_authority_level(finding: AssessmentResult) -> AuthorityLevel:
    """
    Get authority level for finding
    Convenience function for external use
    """
    enforcer = AuthorityHierarchyEnforcer()
    return enforcer.get_finding_authority_level(finding)


def compare_authority_levels(level1: AuthorityLevel, level2: AuthorityLevel) -> int:
    """
    Compare two authority levels
    Returns: -1 if level1 < level2, 0 if equal, 1 if level1 > level2
    """
    enforcer = AuthorityHierarchyEnforcer()
    priority1 = enforcer.AUTHORITY_PRIORITY.get(level1, 999)
    priority2 = enforcer.AUTHORITY_PRIORITY.get(level2, 999)
    
    if priority1 < priority2:
        return 1  # level1 has higher authority
    elif priority1 > priority2:
        return -1  # level2 has higher authority
    else:
        return 0  # equal


def is_authority_sufficient(
    required_level: AuthorityLevel,
    actual_level: AuthorityLevel
) -> bool:
    """
    Check if actual authority level meets or exceeds required level
    """
    return compare_authority_levels(actual_level, required_level) >= 0


# Global enforcer instance for module-level access
global_authority_enforcer = AuthorityHierarchyEnforcer()