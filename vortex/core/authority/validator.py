"""
VORTEX Authority Validator - V17.0 ULTIMATE
Authority validation logic per VORTEX_CORE_AUTHORITY.md

CRITICAL: Validates all authority transitions and decisions
"""

import structlog
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from domain.enums import (
    VerificationStatus, AuthorityLevel, AIVerdict, 
    ConfidenceSource, MatchType
)
from domain.models import AssessmentResult, VerificationResult, AIAnalysisResult
from config.constants import AUTHORITY_THRESHOLDS, EVIDENCE_STANDARDS

logger = structlog.get_logger()


class AuthorityValidator:
    """
    Validates authority compliance for all system decisions
    Per VORTEX_CORE_AUTHORITY.md: Every decision must follow authority hierarchy
    """
    
    def __init__(self):
        self.validation_history = []
    
    def validate_status_transition(
        self,
        finding: AssessmentResult,
        from_status: VerificationStatus,
        to_status: VerificationStatus
    ) -> Tuple[bool, str]:
        """
        Validate status transition follows authority rules
        
        Returns: (is_valid, reason)
        """
        # Define valid transitions per VORTEX_WORKFLOW_LIFECYCLE.md
        valid_transitions = self._get_valid_transitions(from_status)
        
        if to_status not in valid_transitions:
            reason = f"Invalid transition: {from_status} → {to_status}"
            logger.error(
                "Authority validation failed: Invalid state transition",
                finding_id=str(finding.id),
                from_status=from_status,
                to_status=to_status
            )
            self._record_validation(finding, "TRANSITION", False, reason)
            return False, reason
        
        # Additional validation for SUBMIT_READY transitions
        if to_status == VerificationStatus.SUBMIT_READY:
            valid, reason = self._validate_submit_ready_transition(finding, from_status)
            if not valid:
                self._record_validation(finding, "SUBMIT_READY", False, reason)
                return False, reason
        
        self._record_validation(finding, "TRANSITION", True, f"{from_status} → {to_status}")
        return True, "Valid transition"
    
    def _get_valid_transitions(self, status: VerificationStatus) -> List[VerificationStatus]:
        """Get valid transitions from current status."""
        transitions = {
            VerificationStatus.DETECTED: [
                VerificationStatus.AI_ANALYSIS_PENDING,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE,
                VerificationStatus.ERROR_STATE
            ],
            VerificationStatus.AI_ANALYSIS_PENDING: [
                VerificationStatus.AI_CONFIRMED,
                VerificationStatus.AI_FAILED,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.ERROR_STATE
            ],
            VerificationStatus.AI_CONFIRMED: [
                VerificationStatus.SYSTEM_VERIFICATION_PENDING,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE
            ],
            VerificationStatus.AI_FAILED: [
                VerificationStatus.SYSTEM_VERIFICATION_PENDING,
                VerificationStatus.SUBMIT_READY,  # V11.1: If strong system evidence
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE
            ],
            VerificationStatus.SYSTEM_VERIFICATION_PENDING: [
                VerificationStatus.SYSTEM_VERIFIED,
                VerificationStatus.SYSTEM_VERIFICATION_FAILED,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.ERROR_STATE
            ],
            VerificationStatus.SYSTEM_VERIFIED: [
                VerificationStatus.SUBMIT_READY,  # V11.1: Fastpath enabled
                VerificationStatus.NEEDS_MANUAL
            ],
            VerificationStatus.SYSTEM_VERIFICATION_FAILED: [
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE
            ],
            # Terminal states
            VerificationStatus.SUBMIT_READY: [],
            VerificationStatus.NEEDS_MANUAL: [],
            VerificationStatus.FALSE_POSITIVE: [],
            VerificationStatus.ERROR_STATE: [VerificationStatus.NEEDS_MANUAL]
        }
        
        return transitions.get(status, [])
    
    def _validate_submit_ready_transition(
        self,
        finding: AssessmentResult,
        from_status: VerificationStatus
    ) -> Tuple[bool, str]:
        """
        Validate transition to SUBMIT_READY meets all requirements
        Per VORTEX_CORE_AUTHORITY.md: Strict requirements
        """
        violations = []
        
        # Requirement 1: System verification must exist and succeed
        if not finding.verification_result or not finding.verification_result.success:
            violations.append("No successful system verification")
        
        # Requirement 2: System confidence threshold
        if finding.verification_result:
            if finding.verification_result.confidence < AUTHORITY_THRESHOLDS.submit_ready_system_confidence:
                violations.append(
                    f"System confidence {finding.verification_result.confidence:.2f} "
                    f"< threshold {AUTHORITY_THRESHOLDS.submit_ready_system_confidence}"
                )
        
        # Requirement 3: No UNKNOWN values
        if self._has_unknown_values(finding):
            violations.append("UNKNOWN values present in critical fields")
        
        # Requirement 4: Deterministic evidence
        if not self._has_sufficient_evidence(finding):
            violations.append("Evidence does not meet deterministic standards")
        
        if violations:
            reason = "; ".join(violations)
            logger.warning(
                "SUBMIT_READY validation failed",
                finding_id=str(finding.id),
                violations=violations
            )
            return False, reason
        
        logger.info(
            "SUBMIT_READY validation passed",
            finding_id=str(finding.id),
            from_status=from_status
        )
        return True, "All SUBMIT_READY requirements met"
    
    def _has_unknown_values(self, finding: AssessmentResult) -> bool:
        """Check for UNKNOWN values per VORTEX_CORE_AUTHORITY.md"""
        if not finding.ai_analysis:
            return False
        
        ai = finding.ai_analysis
        return any([
            ai.impact == "UNKNOWN",
            ai.exploitability is None,
            ai.reportability is None
        ])
    
    def _has_sufficient_evidence(self, finding: AssessmentResult) -> bool:
        """Check evidence meets deterministic standards"""
        if not finding.verification_result:
            return False
        
        # Check determinism score
        determinism_score = getattr(finding, 'evidence_determinism_score', 0.0)
        if determinism_score < EVIDENCE_STANDARDS.deterministic_min_score:
            return False
        
        # Check match type
        deterministic_types = [MatchType.EXACT_REGEX, MatchType.STRUCTURAL_DIFFERENTIAL]
        if finding.verification_result.match_type not in deterministic_types:
            # If not deterministic type, must have high confidence
            if finding.verification_result.confidence < AUTHORITY_THRESHOLDS.very_strong_system_confidence:
                return False
        
        return True
    
    def validate_ai_authority_limits(self, finding: AssessmentResult) -> Tuple[bool, List[str]]:
        """
        Validate AI is not being used as authoritative source
        Per VORTEX_AI_INTEGRATION.md: AI is ADVISORY ONLY
        
        Returns: (is_valid, violations)
        """
        violations = []
        
        if not finding.ai_analysis:
            return True, []  # No AI analysis, no violations
        
        # Check 1: AI cannot be sole authority for SUBMIT_READY
        if finding.status == VerificationStatus.SUBMIT_READY:
            if not finding.verification_result or not finding.verification_result.success:
                violations.append(
                    "SUBMIT_READY status without system verification (AI cannot be sole authority)"
                )
        
        # Check 2: AI fields should not be derived
        if finding.ai_analysis.is_fallback_result:
            # Fallback results should have limited data
            if finding.ai_analysis.exploitability is not None:
                violations.append("Fallback AI result has derived exploitability")
            if finding.ai_analysis.reportability is not None:
                violations.append("Fallback AI result has derived reportability")
        
        # Check 3: Heuristic PoCs should never be used for system verification
        if (finding.confidence_source == ConfidenceSource.HEURISTIC_ONLY and
            finding.verification_result and 
            finding.verification_result.replay_attempted):
            violations.append("Heuristic PoC was replayed (FORBIDDEN)")
        
        # Check 4: Malformed JSON recovery should not be authoritative
        if (finding.ai_analysis and 
            hasattr(finding.ai_analysis, 'model_used') and
            'malformed_recovery' in finding.ai_analysis.model_used):
            if finding.status == VerificationStatus.SUBMIT_READY:
                violations.append("SUBMIT_READY with malformed JSON recovery (non-authoritative)")
        
        if violations:
            logger.error(
                "AI authority limit violations detected",
                finding_id=str(finding.id),
                violations=violations
            )
            self._record_validation(finding, "AI_AUTHORITY", False, "; ".join(violations))
            return False, violations
        
        self._record_validation(finding, "AI_AUTHORITY", True, "AI authority limits respected")
        return True, []
    
    def validate_evidence_authority(self, finding: AssessmentResult) -> Tuple[bool, str]:
        """
        Validate evidence meets authority requirements for current status
        Per VORTEX_EVIDENCE_STANDARDS.md
        
        Returns: (is_valid, reason)
        """
        if finding.status == VerificationStatus.SUBMIT_READY:
            required_level = "DETERMINISTIC"
            min_score = EVIDENCE_STANDARDS.deterministic_min_score
        elif finding.status == VerificationStatus.SYSTEM_VERIFIED:
            required_level = "BEHAVIORAL"
            min_score = EVIDENCE_STANDARDS.behavioral_min_score
        elif finding.status == VerificationStatus.AI_CONFIRMED:
            required_level = "PATTERN"
            min_score = EVIDENCE_STANDARDS.pattern_min_score
        else:
            # Other statuses don't have evidence requirements
            return True, "No evidence requirements for this status"
        
        # Calculate evidence score
        evidence_score = self._calculate_evidence_score(finding)
        
        if evidence_score < min_score:
            reason = (
                f"Evidence score {evidence_score:.2f} below required "
                f"{required_level} level ({min_score})"
            )
            logger.warning(
                "Evidence authority validation failed",
                finding_id=str(finding.id),
                status=finding.status,
                evidence_score=evidence_score,
                required_level=required_level
            )
            self._record_validation(finding, "EVIDENCE", False, reason)
            return False, reason
        
        self._record_validation(finding, "EVIDENCE", True, f"Evidence meets {required_level} standard")
        return True, f"Evidence meets {required_level} standard"
    
    def _calculate_evidence_score(self, finding: AssessmentResult) -> float:
        """Calculate overall evidence quality score"""
        score = 0.0
        
        # System verification evidence (highest value)
        if finding.verification_result and finding.verification_result.success:
            if finding.verification_result.match_type == MatchType.EXACT_REGEX:
                score += 0.5
            elif finding.verification_result.match_type == MatchType.STRUCTURAL_DIFFERENTIAL:
                score += 0.4
            elif finding.verification_result.match_type == MatchType.FUZZY_MATCH:
                score += 0.3
        
        # AI analysis evidence (advisory)
        if finding.ai_analysis and not finding.ai_analysis.is_fallback_result:
            if finding.ai_analysis.verdict == AIVerdict.CONFIRMED:
                score += 0.3
            elif finding.ai_analysis.verdict == AIVerdict.LIKELY:
                score += 0.2
        
        # Heuristic evidence (lowest)
        if finding.heuristic_score >= 0.8:
            score += 0.2
        elif finding.heuristic_score >= 0.6:
            score += 0.1
        
        return min(score, 1.0)
    
    def validate_complete_finding(self, finding: AssessmentResult) -> Dict[str, any]:
        """
        Perform complete authority validation on finding
        
        Returns: Validation report with all checks
        """
        report = {
            'finding_id': str(finding.id),
            'timestamp': datetime.utcnow(),
            'status': finding.status,
            'checks': {},
            'overall_valid': True,
            'violations': []
        }
        
        # Check 1: AI authority limits
        ai_valid, ai_violations = self.validate_ai_authority_limits(finding)
        report['checks']['ai_authority'] = {
            'valid': ai_valid,
            'violations': ai_violations
        }
        if not ai_valid:
            report['overall_valid'] = False
            report['violations'].extend(ai_violations)
        
        # Check 2: Evidence authority
        evidence_valid, evidence_reason = self.validate_evidence_authority(finding)
        report['checks']['evidence_authority'] = {
            'valid': evidence_valid,
            'reason': evidence_reason
        }
        if not evidence_valid:
            report['overall_valid'] = False
            report['violations'].append(evidence_reason)
        
        # Check 3: SUBMIT_READY specific validation
        if finding.status == VerificationStatus.SUBMIT_READY:
            submit_valid, submit_reason = self._validate_submit_ready_transition(
                finding, finding.status
            )
            report['checks']['submit_ready'] = {
                'valid': submit_valid,
                'reason': submit_reason
            }
            if not submit_valid:
                report['overall_valid'] = False
                report['violations'].append(submit_reason)
        
        logger.info(
            "Complete authority validation performed",
            finding_id=str(finding.id),
            overall_valid=report['overall_valid'],
            violation_count=len(report['violations'])
        )
        
        return report
    
    def _record_validation(
        self,
        finding: AssessmentResult,
        check_type: str,
        valid: bool,
        reason: str
    ) -> None:
        """Record validation result for audit."""
        self.validation_history.append({
            'timestamp': datetime.utcnow(),
            'finding_id': str(finding.id),
            'check_type': check_type,
            'valid': valid,
            'reason': reason,
            'status': finding.status
        })
    
    def get_validation_stats(self) -> Dict[str, any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'success_rate': 1.0,
                'failure_rate': 0.0
            }
        
        total = len(self.validation_history)
        failures = sum(1 for v in self.validation_history if not v['valid'])
        
        return {
            'total_validations': total,
            'success_rate': (total - failures) / total,
            'failure_rate': failures / total,
            'failure_count': failures
        }


# Global validator instance
global_authority_validator = AuthorityValidator()