"""
VORTEX Quality Assurance Engine - V17.0 ULTIMATE
Multi-dimensional validation for submission readiness

Per .clinerules:
- Multi-dimensional validation
- Evidence quality scoring
- Submission readiness checks
- Authority compliance verification
- Quality metrics tracking

FEATURES:
- Comprehensive quality validation
- Evidence determinism scoring
- SUBMIT_READY gate validation
- Quality improvement recommendations
- Detailed quality reports
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from domain.models import AssessmentResult, EvidenceQuality
from domain.enums import VerificationStatus, FindingSeverity, EvidenceLevel
from config.constants import AUTHORITY_THRESHOLDS, EVIDENCE_STANDARDS, VULN_SPECIFIC_EVIDENCE
from core.authority.hierarchy import global_authority_enforcer
from core.evidence.standards import global_evidence_validator
from core.exceptions import QualityError

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Single validation check result."""
    check_name: str
    passed: bool
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class QualityScore:
    """Quality scoring result."""
    overall_score: float  # 0.0-1.0
    dimension_scores: Dict[str, float]
    grade: str  # A+, A, B+, B, C, D, F
    
    # Score breakdown
    authority_score: float = 0.0
    evidence_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    finding_id: str
    submission_ready: bool
    quality_score: QualityScore
    
    # Validation results
    passed_checks: List[ValidationCheck]
    failed_checks: List[ValidationCheck]
    warnings: List[ValidationCheck]
    
    # Recommendations
    improvements_needed: List[str]
    blocking_issues: List[str]
    
    # Metadata
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessor: str = "QualityAssuranceEngine"
    
    def get_summary(self) -> str:
        """Get quality report summary."""
        return f"Quality: {self.quality_score.grade} ({self.quality_score.overall_score:.2%}) - " \
               f"{len(self.passed_checks)} passed, {len(self.failed_checks)} failed"


class QualityDimensionValidator:
    """
    Multi-dimensional quality validator.
    
    Validates findings across multiple quality dimensions:
    1. Authority Compliance
    2. Evidence Quality
    3. Completeness
    4. Accuracy
    5. Reportability
    """
    
    def validate_authority_compliance(self, finding: AssessmentResult) -> List[ValidationCheck]:
        """
        Validate authority hierarchy compliance.
        
        Per VORTEX_CORE_AUTHORITY.md: Strict authority validation
        """
        checks = []
        
        # Check 1: System verification exists
        if not finding.verification_result or not finding.verification_result.success:
            checks.append(ValidationCheck(
                check_name="system_verification_exists",
                passed=False,
                severity="CRITICAL",
                message="No successful system verification",
                details={'status': 'missing'}
            ))
        else:
            checks.append(ValidationCheck(
                check_name="system_verification_exists",
                passed=True,
                severity="CRITICAL",
                message="System verification present"
            ))
            
            # Check 2: System confidence threshold
            if finding.verification_result.confidence < AUTHORITY_THRESHOLDS.submit_ready_system_confidence:
                checks.append(ValidationCheck(
                    check_name="system_confidence_threshold",
                    passed=False,
                    severity="CRITICAL",
                    message=f"System confidence {finding.verification_result.confidence:.2f} below threshold {AUTHORITY_THRESHOLDS.submit_ready_system_confidence}",
                    details={'confidence': finding.verification_result.confidence}
                ))
            else:
                checks.append(ValidationCheck(
                    check_name="system_confidence_threshold",
                    passed=True,
                    severity="CRITICAL",
                    message="System confidence meets threshold"
                ))
        
        # Check 3: No UNKNOWN values
        has_unknowns = global_authority_enforcer._has_unknown_values(finding)
        checks.append(ValidationCheck(
            check_name="no_unknown_values",
            passed=not has_unknowns,
            severity="CRITICAL" if has_unknowns else "HIGH",
            message="UNKNOWN values present" if has_unknowns else "No UNKNOWN values",
            details={'has_unknowns': has_unknowns}
        ))
        
        # Check 4: Deterministic evidence
        is_deterministic = global_authority_enforcer._is_deterministic_evidence(finding)
        checks.append(ValidationCheck(
            check_name="deterministic_evidence",
            passed=is_deterministic,
            severity="CRITICAL" if not is_deterministic else "HIGH",
            message="Deterministic evidence present" if is_deterministic else "Evidence not deterministic",
            details={'is_deterministic': is_deterministic}
        ))
        
        # Check 5: AI role is advisory only (when AI used)
        if finding.ai_analysis:
            ai_advisory_only = not finding.ai_analysis.is_authoritative
            checks.append(ValidationCheck(
                check_name="ai_advisory_only",
                passed=ai_advisory_only,
                severity="CRITICAL",
                message="AI role is advisory" if ai_advisory_only else "AI incorrectly marked authoritative",
                details={'is_authoritative': finding.ai_analysis.is_authoritative}
            ))
        
        return checks
    
    def validate_evidence_quality(self, finding: AssessmentResult) -> List[ValidationCheck]:
        """
        Validate evidence quality.
        
        Per VORTEX_EVIDENCE_STANDARDS.md: Evidence must meet deterministic standards
        """
        checks = []
        
        # Calculate evidence determinism
        determinism_score = global_evidence_validator.assess_evidence_determinism(finding)
        
        # Check 1: Evidence determinism score
        min_determinism = EVIDENCE_STANDARDS.deterministic_min_score
        checks.append(ValidationCheck(
            check_name="evidence_determinism_score",
            passed=determinism_score >= min_determinism,
            severity="CRITICAL",
            message=f"Evidence determinism: {determinism_score:.2f} (min: {min_determinism})",
            details={'score': determinism_score, 'threshold': min_determinism}
        ))
        
        # Check 2: Evidence length
        min_length = 20  # Minimum evidence length
        evidence_length = len(finding.evidence) if finding.evidence else 0
        checks.append(ValidationCheck(
            check_name="evidence_length",
            passed=evidence_length >= min_length,
            severity="HIGH",
            message=f"Evidence length: {evidence_length} chars",
            details={'length': evidence_length}
        ))
        
        # Check 3: Vulnerability-specific evidence
        if finding.vulnerability_type:
            vuln_type = finding.vulnerability_type.lower()
            if vuln_type in VULN_SPECIFIC_EVIDENCE:
                criteria = VULN_SPECIFIC_EVIDENCE[vuln_type]
                evidence_lower = finding.evidence.lower() if finding.evidence else ""
                
                # Check for deterministic indicators
                indicators_found = sum(
                    1 for indicator in criteria['deterministic_indicators']
                    if indicator in evidence_lower
                )
                
                checks.append(ValidationCheck(
                    check_name="vuln_specific_evidence",
                    passed=indicators_found >= 1,
                    severity="MEDIUM",
                    message=f"Vuln-specific indicators: {indicators_found}",
                    details={'indicators_found': indicators_found}
                ))
        
        # Check 4: Behavioral analysis (if present)
        if finding.behavioral_analysis:
            has_uncertainties = len(finding.behavioral_analysis.uncertainty_factors) > 0
            checks.append(ValidationCheck(
                check_name="behavioral_uncertainty_acknowledged",
                passed=has_uncertainties or finding.behavioral_analysis.causation_determination == "UNKNOWN - requires human expert analysis",
                severity="MEDIUM",
                message="Behavioral uncertainty properly acknowledged" if has_uncertainties else "Behavioral analysis complete",
                details={'uncertainty_factors': len(finding.behavioral_analysis.uncertainty_factors) if finding.behavioral_analysis else 0}
            ))
        
        return checks
    
    def validate_completeness(self, finding: AssessmentResult) -> List[ValidationCheck]:
        """Validate finding completeness."""
        checks = []
        
        # Required fields check
        required_fields = {
            'url': finding.url,
            'finding_type': finding.finding_type,
            'severity': finding.severity,
            'evidence': finding.evidence,
            'status': finding.status
        }
        
        for field_name, field_value in required_fields.items():
            checks.append(ValidationCheck(
                check_name=f"required_field_{field_name}",
                passed=field_value is not None and field_value != "",
                severity="CRITICAL",
                message=f"Field '{field_name}' present" if field_value else f"Field '{field_name}' missing"
            ))
        
        # Optional but valuable fields
        valuable_fields = {
            'vulnerable_parameter': finding.vulnerable_parameter,
            'payload': finding.payload,
        }
        
        for field_name, field_value in valuable_fields.items():
            checks.append(ValidationCheck(
                check_name=f"valuable_field_{field_name}",
                passed=field_value is not None and field_value != "",
                severity="MEDIUM",
                message=f"Field '{field_name}' present" if field_value else f"Field '{field_name}' missing"
            ))
        
        return checks
    
    def validate_accuracy(self, finding: AssessmentResult) -> List[ValidationCheck]:
        """Validate finding accuracy indicators."""
        checks = []
        
        # Check 1: Heuristic score reasonableness
        if finding.heuristic_score > 0:
            checks.append(ValidationCheck(
                check_name="heuristic_score_reasonable",
                passed=0.0 <= finding.heuristic_score <= 1.0,
                severity="HIGH",
                message=f"Heuristic score: {finding.heuristic_score:.2f}",
                details={'score': finding.heuristic_score}
            ))
        
        # Check 2: AI confidence reasonableness (if present)
        if finding.ai_analysis and finding.ai_analysis.confidence > 0:
            checks.append(ValidationCheck(
                check_name="ai_confidence_reasonable",
                passed=0.0 <= finding.ai_analysis.confidence <= 1.0,
                severity="MEDIUM",
                message=f"AI confidence: {finding.ai_analysis.confidence:.2f}"
            ))
        
        # Check 3: Consistency between scores
        if finding.verification_result and finding.ai_analysis:
            sys_conf = finding.verification_result.confidence
            ai_conf = finding.ai_analysis.confidence
            
            # They should be relatively aligned
            difference = abs(sys_conf - ai_conf)
            checks.append(ValidationCheck(
                check_name="score_consistency",
                passed=difference < 0.3,  # Less than 30% difference
                severity="LOW",
                message=f"Score consistency: {difference:.2f} difference",
                details={'system': sys_conf, 'ai': ai_conf}
            ))
        
        return checks


class SubmissionReadinessValidator:
    """
    SUBMIT_READY gate validator.
    
    Final validation before marking finding as submission-ready.
    Per .clinerules: All requirements must be met for SUBMIT_READY.
    """
    
    def validate_submit_ready(self, finding: AssessmentResult) -> Tuple[bool, List[str]]:
        """
        Validate finding meets all SUBMIT_READY requirements.
        
        Returns:
            (ready, blocking_issues)
        """
        blocking_issues = []
        
        # REQUIREMENT 1: System verification successful
        if not finding.verification_result or not finding.verification_result.success:
            blocking_issues.append("No successful system verification")
        
        # REQUIREMENT 2: Confidence threshold
        if finding.verification_result:
            if finding.verification_result.confidence < AUTHORITY_THRESHOLDS.submit_ready_system_confidence:
                blocking_issues.append(f"System confidence {finding.verification_result.confidence:.2f} below {AUTHORITY_THRESHOLDS.submit_ready_system_confidence}")
        
        # REQUIREMENT 3: No UNKNOWN values
        if global_authority_enforcer._has_unknown_values(finding):
            blocking_issues.append("UNKNOWN values present in critical fields")
        
        # REQUIREMENT 4: Deterministic evidence
        if not global_authority_enforcer._is_deterministic_evidence(finding):
            blocking_issues.append("Evidence does not meet deterministic standards")
        
        # REQUIREMENT 5: Evidence quality
        determinism_score = global_evidence_validator.assess_evidence_determinism(finding)
        if determinism_score < AUTHORITY_THRESHOLDS.submit_ready_evidence_determinism:
            blocking_issues.append(f"Evidence determinism {determinism_score:.2f} below threshold {AUTHORITY_THRESHOLDS.submit_ready_evidence_determinism}")
        
        # REQUIREMENT 6: Complete required fields
        if not all([finding.url, finding.finding_type, finding.severity, finding.evidence]):
            blocking_issues.append("Missing required fields")
        
        # REQUIREMENT 7: Legal compliance (if checked)
        if hasattr(finding, 'compliance_status'):
            from domain.enums import ComplianceStatus
            if finding.compliance_status == ComplianceStatus.VIOLATION:
                blocking_issues.append("Legal compliance violation detected")
        
        is_ready = len(blocking_issues) == 0
        return is_ready, blocking_issues


class QualityAssuranceEngine:
    """
    Quality assurance engine for submission readiness.
    
    RESPONSIBILITIES:
    - Multi-dimensional quality validation
    - Evidence quality scoring
    - Submission readiness verification
    - Quality improvement recommendations
    - Quality metrics tracking
    
    Per .clinerules: Critical quality gate before submission
    """
    
    def __init__(self):
        self.dimension_validator = QualityDimensionValidator()
        self.readiness_validator = SubmissionReadinessValidator()
        
        # Statistics
        self.total_assessments = 0
        self.quality_scores: List[float] = []
        
        logger.info("Quality Assurance Engine initialized")
    
    def assess_quality(self, finding: AssessmentResult) -> QualityReport:
        """
        Comprehensive quality assessment.
        
        Args:
            finding: Finding to assess
            
        Returns:
            Detailed quality report
        """
        self.total_assessments += 1
        
        all_checks = []
        
        # Run all validation dimensions
        all_checks.extend(self.dimension_validator.validate_authority_compliance(finding))
        all_checks.extend(self.dimension_validator.validate_evidence_quality(finding))
        all_checks.extend(self.dimension_validator.validate_completeness(finding))
        all_checks.extend(self.dimension_validator.validate_accuracy(finding))
        
        # Separate by result
        passed_checks = [c for c in all_checks if c.passed]
        failed_checks = [c for c in all_checks if not c.passed and c.severity in ['CRITICAL', 'HIGH']]
        warnings = [c for c in all_checks if not c.passed and c.severity in ['MEDIUM', 'LOW']]
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(all_checks, finding)
        self.quality_scores.append(quality_score.overall_score)
        
        # Check submission readiness
        is_ready, blocking_issues = self.readiness_validator.validate_submit_ready(finding)
        
        # Generate improvements needed
        improvements = self._generate_improvements(failed_checks, warnings)
        
        # Create report
        report = QualityReport(
            finding_id=str(finding.id),
            submission_ready=is_ready,
            quality_score=quality_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            improvements_needed=improvements,
            blocking_issues=blocking_issues
        )
        
        # Log assessment
        logger.info(f"Quality assessment for {finding.id}: {report.get_summary()}")
        
        return report
    
    def _calculate_quality_score(self, checks: List[ValidationCheck], finding: AssessmentResult) -> QualityScore:
        """Calculate comprehensive quality score."""
        
        # Dimension scores
        authority_checks = [c for c in checks if 'authority' in c.check_name or 'system' in c.check_name]
        evidence_checks = [c for c in checks if 'evidence' in c.check_name]
        completeness_checks = [c for c in checks if 'field' in c.check_name]
        accuracy_checks = [c for c in checks if 'score' in c.check_name or 'consistency' in c.check_name]
        
        def score_checks(check_list: List[ValidationCheck]) -> float:
            if not check_list:
                return 1.0
            
            # Weight by severity
            weights = {'CRITICAL': 1.0, 'HIGH': 0.8, 'MEDIUM': 0.5, 'LOW': 0.3}
            
            total_weight = sum(weights.get(c.severity, 0.5) for c in check_list)
            passed_weight = sum(weights.get(c.severity, 0.5) for c in check_list if c.passed)
            
            return passed_weight / total_weight if total_weight > 0 else 0.0
        
        authority_score = score_checks(authority_checks)
        evidence_score = score_checks(evidence_checks)
        completeness_score = score_checks(completeness_checks)
        accuracy_score = score_checks(accuracy_checks)
        
        # Overall score (weighted)
        overall = (
            authority_score * 0.35 +      # 35% - most critical
            evidence_score * 0.35 +       # 35% - equally critical
            completeness_score * 0.20 +   # 20% - important
            accuracy_score * 0.10         # 10% - nice to have
        )
        
        # Determine grade
        if overall >= 0.95:
            grade = "A+"
        elif overall >= 0.90:
            grade = "A"
        elif overall >= 0.85:
            grade = "B+"
        elif overall >= 0.80:
            grade = "B"
        elif overall >= 0.70:
            grade = "C"
        elif overall >= 0.60:
            grade = "D"
        else:
            grade = "F"
        
        return QualityScore(
            overall_score=overall,
            dimension_scores={
                'authority': authority_score,
                'evidence': evidence_score,
                'completeness': completeness_score,
                'accuracy': accuracy_score
            },
            grade=grade,
            authority_score=authority_score,
            evidence_score=evidence_score,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score
        )
    
    def _generate_improvements(self, failed_checks: List[ValidationCheck], warnings: List[ValidationCheck]) -> List[str]:
        """Generate improvement recommendations."""
        improvements = []
        
        # Critical failures
        for check in failed_checks:
            if 'system_verification' in check.check_name:
                improvements.append("Perform system verification with PoC replay")
            elif 'confidence' in check.check_name:
                improvements.append("Gather additional evidence to increase confidence")
            elif 'unknown' in check.check_name:
                improvements.append("Resolve UNKNOWN values through additional analysis")
            elif 'deterministic' in check.check_name:
                improvements.append("Obtain more deterministic evidence (exact matches, structural changes)")
        
        # Warnings
        for check in warnings:
            if 'evidence_length' in check.check_name:
                improvements.append("Capture more detailed evidence")
            elif 'vuln_specific' in check.check_name:
                improvements.append("Include vulnerability-specific indicators in evidence")
        
        return list(set(improvements))  # Remove duplicates
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """Get quality assessment statistics."""
        if not self.quality_scores:
            return {
                'total_assessments': 0,
                'average_score': 0.0
            }
        
        avg_score = sum(self.quality_scores) / len(self.quality_scores)
        
        return {
            'total_assessments': self.total_assessments,
            'average_score': avg_score,
            'scores_tracked': len(self.quality_scores),
            'high_quality_rate': sum(1 for s in self.quality_scores if s >= 0.80) / len(self.quality_scores)
        }


# Global quality assurance engine instance
global_quality_assurance = QualityAssuranceEngine()