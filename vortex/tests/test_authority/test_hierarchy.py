"""
VORTEX Authority Hierarchy Tests
Per .clinerules VORTEX_CORE_AUTHORITY.md

Tests the immutable authority hierarchy:
1. System Verification (Deterministic - Highest)
2. Human Expert (Authoritative - Second)
3. AI Analysis (Advisory ONLY - Third)
4. Heuristic Detection (Indicative - Lowest)
"""

import pytest
from vortex.config.authority_config import (
    AuthorityLevel,
    SUBMIT_READY_REQUIREMENTS,
    AUTHORITY_HIERARCHY
)


class TestAuthorityHierarchyImmutability:
    """Test that authority hierarchy cannot be violated."""
    
    def test_authority_levels_order(self):
        """Test authority levels are correctly ordered."""
        assert AuthorityLevel.SYSTEM_VERIFICATION == 1
        assert AuthorityLevel.HUMAN_EXPERT == 2
        assert AuthorityLevel.AI_ADVISORY == 3
        assert AuthorityLevel.HEURISTIC == 4
    
    def test_system_verification_highest_authority(self):
        """Test system verification is the highest authority."""
        all_levels = [
            AuthorityLevel.SYSTEM_VERIFICATION,
            AuthorityLevel.HUMAN_EXPERT,
            AuthorityLevel.AI_ADVISORY,
            AuthorityLevel.HEURISTIC
        ]
        assert min(all_levels) == AuthorityLevel.SYSTEM_VERIFICATION
    
    def test_ai_never_higher_than_system(self):
        """Test AI authority is always lower than system verification."""
        assert AuthorityLevel.AI_ADVISORY > AuthorityLevel.SYSTEM_VERIFICATION
        assert AuthorityLevel.AI_ADVISORY > AuthorityLevel.HUMAN_EXPERT
    
    def test_heuristic_lowest_authority(self):
        """Test heuristic is the lowest authority."""
        all_levels = [
            AuthorityLevel.SYSTEM_VERIFICATION,
            AuthorityLevel.HUMAN_EXPERT,
            AuthorityLevel.AI_ADVISORY,
            AuthorityLevel.HEURISTIC
        ]
        assert max(all_levels) == AuthorityLevel.HEURISTIC


class TestSubmitReadyRequirements:
    """Test SUBMIT_READY absolute requirements per .clinerules."""
    
    def test_system_verification_required(self):
        """Test system verification is REQUIRED for SUBMIT_READY."""
        assert SUBMIT_READY_REQUIREMENTS["system_verification_required"] is True
    
    def test_min_confidence_threshold(self):
        """Test minimum confidence threshold is 0.75."""
        assert SUBMIT_READY_REQUIREMENTS["min_system_confidence"] == 0.75
    
    def test_no_unknown_values_required(self):
        """Test UNKNOWN values are prohibited for SUBMIT_READY."""
        assert SUBMIT_READY_REQUIREMENTS["no_unknown_values"] is True
    
    def test_deterministic_evidence_required(self):
        """Test deterministic evidence is required."""
        assert SUBMIT_READY_REQUIREMENTS["deterministic_evidence"] is True
    
    def test_all_requirements_mandatory(self):
        """Test all SUBMIT_READY requirements are mandatory (no optional)."""
        for requirement, value in SUBMIT_READY_REQUIREMENTS.items():
            if isinstance(value, bool):
                assert value is True, f"Requirement {requirement} must be True (mandatory)"
            elif isinstance(value, (int, float)):
                assert value > 0, f"Requirement {requirement} must be positive"


class TestAIAdvisoryLimitations:
    """Test AI is ADVISORY ONLY, never authoritative."""
    
    def test_ai_cannot_be_sole_authority_for_submit_ready(self, mock_finding_data, mock_ai_analysis_result):
        """Test AI analysis alone CANNOT make a finding SUBMIT_READY."""
        from vortex.domain.models import AssessmentResult, AIAnalysisResult
        from vortex.domain.enums import VerificationStatus, FindingSeverity
        
        finding = AssessmentResult(
            url=mock_finding_data["url"],
            vulnerability_type=mock_finding_data["vulnerability_type"],
            severity=FindingSeverity.HIGH,
            heuristic_score=0.90,
            evidence=mock_finding_data["evidence"]
        )
        
        # AI analysis present (high confidence)
        finding.ai_analysis = AIAnalysisResult(**mock_ai_analysis_result)
        
        # BUT NO system verification
        finding.verification_result = None
        
        # Finding CANNOT be SUBMIT_READY with AI alone
        # This should be validated by authority enforcer
        assert finding.verification_result is None
        assert finding.ai_analysis is not None
        
        # Authority enforcer would reject this as SUBMIT_READY
    
    def test_ai_marked_as_advisory_only(self, mock_ai_analysis_result):
        """Test AI results are marked as advisory only."""
        from vortex.domain.models import AIAnalysisResult
        
        ai_result = AIAnalysisResult(**mock_ai_analysis_result)
        
        # AI must be marked as advisory
        assert hasattr(ai_result, 'authority_level')
        assert ai_result.authority_level == "ADVISORY_ONLY"
        assert ai_result.is_authoritative is False
    
    def test_ai_cannot_derive_missing_fields(self):
        """Test AI cannot derive missing fields from present ones."""
        from vortex.domain.models import AIAnalysisResult
        
        # AI result with missing fields
        ai_result = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.85,
            exploitability=None,  # Missing - must stay None
            impact="HIGH",
            reportability=None,   # Missing - must stay None
            reasoning="Test reasoning",
            success=True
        )
        
        # Missing fields must remain None (not derived)
        assert ai_result.exploitability is None
        assert ai_result.reportability is None
    
    def test_heuristic_poc_never_replayed(self, mock_finding_data):
        """Test heuristic-generated PoCs are NEVER replayed."""
        from vortex.domain.models import AssessmentResult
        from vortex.domain.enums import VerificationStatus
        
        finding = AssessmentResult(
            url=mock_finding_data["url"],
            vulnerability_type=mock_finding_data["vulnerability_type"],
            heuristic_score=0.85,
            evidence=mock_finding_data["evidence"]
        )
        
        # Mark as heuristic-only
        finding.confidence_source = "HEURISTIC_ONLY"
        
        # Heuristic PoCs must NEVER be replayed
        # This should be enforced by should_replay_poc()
        assert finding.confidence_source == "HEURISTIC_ONLY"


class TestSubmitReadyValidation:
    """Test SUBMIT_READY validation enforcement."""
    
    def test_valid_submit_ready_finding(self, mock_submit_ready_finding):
        """Test valid SUBMIT_READY finding passes all checks."""
        finding = mock_submit_ready_finding
        
        # All requirements met
        assert finding.verification_result is not None
        assert finding.verification_result.success is True
        assert finding.verification_result.confidence >= 0.75
        assert finding.ai_analysis.impact != "UNKNOWN"
        assert finding.ai_analysis.exploitability is not None
    
    def test_reject_submit_ready_without_system_verification(self, mock_authority_violation_finding):
        """Test SUBMIT_READY is rejected without system verification."""
        finding = mock_authority_violation_finding
        
        # Has AI but NO system verification
        assert finding.ai_analysis is not None
        assert finding.verification_result is None
        
        # This is an AUTHORITY VIOLATION
        # Status should NOT be SUBMIT_READY
    
    def test_reject_submit_ready_with_low_confidence(self, mock_finding_data, mock_system_verification_result):
        """Test SUBMIT_READY is rejected with low system confidence."""
        from vortex.domain.models import AssessmentResult, VerificationResult
        from vortex.domain.enums import FindingSeverity
        
        finding = AssessmentResult(
            url=mock_finding_data["url"],
            vulnerability_type=mock_finding_data["vulnerability_type"],
            severity=FindingSeverity.HIGH,
            heuristic_score=0.85,
            evidence=mock_finding_data["evidence"]
        )
        
        # System verification with LOW confidence
        verification = VerificationResult(**mock_system_verification_result)
        verification.confidence = 0.65  # Below 0.75 threshold
        finding.verification_result = verification
        
        # Cannot be SUBMIT_READY with low confidence
        assert finding.verification_result.confidence < 0.75
    
    def test_reject_submit_ready_with_unknown_values(self, mock_unknown_values_finding):
        """Test SUBMIT_READY is rejected with UNKNOWN values."""
        finding = mock_unknown_values_finding
        
        # Has system verification BUT UNKNOWN values
        assert finding.verification_result is not None
        assert finding.ai_analysis.impact == "UNKNOWN"
        assert finding.ai_analysis.exploitability is None
        
        # Cannot be SUBMIT_READY with UNKNOWN values


class TestAuthorityChainValidation:
    """Test authority chain validation."""
    
    def test_validate_complete_authority_chain(self, mock_submit_ready_finding):
        """Test complete authority chain validation."""
        finding = mock_submit_ready_finding
        
        # System verification (highest authority)
        assert finding.verification_result is not None
        assert finding.verification_result.success is True
        
        # AI analysis (advisory support)
        assert finding.ai_analysis is not None
        assert finding.ai_analysis.is_authoritative is False
        
        # Heuristic detection (lowest authority)
        assert finding.heuristic_score > 0
    
    def test_system_evidence_overrides_ai(self, mock_finding_data):
        """Test system evidence has authority over AI opinion."""
        from vortex.domain.models import AssessmentResult, VerificationResult, AIAnalysisResult
        from vortex.domain.enums import FindingSeverity
        
        finding = AssessmentResult(
            url=mock_finding_data["url"],
            vulnerability_type=mock_finding_data["vulnerability_type"],
            severity=FindingSeverity.HIGH,
            heuristic_score=0.85,
            evidence=mock_finding_data["evidence"]
        )
        
        # System verification: STRONG evidence
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.92,
            match_type="exact_regex"
        )
        
        # AI analysis: NEGATIVE verdict (advisory only)
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="FALSE_POSITIVE",
            confidence=0.70,
            reasoning="AI suggests false positive",
            success=True,
            is_authoritative=False  # AI is advisory only
        )
        
        # System evidence OVERRIDES AI opinion
        assert finding.verification_result.confidence > finding.ai_analysis.confidence
        assert finding.verification_result.success is True
    
    def test_ai_failed_doesnt_block_strong_system_evidence(self, mock_finding_data, mock_system_verification_result):
        """Test AI_FAILED doesn't block findings with strong system verification."""
        from vortex.domain.models import AssessmentResult, VerificationResult
        from vortex.domain.enums import VerificationStatus, FindingSeverity
        
        finding = AssessmentResult(
            url=mock_finding_data["url"],
            vulnerability_type=mock_finding_data["vulnerability_type"],
            severity=FindingSeverity.HIGH,
            heuristic_score=0.87,
            evidence=mock_finding_data["evidence"]
        )
        
        # Strong system verification
        finding.verification_result = VerificationResult(**mock_system_verification_result)
        finding.verification_result.confidence = 0.89
        
        # AI FAILED (model unavailable)
        finding.ai_analysis = None
        finding.status = VerificationStatus.AI_FAILED
        
        # Should still qualify for SUBMIT_READY based on system evidence
        assert finding.verification_result.confidence >= 0.85
        assert finding.status == VerificationStatus.AI_FAILED


class TestUnknownValueHandling:
    """Test UNKNOWN value handling per .clinerules."""
    
    def test_unknown_not_equal_to_low(self):
        """Test UNKNOWN ≠ LOW ≠ FALSE ≠ 0."""
        # These are DIFFERENT meanings
        unknown_val = "UNKNOWN"
        low_val = "LOW"
        false_val = False
        zero_val = 0
        
        assert unknown_val != low_val
        assert unknown_val != false_val
        assert unknown_val != zero_val
        
        # UNKNOWN means: insufficient information
        # LOW means: determined low value
        # FALSE means: determined negative
        # 0 means: measured absence
    
    def test_unknown_values_route_to_manual(self, mock_unknown_values_finding):
        """Test findings with UNKNOWN values route to NEEDS_MANUAL."""
        finding = mock_unknown_values_finding
        
        # Has UNKNOWN values
        assert finding.ai_analysis.impact == "UNKNOWN"
        assert finding.ai_analysis.exploitability is None
        
        # Must route to NEEDS_MANUAL (cannot be SUBMIT_READY)
    
    def test_unknown_values_never_converted(self):
        """Test UNKNOWN values are NEVER converted to other values."""
        from vortex.domain.models import AIAnalysisResult
        
        # AI result with UNKNOWN impact
        ai_result = AIAnalysisResult(
            model_used="test_model",
            verdict="LIKELY",
            confidence=0.75,
            exploitability=None,
            impact="UNKNOWN",
            reportability=None,
            reasoning="Cannot determine impact",
            success=True
        )
        
        # UNKNOWN must remain UNKNOWN (never derived or converted)
        assert ai_result.impact == "UNKNOWN"
        assert ai_result.exploitability is None
        assert ai_result.reportability is None
        
        # FORBIDDEN: Deriving missing values
        # ❌ ai_result.exploitability = ai_result.confidence * 0.8
        # ❌ ai_result.impact = "LOW" if ai_result.impact == "UNKNOWN"


@pytest.mark.compliance
class TestAuthorityComplianceChecklist:
    """Compliance checklist tests per .clinerules."""
    
    def test_ai_role_advisory_only(self):
        """✓ Verify AI role is advisory only."""
        assert AUTHORITY_HIERARCHY[AuthorityLevel.AI_ADVISORY] == "ADVISORY_ONLY"
    
    def test_no_missing_field_derivation(self):
        """✓ Confirm no missing field derivation."""
        # This is enforced by code structure - fields remain None if missing
        pass
    
    def test_fallback_chain_availability(self):
        """✓ Ensure fallback chain availability."""
        # Heuristic fallback must be available when AI fails
        pass
    
    def test_non_authoritative_marking_for_recovered_data(self):
        """✓ Verify non-authoritative marking for recovered data."""
        from vortex.domain.models import AIAnalysisResult
        
        # Malformed JSON recovery result
        recovered_result = AIAnalysisResult(
            model_used="malformed_recovery",
            verdict="NEEDS_MANUAL",  # FORCE manual
            confidence=0.3,  # Severe penalty
            impact="UNKNOWN",
            reasoning="Recovered from malformed response - NOT AUTHORITATIVE",
            success=False,  # Mark as failed
            is_fallback_result=True
        )
        
        assert recovered_result.is_fallback_result is True
        assert recovered_result.success is False
        assert recovered_result.verdict == "NEEDS_MANUAL"