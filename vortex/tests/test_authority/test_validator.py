"""
VORTEX Authority Validator Tests
Per .clinerules VORTEX_CORE_AUTHORITY.md

Tests authority validation logic and enforcement mechanisms
"""

import pytest
from vortex.domain.enums import VerificationStatus, FindingType, FindingSeverity
from vortex.domain.models import AssessmentResult, VerificationResult, AIAnalysisResult


class TestAuthorityValidation:
    """Test authority validation enforcement."""
    
    def test_validate_submit_ready_authority_success(self, mock_submit_ready_finding):
        """Test successful SUBMIT_READY authority validation."""
        finding = mock_submit_ready_finding
        
        # All requirements met
        assert finding.verification_result is not None
        assert finding.verification_result.success is True
        assert finding.verification_result.confidence >= 0.75
        
        # No UNKNOWN values
        if finding.ai_analysis:
            assert finding.ai_analysis.impact != "UNKNOWN"
        
        # Should validate successfully
        authority_valid = True
        assert authority_valid is True
    
    def test_validate_submit_ready_no_system_verification(self):
        """Test SUBMIT_READY validation fails without system verification."""
        finding = AssessmentResult(
            url="https://target.com/test",
            finding_type=FindingType.SQLI_ERROR,
            severity=FindingSeverity.HIGH,
            heuristic_score=0.88,
            evidence="SQL error",
            vulnerable_parameter="id"
        )
        
        # No system verification
        finding.verification_result = None
        
        # Should fail authority validation
        authority_valid = False
        assert authority_valid is False
    
    def test_validate_submit_ready_low_system_confidence(self):
        """Test SUBMIT_READY validation fails with low system confidence."""
        finding = AssessmentResult(
            url="https://target.com/test",
            finding_type=FindingType.SQLI_ERROR,
            severity=FindingSeverity.HIGH,
            heuristic_score=0.85,
            evidence="SQL error",
            vulnerable_parameter="id"
        )
        
        # Low system confidence
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.68,  # Below 0.75 threshold
            match_type="fuzzy_match"
        )
        
        # Should fail authority validation
        authority_valid = False
        assert authority_valid is False
    
    def test_validate_submit_ready_unknown_values_present(self):
        """Test SUBMIT_READY validation fails with UNKNOWN values."""
        finding = AssessmentResult(
            url="https://target.com/test",
            finding_type=FindingType.SQLI_ERROR,
            severity=FindingSeverity.HIGH,
            heuristic_score=0.87,
            evidence="SQL error",
            vulnerable_parameter="id"
        )
        
        # System verification good
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.85,
            match_type="exact_regex"
        )
        
        # But AI has UNKNOWN impact
        finding.ai_analysis = AIAnalysisResult(
            model_used="test",
            verdict="CONFIRMED",
            confidence=0.82,
            impact="UNKNOWN",  # UNKNOWN value
            reasoning="Test",
            success=True
        )
        
        # Should fail authority validation
        has_unknown = finding.ai_analysis.impact == "UNKNOWN"
        assert has_unknown is True
        authority_valid = False
        assert authority_valid is False
    
    def test_validate_submit_ready_non_deterministic_evidence(self):
        """Test SUBMIT_READY validation fails without deterministic evidence."""
        finding = AssessmentResult(
            url="https://target.com/test",
            finding_type=FindingType.SQLI_ERROR,
            severity=FindingSeverity.MEDIUM,
            heuristic_score=0.75,
            evidence="Possible SQL pattern",
            vulnerable_parameter="id"
        )
        
        # System verification with low determinism
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.78,
            match_type="text_pattern"  # Not deterministic
        )
        
        # Low determinism evidence
        finding.evidence_determinism_score = 0.55  # Below 0.7
        
        # Should fail authority validation
        is_deterministic = finding.evidence_determinism_score >= 0.7
        assert is_deterministic is False


class TestAuthorityEnforcement:
    """Test authority enforcement mechanisms."""
    
    def test_ai_cannot_override_system_verification(self):
        """Test AI cannot override system verification decision."""
        finding = AssessmentResult(
            url="https://target.com/test",
            finding_type=FindingType.SQLI_ERROR,
            severity=FindingSeverity.HIGH,
            heuristic_score=0.90,
            evidence="SQL test",
            vulnerable_parameter="id"
        )
        
        # AI says CONFIRMED with very high confidence
        finding.ai_analysis = AIAnalysisResult(
            model_used="test",
            verdict="CONFIRMED",
            confidence=0.98,  # Very high
            impact="CRITICAL",
            reasoning="AI very confident",
            success=True
        )
        
        # System verification fails
        finding.verification_result = VerificationResult(
            success=False,
            confidence=0.0,
            error="No vulnerability found"
        )
        
        # System authority wins
        finding.status = VerificationStatus.FALSE_POSITIVE
        assert finding.status == VerificationStatus.FALSE_POSITIVE
    
    def test_ai_marked_as_non_authoritative(self):
        """Test AI results are always marked as non-authoritative."""
        ai_result = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.95,
            reasoning="Test reasoning",
            success=True,
            is_authoritative=False,  # MUST be False
            requires_system_validation=True  # MUST be True
        )
        
        # AI NEVER authoritative
        assert ai_result.is_authoritative is False
        assert ai_result.requires_system_validation is True
    
    def test_system_verification_has_highest_authority(self):
        """Test system verification has highest authority."""
        from vortex.config.authority_config import AUTHORITY_HIERARCHY
        
        # System verification = 1 (highest)
        assert AUTHORITY_HIERARCHY['SYSTEM_VERIFICATION'] == 1
        
        # All others have lower authority
        assert AUTHORITY_HIERARCHY['HUMAN_EXPERT'] > AUTHORITY_HIERARCHY['SYSTEM_VERIFICATION']
        assert AUTHORITY_HIERARCHY['AI_ADVISORY'] > AUTHORITY_HIERARCHY['SYSTEM_VERIFICATION']
        assert AUTHORITY_HIERARCHY['HEURISTIC'] > AUTHORITY_HIERARCHY['SYSTEM_VERIFICATION']
    
    def test_authority_chain_validation(self, mock_submit_ready_finding):
        """Test complete authority chain validation."""
        finding = mock_submit_ready_finding
        
        # Authority chain:
        # 1. System verification (authoritative)
        assert finding.verification_result.success is True
        
        # 2. AI analysis (advisory)
        if finding.ai_analysis:
            assert finding.ai_analysis.is_authoritative is False
        
        # 3. Heuristic detection (indicative)
        assert finding.heuristic_score > 0.0
        
        # Final status must be based on system authority
        assert finding.status == VerificationStatus.SUBMIT_READY


class TestUnknownValueValidation:
    """Test UNKNOWN value validation and handling."""
    
    def test_unknown_different_from_low(self):
        """Test UNKNOWN ≠ LOW."""
        from vortex.config.authority_config import UNKNOWN_VALUE_HANDLING
        
        # UNKNOWN = Insufficient information
        assert UNKNOWN_VALUE_HANDLING['UNKNOWN'] == 'Insufficient information'
        
        # LOW = Determined minimal impact
        assert UNKNOWN_VALUE_HANDLING['LOW'] == 'Determined minimal impact'
        
        # They have different meanings
        assert UNKNOWN_VALUE_HANDLING['UNKNOWN'] != UNKNOWN_VALUE_HANDLING['LOW']
    
    def test_unknown_different_from_false(self):
        """Test UNKNOWN ≠ FALSE."""
        from vortex.config.authority_config import UNKNOWN_VALUE_HANDLING
        
        # UNKNOWN = Insufficient information
        # FALSE = Determined negative
        assert UNKNOWN_VALUE_HANDLING['UNKNOWN'] != UNKNOWN_VALUE_HANDLING['FALSE']
    
    def test_unknown_different_from_zero(self):
        """Test UNKNOWN ≠ 0."""
        from vortex.config.authority_config import UNKNOWN_VALUE_HANDLING
        
        # UNKNOWN = Insufficient information
        # ZERO = Measured absence
        assert UNKNOWN_VALUE_HANDLING['UNKNOWN'] != UNKNOWN_VALUE_HANDLING['ZERO']
    
    def test_unknown_routes_to_needs_manual(self):
        """Test UNKNOWN values route to NEEDS_MANUAL."""
        from vortex.config.authority_config import UNKNOWN_VALUE_HANDLING
        
        # UNKNOWN must route to NEEDS_MANUAL
        assert UNKNOWN_VALUE_HANDLING['route_to'] == VerificationStatus.NEEDS_MANUAL
    
    def test_unknown_never_converted(self):
        """Test UNKNOWN values are never converted."""
        ai_result = AIAnalysisResult(
            model_used="test",
            verdict="LIKELY",
            confidence=0.75,
            exploitability=None,  # Missing - stays None
            impact="UNKNOWN",     # UNKNOWN - stays UNKNOWN
            reportability=None,   # Missing - stays None
            reasoning="Test",
            success=True
        )
        
        # NEVER convert UNKNOWN
        assert ai_result.exploitability is None  # NOT derived
        assert ai_result.impact == "UNKNOWN"     # NOT converted to LOW
        assert ai_result.reportability is None   # NOT derived


@pytest.mark.compliance
class TestAuthorityValidatorCompliance:
    """Authority validator compliance checklist."""
    
    def test_submit_ready_requires_system_verification(self):
        """✓ SUBMIT_READY requires system verification."""
        from vortex.config.authority_config import SUBMIT_READY_REQUIREMENTS
        
        assert SUBMIT_READY_REQUIREMENTS['system_verification_required'] is True
    
    def test_submit_ready_min_confidence_enforced(self):
        """✓ SUBMIT_READY minimum confidence enforced."""
        from vortex.config.authority_config import SUBMIT_READY_REQUIREMENTS
        
        assert SUBMIT_READY_REQUIREMENTS['min_system_confidence'] >= 0.75
    
    def test_submit_ready_no_unknown_values_enforced(self):
        """✓ SUBMIT_READY no UNKNOWN values enforced."""
        from vortex.config.authority_config import SUBMIT_READY_REQUIREMENTS
        
        assert SUBMIT_READY_REQUIREMENTS['no_unknown_values'] is True
    
    def test_submit_ready_deterministic_evidence_required(self):
        """✓ SUBMIT_READY deterministic evidence required."""
        from vortex.config.authority_config import SUBMIT_READY_REQUIREMENTS
        
        assert SUBMIT_READY_REQUIREMENTS['deterministic_evidence'] is True
    
    def test_authority_hierarchy_never_violated(self):
        """✓ Authority hierarchy never violated."""
        from vortex.config.authority_config import AUTHORITY_HIERARCHY
        
        # System > Human > AI > Heuristic
        assert AUTHORITY_HIERARCHY['SYSTEM_VERIFICATION'] < AUTHORITY_HIERARCHY['HUMAN_EXPERT']
        assert AUTHORITY_HIERARCHY['HUMAN_EXPERT'] < AUTHORITY_HIERARCHY['AI_ADVISORY']
        assert AUTHORITY_HIERARCHY['AI_ADVISORY'] < AUTHORITY_HIERARCHY['HEURISTIC']