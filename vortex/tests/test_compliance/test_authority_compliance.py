"""
VORTEX Authority Compliance Tests - V17.0 ULTIMATE
Critical validation: Authority hierarchy NEVER violated
"""

import pytest
from datetime import datetime
from domain.enums import VerificationStatus, AuthorityLevel, ConfidenceSource
from domain.models import AssessmentResult, AIAnalysisResult, VerificationResult
from core.authority.hierarchy import AuthorityHierarchyEnforcer
from core.authority.validator import AuthorityValidator


@pytest.fixture
def authority_enforcer():
    """Create authority enforcer instance."""
    return AuthorityHierarchyEnforcer()


@pytest.fixture
def authority_validator():
    """Create authority validator instance."""
    return AuthorityValidator()


class TestAuthorityHierarchy:
    """Test immutable authority hierarchy enforcement."""
    
    def test_authority_levels_order(self, authority_enforcer):
        """CRITICAL: Verify authority levels are correctly ordered."""
        levels = authority_enforcer.authority_levels
        
        assert levels['SYSTEM_VERIFICATION'] == 1  # Highest
        assert levels['HUMAN_EXPERT'] == 2
        assert levels['AI_ADVISORY'] == 3
        assert levels['HEURISTIC'] == 4  # Lowest
        
        # Verify ordering
        assert levels['SYSTEM_VERIFICATION'] < levels['HUMAN_EXPERT']
        assert levels['HUMAN_EXPERT'] < levels['AI_ADVISORY']
        assert levels['AI_ADVISORY'] < levels['HEURISTIC']
    
    def test_ai_never_authoritative(self, authority_enforcer):
        """GOLDEN RULE: AI IS NEVER AUTHORITATIVE."""
        # Create finding with only AI confirmation (no system verification)
        finding = AssessmentResult(
            id="test-001",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.AI_CONFIRMED,
            heuristic_score=0.85,
            evidence="SQL error detected"
        )
        
        # Strong AI result
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.95,
            impact="HIGH",
            reportability=0.90,
            reasoning="Strong SQL injection evidence",
            success=True,
            authority_level=AuthorityLevel.AI_ADVISORY,
            is_authoritative=False
        )
        
        # NO system verification
        finding.verification_result = None
        
        # Should FAIL authority validation for SUBMIT_READY
        assert not authority_enforcer.validate_submit_ready_authority(finding)
        
        # Final determination should be NEEDS_MANUAL
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.NEEDS_MANUAL
    
    def test_system_verification_required(self, authority_enforcer):
        """REQUIREMENT 1: System verification is MANDATORY for SUBMIT_READY."""
        finding = AssessmentResult(
            id="test-002",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.AI_CONFIRMED,
            heuristic_score=0.85,
            evidence="SQL error"
        )
        
        # Strong AI + Heuristic, but NO system verification
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.95,
            impact="HIGH",
            reportability=0.92,
            reasoning="Clear evidence",
            success=True
        )
        
        # Test without system verification
        assert not authority_enforcer.validate_submit_ready_authority(finding)
        
        # Add successful system verification
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.88,
            match_type="exact_regex",
            matched_pattern="SQL syntax error",
            response_time=0.5,
            response_status=500
        )
        
        # Should pass now
        assert authority_enforcer.validate_submit_ready_authority(finding)
    
    def test_confidence_threshold_required(self, authority_enforcer):
        """REQUIREMENT 2: Confidence threshold ≥0.75 required."""
        finding = AssessmentResult(
            id="test-003",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.70,
            evidence="XSS pattern"
        )
        
        # Low confidence system verification
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.65,  # Below threshold
            match_type="fuzzy_match",
            response_time=0.3
        )
        
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.80,
            impact="MEDIUM",
            reportability=0.75,
            reasoning="XSS detected",
            success=True
        )
        
        # Should fail due to low system confidence
        assert not authority_enforcer.validate_submit_ready_authority(finding)
    
    def test_unknown_values_block_submit_ready(self, authority_enforcer):
        """REQUIREMENT 3: UNKNOWN values must route to NEEDS_MANUAL."""
        finding = AssessmentResult(
            id="test-004",
            url="https://target.com/test",
            vulnerability_type="lfi",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.82,
            evidence="File content exposed"
        )
        
        # Strong system verification
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.88,
            match_type="exact_regex",
            response_time=0.4
        )
        
        # AI result with UNKNOWN impact
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.85,
            impact="UNKNOWN",  # CRITICAL: UNKNOWN value
            exploitability=None,  # Missing field
            reportability=0.80,
            reasoning="LFI confirmed",
            success=True
        )
        
        # Should fail due to UNKNOWN values
        assert not authority_enforcer.validate_submit_ready_authority(finding)
        
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.NEEDS_MANUAL
    
    def test_deterministic_evidence_required(self, authority_enforcer):
        """REQUIREMENT 4: Deterministic evidence required for SUBMIT_READY."""
        finding = AssessmentResult(
            id="test-005",
            url="https://target.com/test",
            vulnerability_type="ssrf",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.78,
            evidence="Internal response"
        )
        
        # Non-deterministic verification
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.82,
            match_type="behavioral_heuristic",  # Not deterministic
            response_time=0.5
        )
        
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="LIKELY",
            confidence=0.75,
            impact="MEDIUM",
            reportability=0.70,
            reasoning="Possible SSRF",
            success=True
        )
        
        # Should fail without deterministic evidence
        assert not authority_enforcer.validate_submit_ready_authority(finding)


class TestAIAdvisoryLimitation:
    """Test AI advisory-only limitations."""
    
    def test_ai_field_derivation_prohibited(self):
        """CRITICAL: Missing AI fields must remain UNKNOWN, never derived."""
        # Simulate AI response with missing exploitability
        ai_response = {
            "verdict": "CONFIRMED",
            "confidence": 0.85,
            "impact": "HIGH",
            "reasoning": "Clear vulnerability"
            # exploitability intentionally missing
        }
        
        result = AIAnalysisResult(
            model_used="test_model",
            verdict=ai_response["verdict"],
            confidence=ai_response["confidence"],
            impact=ai_response["impact"],
            exploitability=ai_response.get("exploitability"),  # Should be None
            reportability=ai_response.get("reportability"),    # Should be None
            reasoning=ai_response["reasoning"],
            success=True
        )
        
        # VERIFY: Missing fields are None, not derived
        assert result.exploitability is None  # NOT derived from confidence
        assert result.reportability is None   # NOT derived from confidence
    
    def test_malformed_json_recovery_non_authoritative(self, authority_enforcer):
        """CRITICAL: Malformed JSON recovery is NON-AUTHORITATIVE."""
        finding = AssessmentResult(
            id="test-006",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.AI_CONFIRMED,
            heuristic_score=0.80,
            evidence="SQL error"
        )
        
        # Malformed JSON recovery result
        finding.ai_analysis = AIAnalysisResult(
            model_used="malformed_recovery",
            verdict="NEEDS_MANUAL",  # Must be NEEDS_MANUAL
            confidence=0.30,  # Severe penalty
            exploitability=None,
            impact="UNKNOWN",
            reportability=None,
            reasoning="Recovered from malformed response - NOT AUTHORITATIVE",
            success=False,
            is_fallback_result=True
        )
        
        # Even with strong system verification, recovered AI should not authorize
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.85,
            match_type="exact_regex"
        )
        
        # Should route to manual due to AI recovery
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.NEEDS_MANUAL
    
    def test_heuristic_poc_replay_prohibited(self):
        """CRITICAL: Heuristic PoCs must NEVER be replayed."""
        finding = AssessmentResult(
            id="test-007",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            status=VerificationStatus.DETECTED,
            heuristic_score=0.88,
            evidence="XSS pattern",
            confidence_source=ConfidenceSource.HEURISTIC_ONLY
        )
        
        # Heuristic-generated PoC
        finding.heuristic_poc = "<script>alert(1)</script>"
        
        # Should NOT be replayed
        from core.ai.advisory import ProductionAIIntegrationEngine
        ai_engine = ProductionAIIntegrationEngine()
        
        assert not ai_engine.should_replay_poc(finding)


class TestFastPathCompliance:
    """Test V11.1 FastPath promotion with authority compliance."""
    
    def test_fastpath_respects_authority_hierarchy(self, authority_enforcer):
        """FastPath MUST respect authority hierarchy."""
        finding = AssessmentResult(
            id="test-008",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.89,
            evidence="MySQL error"
        )
        
        # Strong system verification (FastPath eligible)
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.92,  # Very strong
            match_type="exact_regex",
            matched_pattern="MySQL syntax error"
        )
        
        # AI support (advisory)
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.88,
            impact="HIGH",
            reportability=0.90,
            reasoning="Clear SQL injection",
            success=True
        )
        
        # Should reach SUBMIT_READY via FastPath
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.SUBMIT_READY
        
        # BUT authority validation must still pass
        assert authority_enforcer.validate_submit_ready_authority(finding)
    
    def test_ai_failed_recovery_with_strong_system(self, authority_enforcer):
        """AI_FAILED doesn't block strong system evidence (V11.1)."""
        finding = AssessmentResult(
            id="test-009",
            url="https://target.com/test",
            vulnerability_type="xss_stored",
            status=VerificationStatus.AI_FAILED,
            heuristic_score=0.84,
            evidence="Stored XSS"
        )
        
        # AI failed
        finding.ai_analysis = AIAnalysisResult(
            model_used="ai_unavailable",
            verdict="NEEDS_MANUAL",
            confidence=0.0,
            impact="UNKNOWN",
            reasoning="AI models unavailable",
            success=False,
            is_fallback_result=True
        )
        
        # But strong system verification
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.91,
            match_type="structural_differential",
            matched_pattern="Script stored in database"
        )
        
        # Should still reach SUBMIT_READY
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.SUBMIT_READY


@pytest.mark.critical
class TestAuthorityViolationDetection:
    """Test authority violation detection and prevention."""
    
    def test_detect_authority_violation(self, authority_validator):
        """System must detect authority hierarchy violations."""
        # Simulate SUBMIT_READY without system verification (violation)
        finding = AssessmentResult(
            id="test-010",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.SUBMIT_READY,  # Invalid state
            heuristic_score=0.90,
            evidence="SQL error"
        )
        
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.95,
            impact="CRITICAL",
            reportability=0.95,
            reasoning="Critical SQLi",
            success=True
        )
        
        # NO system verification - VIOLATION
        finding.verification_result = None
        
        # Should detect violation
        violations = authority_validator.validate_finding_authority(finding)
        assert len(violations) > 0
        assert any("system verification" in v.lower() for v in violations)
    
    def test_prevent_unknown_value_progression(self, authority_enforcer):
        """UNKNOWN values must prevent SUBMIT_READY progression."""
        finding = AssessmentResult(
            id="test-011",
            url="https://target.com/test",
            vulnerability_type="lfi",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.85,
            evidence="File accessed"
        )
        
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.87,
            match_type="exact_regex"
        )
        
        # All possible UNKNOWN values
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.82,
            impact="UNKNOWN",        # UNKNOWN
            exploitability=None,     # Missing
            reportability=None,      # Missing
            reasoning="LFI detected",
            success=True
        )
        
        # Must route to NEEDS_MANUAL
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.NEEDS_MANUAL


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])