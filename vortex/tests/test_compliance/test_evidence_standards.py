"""
VORTEX Evidence Standards Compliance Tests - V17.0 ULTIMATE
Critical validation: Evidence standards never lowered
"""

import pytest
from domain.enums import VerificationStatus, FindingType
from domain.models import AssessmentResult, VerificationResult, AIAnalysisResult
from core.evidence.standards import EvidenceStandardsValidator
from core.evidence.behavioral import BehavioralEvidenceAnalyzer
from core.evidence.determinism import EvidenceDeterminismScorer


@pytest.fixture
def evidence_validator():
    """Create evidence standards validator."""
    return EvidenceStandardsValidator()


@pytest.fixture
def behavioral_analyzer():
    """Create behavioral evidence analyzer."""
    return BehavioralEvidenceAnalyzer()


@pytest.fixture
def determinism_scorer():
    """Create evidence determinism scorer."""
    return EvidenceDeterminismScorer()


class TestEvidenceHierarchy:
    """Test evidence hierarchy levels."""
    
    def test_evidence_levels_defined(self, evidence_validator):
        """Verify all evidence levels are properly defined."""
        levels = evidence_validator.evidence_levels
        
        assert 'DETERMINISTIC' in levels
        assert 'BEHAVIORAL' in levels
        assert 'PATTERN' in levels
        
        # Check minimum scores
        assert levels['DETERMINISTIC']['min_score'] >= 0.8
        assert levels['BEHAVIORAL']['min_score'] >= 0.6
        assert levels['PATTERN']['min_score'] >= 0.4
    
    def test_deterministic_required_for_submit_ready(self, evidence_validator):
        """CRITICAL: Deterministic evidence required for SUBMIT_READY."""
        finding = AssessmentResult(
            id="test-ev-001",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.85,
            evidence="SQL error detected"
        )
        
        # Pattern-only evidence (low determinism)
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.65,
            match_type="fuzzy_match",
            response_time=0.4
        )
        
        # Should fail validation for SUBMIT_READY
        determinism_score = evidence_validator.assess_evidence_determinism(finding)
        assert determinism_score < 0.8  # Below DETERMINISTIC threshold
        
        is_valid = evidence_validator.validate_evidence_for_status(
            finding, 
            VerificationStatus.SUBMIT_READY
        )
        assert not is_valid
    
    def test_behavioral_sufficient_for_system_verified(self, evidence_validator):
        """Behavioral evidence sufficient for SYSTEM_VERIFIED."""
        finding = AssessmentResult(
            id="test-ev-002",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            status=VerificationStatus.AI_CONFIRMED,
            heuristic_score=0.78,
            evidence="XSS pattern"
        )
        
        # Behavioral evidence
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.72,
            match_type="structural_differential",
            response_time=0.5
        )
        
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="LIKELY",
            confidence=0.75,
            impact="MEDIUM",
            reasoning="XSS likely",
            success=True
        )
        
        # Should pass for SYSTEM_VERIFIED
        determinism_score = evidence_validator.assess_evidence_determinism(finding)
        assert determinism_score >= 0.6  # Above BEHAVIORAL threshold
        
        is_valid = evidence_validator.validate_evidence_for_status(
            finding,
            VerificationStatus.SYSTEM_VERIFIED
        )
        assert is_valid


class TestBehavioralEvidenceUncertainty:
    """Test behavioral evidence with uncertainty acknowledgment."""
    
    def test_behavioral_differences_indicative_not_conclusive(self, behavioral_analyzer):
        """CRITICAL: Behavioral differences are INDICATIVE, not CONCLUSIVE."""
        original_response = {
            'status_code': 200,
            'body': 'Original content here',
            'response_time': 0.5
        }
        
        replay_response = {
            'status_code': 500,  # Status changed
            'body': 'Error: database error',  # Content changed
            'response_time': 2.5  # Time increased significantly
        }
        
        payload = "' OR 1=1--"
        
        analysis = behavioral_analyzer.assess_behavioral_evidence_with_uncertainty(
            original_response,
            replay_response,
            payload
        )
        
        # Should have indicators
        assert len(analysis['indicators']) > 0
        
        # Should have uncertainty factors
        assert len(analysis['uncertainty_factors']) > 0
        
        # Causation must be UNKNOWN
        assert analysis['causation_determination'] == "UNKNOWN - requires human expert analysis"
        
        # Max status is SYSTEM_VERIFIED (never auto-SUBMIT_READY)
        assert analysis['max_automated_status'] == VerificationStatus.SYSTEM_VERIFIED
    
    def test_uncertainty_penalty_applied(self, behavioral_analyzer):
        """Uncertainty factors reduce confidence."""
        original = {
            'status_code': 200,
            'body': 'Content A',
            'response_time': 0.5
        }
        
        replay = {
            'status_code': 500,
            'body': 'Content B',
            'response_time': 3.0
        }
        
        analysis = behavioral_analyzer.assess_behavioral_evidence_with_uncertainty(
            original, replay, "payload"
        )
        
        # Base confidence from indicators
        base_indicators = len(analysis['indicators'])
        base_confidence = min(base_indicators * 0.3, 0.9)
        
        # Should have penalty applied
        uncertainty_penalty = len(analysis['uncertainty_factors']) * 0.1
        expected_confidence = max(0.0, base_confidence - uncertainty_penalty)
        
        assert abs(analysis['confidence'] - expected_confidence) < 0.01
    
    def test_payload_reflection_increases_determinism(self, behavioral_analyzer):
        """Payload reflection is more deterministic evidence."""
        payload = "<script>alert(1)</script>"
        
        original = {
            'status_code': 200,
            'body': 'Search results for: query',
            'response_time': 0.3
        }
        
        replay = {
            'status_code': 200,
            'body': f'Search results for: {payload}',  # Payload reflected
            'response_time': 0.35
        }
        
        analysis = behavioral_analyzer.assess_behavioral_evidence_with_uncertainty(
            original, replay, payload
        )
        
        # Should detect payload reflection
        assert analysis['payload_reflected'] is True
        
        # Should increase confidence
        assert 'Payload reflection detected' in analysis['indicators']


class TestVulnerabilitySpecificEvidence:
    """Test vulnerability-specific evidence criteria."""
    
    def test_sqli_deterministic_indicators(self, evidence_validator):
        """SQL injection with database errors = high determinism."""
        finding = AssessmentResult(
            id="test-ev-003",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.82,
            evidence="MySQL error: You have an error in your SQL syntax near 'SELECT' at line 1. Check MySQL manual for version."
        )
        
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.85,
            match_type="exact_regex",
            matched_pattern="MySQL error: SQL syntax"
        )
        
        # Should get vulnerability-specific bonus
        vuln_bonus = evidence_validator._assess_vuln_specific_evidence(finding)
        assert vuln_bonus > 0
        
        # Total determinism should be high
        total_determinism = evidence_validator.assess_evidence_determinism(finding)
        assert total_determinism >= 0.8
    
    def test_xss_javascript_execution_deterministic(self, evidence_validator):
        """XSS with JS execution = highest determinism."""
        finding = AssessmentResult(
            id="test-ev-004",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.88,
            evidence="JavaScript alert fired: XSS confirmed"
        )
        
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.90,
            match_type="exact_regex",
            matched_pattern="<script>alert('XSS')</script>"
        )
        
        # XSS JS execution has highest bonus (0.20)
        vuln_bonus = evidence_validator._assess_vuln_specific_evidence(finding)
        assert vuln_bonus == 0.20
    
    def test_lfi_lower_determinism(self, evidence_validator):
        """LFI has lower confidence bonus due to ambiguity."""
        finding = AssessmentResult(
            id="test-ev-005",
            url="https://target.com/test",
            vulnerability_type="lfi",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.75,
            evidence="File content exposed: etc/passwd root:x:0:0:root:/root:/bin/bash"
        )
        
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.78,
            match_type="fuzzy_match",
            matched_pattern="etc/passwd"
        )
        
        # LFI has lower bonus (0.05)
        vuln_bonus = evidence_validator._assess_vuln_specific_evidence(finding)
        assert vuln_bonus <= 0.05
    
    def test_minimum_evidence_length_enforced(self, evidence_validator):
        """Minimum evidence length must be met."""
        finding = AssessmentResult(
            id="test-ev-006",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.DETECTED,
            heuristic_score=0.80,
            evidence="SQL error"  # Too short (< 50 chars for SQLi)
        )
        
        # Should not get bonus due to short evidence
        vuln_bonus = evidence_validator._assess_vuln_specific_evidence(finding)
        assert vuln_bonus == 0.0


class TestTextMatchingLimitations:
    """Test text matching limitations."""
    
    def test_text_match_alone_not_proof(self, evidence_validator):
        """TEXT MATCHING ALONE DOES NOT PROVE VULNERABILITY."""
        finding = AssessmentResult(
            id="test-ev-007",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.DETECTED,
            heuristic_score=0.70,
            evidence="error in response"  # Generic text match
        )
        
        # Only text pattern match (no system verification)
        finding.verification_result = None
        
        # Should have very low determinism
        determinism = evidence_validator.assess_evidence_determinism(finding)
        assert determinism < 0.5  # Pattern evidence only
        
        # Should not be valid for SUBMIT_READY
        is_valid = evidence_validator.validate_evidence_for_status(
            finding,
            VerificationStatus.SUBMIT_READY
        )
        assert not is_valid
    
    def test_exact_regex_match_increases_confidence(self, evidence_validator):
        """Exact regex matches are more deterministic than fuzzy."""
        finding_exact = AssessmentResult(
            id="test-ev-008a",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            heuristic_score=0.85,
            evidence="MySQL syntax error"
        )
        
        finding_exact.verification_result = VerificationResult(
            success=True,
            confidence=0.88,
            match_type="exact_regex",  # Exact match
            matched_pattern="MySQL.*syntax.*error"
        )
        
        finding_fuzzy = AssessmentResult(
            id="test-ev-008b",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            heuristic_score=0.85,
            evidence="MySQL syntax error"
        )
        
        finding_fuzzy.verification_result = VerificationResult(
            success=True,
            confidence=0.88,
            match_type="fuzzy_match",  # Fuzzy match
            matched_pattern="syntax"
        )
        
        # Exact match should have higher determinism
        exact_score = evidence_validator.assess_evidence_determinism(finding_exact)
        fuzzy_score = evidence_validator.assess_evidence_determinism(finding_fuzzy)
        
        assert exact_score > fuzzy_score


class TestEvidenceDeterminismScoring:
    """Test evidence determinism scoring algorithm."""
    
    def test_scoring_components(self, determinism_scorer):
        """Verify all scoring components are considered."""
        finding = AssessmentResult(
            id="test-ev-009",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.88,
            evidence="MySQL error: syntax error at line 1"
        )
        
        # Strong system verification (0.5)
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.90,
            match_type="exact_regex"
        )
        
        # AI confirmation (0.3)
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.88,
            impact="HIGH",
            reasoning="Clear SQL injection",
            success=True,
            is_fallback_result=False
        )
        
        # High heuristic (0.2)
        finding.heuristic_score = 0.85
        
        score = determinism_scorer.calculate_determinism_score(finding)
        
        # Should sum to ~1.0 (capped at 1.0)
        assert 0.9 <= score <= 1.0
    
    def test_fallback_ai_reduced_weight(self, determinism_scorer):
        """Fallback AI results have reduced weight."""
        finding = AssessmentResult(
            id="test-ev-010",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            heuristic_score=0.80,
            evidence="XSS pattern"
        )
        
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.82,
            match_type="structural_differential"
        )
        
        # Fallback AI result
        finding.ai_analysis = AIAnalysisResult(
            model_used="heuristic_fallback",
            verdict="LIKELY",
            confidence=0.70,
            impact="MEDIUM",
            reasoning="Heuristic-only fallback",
            success=True,
            is_fallback_result=True  # Fallback
        )
        
        score = determinism_scorer.calculate_determinism_score(finding)
        
        # Should not include AI component due to fallback
        # Max score: 0.4 (system) + 0.2 (heuristic) = 0.6
        assert score <= 0.7


@pytest.mark.critical
class TestEvidenceStandardsEnforcement:
    """Critical tests for evidence standards enforcement."""
    
    def test_submit_ready_requires_deterministic(self, evidence_validator):
        """SUBMIT_READY absolutely requires deterministic evidence."""
        # Create 10 findings with varying evidence quality
        findings = []
        for i in range(10):
            finding = AssessmentResult(
                id=f"test-ev-enforce-{i}",
                url=f"https://target.com/test{i}",
                vulnerability_type="sql_injection",
                status=VerificationStatus.SYSTEM_VERIFIED,
                heuristic_score=0.75 + (i * 0.01),
                evidence=f"Evidence {i}"
            )
            
            # Vary evidence determinism
            match_types = ["fuzzy_match", "fuzzy_match", "structural_differential", 
                          "exact_regex", "exact_regex", "exact_regex",
                          "database_error_confirmed", "exact_regex", 
                          "structural_differential", "exact_regex"]
            
            finding.verification_result = VerificationResult(
                success=True,
                confidence=0.75 + (i * 0.02),
                match_type=match_types[i]
            )
            
            findings.append(finding)
        
        # Count valid for SUBMIT_READY
        valid_count = sum(
            1 for f in findings 
            if evidence_validator.validate_evidence_for_status(
                f, VerificationStatus.SUBMIT_READY
            )
        )
        
        # Only high-determinism findings should pass
        assert valid_count < len(findings)  # Not all pass
        assert valid_count >= 3  # But some should pass
    
    def test_no_evidence_quality_degradation(self, evidence_validator):
        """Evidence standards must NEVER be lowered."""
        # Simulate pressure to lower standards
        original_threshold = evidence_validator.evidence_levels['DETERMINISTIC']['min_score']
        
        # Threshold should never be < 0.7
        assert original_threshold >= 0.7
        
        # Even under "optimization" pressure
        finding = AssessmentResult(
            id="test-ev-pressure",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            heuristic_score=0.85,
            evidence="SQL error"
        )
        
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.68,  # Below threshold
            match_type="fuzzy_match"
        )
        
        # Must still fail validation
        is_valid = evidence_validator.validate_evidence_for_status(
            finding,
            VerificationStatus.SUBMIT_READY
        )
        assert not is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])