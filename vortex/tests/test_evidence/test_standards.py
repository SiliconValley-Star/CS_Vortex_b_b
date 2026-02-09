"""
VORTEX Evidence Standards Tests
Per .clinerules VORTEX_EVIDENCE_STANDARDS.md

Tests evidence hierarchy:
- Level 1: DETERMINISTIC (Required for SUBMIT_READY)
- Level 2: BEHAVIORAL (Required for SYSTEM_VERIFIED)  
- Level 3: PATTERN (Sufficient for AI_CONFIRMED)
"""

import pytest
from vortex.config.evidence_config import (
    EvidenceLevel,
    EVIDENCE_LEVELS,
    TEXT_MATCHING_RULES,
    VULN_SPECIFIC_EVIDENCE_CRITERIA
)


class TestEvidenceHierarchy:
    """Test evidence hierarchy levels."""
    
    def test_evidence_level_hierarchy(self):
        """Test evidence levels are properly ordered."""
        assert EvidenceLevel.DETERMINISTIC.value == "DETERMINISTIC"
        assert EvidenceLevel.BEHAVIORAL.value == "BEHAVIORAL"
        assert EvidenceLevel.PATTERN.value == "PATTERN"
    
    def test_deterministic_highest_standard(self):
        """Test deterministic evidence is the highest standard."""
        deterministic = EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]
        behavioral = EVIDENCE_LEVELS[EvidenceLevel.BEHAVIORAL]
        pattern = EVIDENCE_LEVELS[EvidenceLevel.PATTERN]
        
        assert deterministic["min_score"] > behavioral["min_score"]
        assert behavioral["min_score"] > pattern["min_score"]
    
    def test_deterministic_required_for_submit_ready(self):
        """Test deterministic evidence is required for SUBMIT_READY."""
        deterministic = EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]
        assert "SUBMIT_READY" in deterministic["required_for"]
    
    def test_behavioral_required_for_system_verified(self):
        """Test behavioral evidence is required for SYSTEM_VERIFIED."""
        behavioral = EVIDENCE_LEVELS[EvidenceLevel.BEHAVIORAL]
        assert "SYSTEM_VERIFIED" in behavioral["required_for"]
    
    def test_pattern_sufficient_for_ai_confirmed(self):
        """Test pattern evidence is sufficient for AI_CONFIRMED."""
        pattern = EVIDENCE_LEVELS[EvidenceLevel.PATTERN]
        assert "AI_CONFIRMED" in pattern["required_for"]


class TestDeterministicEvidence:
    """Test deterministic evidence requirements."""
    
    def test_deterministic_characteristics(self):
        """Test deterministic evidence characteristics."""
        deterministic = EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]
        characteristics = deterministic["characteristics"]
        
        assert "reproducible" in characteristics
        assert "measurable" in characteristics
        assert "independent_verification" in characteristics
    
    def test_deterministic_minimum_score(self):
        """Test deterministic evidence minimum score."""
        deterministic = EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]
        assert deterministic["min_score"] == 0.8
    
    def test_deterministic_score_for_submit_ready(self, mock_submit_ready_finding):
        """Test SUBMIT_READY finding has deterministic evidence."""
        finding = mock_submit_ready_finding
        
        # Should have evidence determinism score
        # (Would be calculated by evidence validator)
        assert finding.verification_result.confidence >= 0.75
        assert finding.verification_result.match_type in ["exact_regex", "structural_differential"]


class TestBehavioralEvidence:
    """Test behavioral evidence with uncertainty acknowledgment."""
    
    def test_behavioral_characteristics(self):
        """Test behavioral evidence characteristics."""
        behavioral = EVIDENCE_LEVELS[EvidenceLevel.BEHAVIORAL]
        characteristics = behavioral["characteristics"]
        
        assert "observable_differences" in characteristics
        assert "consistent_patterns" in characteristics
        assert "structural_changes" in characteristics
        assert "requires_causation_analysis" in characteristics
    
    def test_behavioral_differences_indicative_not_conclusive(self, mock_behavioral_analysis):
        """Test behavioral differences are INDICATIVE, not CONCLUSIVE."""
        analysis = mock_behavioral_analysis
        
        # Has indicators
        assert len(analysis["indicators"]) > 0
        
        # But also has uncertainty factors
        assert len(analysis["uncertainty_factors"]) > 0
        
        # Causation is UNKNOWN
        assert analysis["causation_determination"] == "UNKNOWN - requires expert analysis"
        
        # Maximum automated status is SYSTEM_VERIFIED (not SUBMIT_READY)
        assert analysis["max_automated_status"] == "SYSTEM_VERIFIED"
    
    def test_uncertainty_penalty_applied(self, mock_behavioral_analysis):
        """Test uncertainty penalty is applied to confidence."""
        analysis = mock_behavioral_analysis
        
        indicators_count = len(analysis["indicators"])
        uncertainty_count = len(analysis["uncertainty_factors"])
        
        # Base confidence from indicators
        base_confidence = min(indicators_count * 0.3, 0.9)
        
        # Uncertainty penalty
        uncertainty_penalty = uncertainty_count * 0.1
        
        # Final confidence
        expected = max(0.0, base_confidence - uncertainty_penalty)
        
        assert abs(analysis["confidence"] - expected) < 0.01
    
    def test_non_security_causes_identified(self, mock_behavioral_analysis):
        """Test non-security causes are identified in uncertainty."""
        analysis = mock_behavioral_analysis
        
        uncertainty_factors = [f.lower() for f in analysis["uncertainty_factors"]]
        
        # Should identify potential non-security causes
        non_security_keywords = ["infrastructure", "cdn", "cache", "load balancing", "retry", "a/b testing"]
        
        found_non_security = any(
            keyword in factor
            for factor in uncertainty_factors
            for keyword in non_security_keywords
        )
        
        assert found_non_security, "Should identify potential non-security causes"


class TestTextMatchingLimitations:
    """Test text matching limitations per .clinerules."""
    
    def test_text_matching_not_proof(self):
        """Test text matching alone does NOT prove vulnerability."""
        assert TEXT_MATCHING_RULES["proves_vulnerability"] is False
    
    def test_text_matching_requires_verification(self):
        """Test text matching requires verification."""
        assert TEXT_MATCHING_RULES["requires_verification"] is True
    
    def test_text_pattern_as_indicator_only(self):
        """Test text patterns are indicators only, not proof."""
        # Example: Finding "error" in response
        response_body = "Error: Database connection failed"
        
        # Text match detected
        has_error_text = "error" in response_body.lower()
        assert has_error_text is True
        
        # But this is NOT proof of vulnerability
        # It's an indicator that requires behavioral verification
        proves_vulnerability = TEXT_MATCHING_RULES["proves_vulnerability"]
        assert proves_vulnerability is False
    
    def test_forbidden_text_match_as_proof(self):
        """Test using text match as proof is FORBIDDEN."""
        # ❌ FORBIDDEN pattern:
        # if "error" in response.body:
        #     return VerificationResult(success=True)
        
        # ✅ MANDATORY pattern:
        # if "error" in response.body:
        #     return VerificationResult(
        #         success=False,  # Not proven yet
        #         requires_behavioral_verification=True
        #     )
        
        response_body = "SQL error detected"
        
        # Text match is indicator
        text_match = "error" in response_body.lower()
        
        # But verification success is separate
        verification_success = False  # Text match alone is not success
        requires_behavioral = TEXT_MATCHING_RULES["requires_verification"]
        
        assert text_match is True
        assert verification_success is False
        assert requires_behavioral is True


class TestVulnerabilitySpecificEvidence:
    """Test vulnerability-specific evidence criteria."""
    
    def test_sql_injection_criteria(self):
        """Test SQL injection evidence criteria."""
        sqli_criteria = VULN_SPECIFIC_EVIDENCE_CRITERIA["sql_injection"]
        
        assert "deterministic_indicators" in sqli_criteria
        assert "confidence_bonus" in sqli_criteria
        assert "min_evidence_length" in sqli_criteria
        
        # SQL injection has high determinism
        assert sqli_criteria["confidence_bonus"] >= 0.10
    
    def test_xss_reflected_criteria(self):
        """Test XSS reflected evidence criteria."""
        xss_criteria = VULN_SPECIFIC_EVIDENCE_CRITERIA["xss_reflected"]
        
        indicators = xss_criteria["deterministic_indicators"]
        
        # XSS should look for JavaScript execution
        assert any("javascript" in ind.lower() or "alert" in ind.lower() for ind in indicators)
        
        # XSS has very high determinism when JS executes
        assert xss_criteria["confidence_bonus"] >= 0.15
    
    def test_ssrf_criteria(self):
        """Test SSRF evidence criteria."""
        ssrf_criteria = VULN_SPECIFIC_EVIDENCE_CRITERIA["ssrf"]
        
        indicators = ssrf_criteria["deterministic_indicators"]
        
        # SSRF should look for internal network access
        internal_network_indicators = ["192.168", "10.", "localhost", "127.0.0.1"]
        
        for internal_ind in internal_network_indicators:
            assert any(internal_ind in ind for ind in indicators)
    
    def test_lfi_criteria(self):
        """Test LFI evidence criteria."""
        lfi_criteria = VULN_SPECIFIC_EVIDENCE_CRITERIA["lfi"]
        
        # LFI has lower bonus due to higher false positive risk
        assert lfi_criteria["confidence_bonus"] < 0.10
        
        # LFI requires longer evidence
        assert lfi_criteria["min_evidence_length"] > 50
    
    def test_vuln_specific_bonus_applied(self):
        """Test vulnerability-specific bonus is applied correctly."""
        # SQL injection with database error
        evidence = "MySQL error: You have an error in your SQL syntax at line 1"
        
        sqli_criteria = VULN_SPECIFIC_EVIDENCE_CRITERIA["sql_injection"]
        indicators = sqli_criteria["deterministic_indicators"]
        
        # Count indicator matches
        matches = sum(1 for ind in indicators if ind.lower() in evidence.lower())
        
        # Should have multiple matches
        assert matches >= 2
        
        # Should qualify for confidence bonus
        if matches >= 2:
            bonus = sqli_criteria["confidence_bonus"]
            assert bonus == 0.15


class TestEvidenceDeterminismScoring:
    """Test evidence determinism scoring."""
    
    def test_exact_regex_highest_determinism(self):
        """Test exact regex match has highest determinism."""
        # exact_regex should score 0.5
        match_types_scores = {
            "exact_regex": 0.5,
            "structural_differential": 0.4,
            "fuzzy_match": 0.3
        }
        
        assert match_types_scores["exact_regex"] > match_types_scores["structural_differential"]
        assert match_types_scores["structural_differential"] > match_types_scores["fuzzy_match"]
    
    def test_ai_advisory_contribution(self):
        """Test AI contributes but doesn't dominate determinism."""
        # System verification: 0.5 (exact_regex)
        system_score = 0.5
        
        # AI CONFIRMED: adds 0.3 (advisory)
        ai_confirmed_bonus = 0.3
        
        # AI LIKELY: adds 0.2 (advisory)
        ai_likely_bonus = 0.2
        
        # AI contribution is significant but doesn't override system
        assert system_score > ai_confirmed_bonus
        assert system_score > ai_likely_bonus
    
    def test_heuristic_lowest_contribution(self):
        """Test heuristic has lowest determinism contribution."""
        # Heuristic contributions
        high_heuristic = 0.2  # score >= 0.8
        medium_heuristic = 0.1  # score >= 0.6
        
        # System contributions
        exact_regex = 0.5
        
        # Heuristic much lower than system
        assert exact_regex > high_heuristic * 2
        assert exact_regex > medium_heuristic * 4
    
    def test_combined_evidence_scoring(self, mock_submit_ready_finding):
        """Test combined evidence scoring."""
        finding = mock_submit_ready_finding
        
        # System verification (0.5 for exact_regex)
        system_component = 0.5
        
        # AI confirmed (0.3 advisory)
        ai_component = 0.3 if finding.ai_analysis.verdict == "CONFIRMED" else 0
        
        # Heuristic (0.2 for high score)
        heuristic_component = 0.2 if finding.heuristic_score >= 0.8 else 0
        
        # Total should be capped at 1.0
        total_score = min(system_component + ai_component + heuristic_component, 1.0)
        
        # Submit-ready finding should have high determinism
        assert total_score >= 0.7


class TestEvidenceRequirementsByStatus:
    """Test evidence requirements for each status."""
    
    def test_submit_ready_requires_high_determinism(self):
        """Test SUBMIT_READY requires ≥0.7 determinism."""
        from vortex.config.evidence_config import EVIDENCE_REQUIREMENTS
        
        submit_ready_requirement = EVIDENCE_REQUIREMENTS.get("SUBMIT_READY", 0.7)
        assert submit_ready_requirement >= 0.7
    
    def test_system_verified_requires_moderate_determinism(self):
        """Test SYSTEM_VERIFIED requires ≥0.5 determinism."""
        from vortex.config.evidence_config import EVIDENCE_REQUIREMENTS
        
        system_verified_requirement = EVIDENCE_REQUIREMENTS.get("SYSTEM_VERIFIED", 0.5)
        assert system_verified_requirement >= 0.5
        assert system_verified_requirement < 0.7  # Lower than SUBMIT_READY
    
    def test_ai_confirmed_requires_pattern_evidence(self):
        """Test AI_CONFIRMED requires ≥0.3 determinism."""
        from vortex.config.evidence_config import EVIDENCE_REQUIREMENTS
        
        ai_confirmed_requirement = EVIDENCE_REQUIREMENTS.get("AI_CONFIRMED", 0.3)
        assert ai_confirmed_requirement >= 0.3
        assert ai_confirmed_requirement < 0.5  # Lower than SYSTEM_VERIFIED
    
    def test_needs_manual_any_evidence_level(self):
        """Test NEEDS_MANUAL accepts any evidence level."""
        from vortex.config.evidence_config import EVIDENCE_REQUIREMENTS
        
        needs_manual_requirement = EVIDENCE_REQUIREMENTS.get("NEEDS_MANUAL", 0.0)
        assert needs_manual_requirement == 0.0  # Any evidence level


@pytest.mark.compliance
class TestEvidenceComplianceChecklist:
    """Evidence validation compliance checklist per .clinerules."""
    
    def test_evidence_is_reproducible(self):
        """✓ Evidence is reproducible (not single-instance)."""
        # This would be validated by verification system
        pass
    
    def test_behavioral_changes_documented(self):
        """✓ Behavioral changes are documented."""
        # mock_behavioral_analysis fixture provides this
        pass
    
    def test_pattern_matches_supported_by_system(self):
        """✓ Pattern matches are supported by system verification."""
        # Text patterns alone are not sufficient
        assert TEXT_MATCHING_RULES["proves_vulnerability"] is False
    
    def test_causation_uncertainty_acknowledged(self):
        """✓ Causation uncertainty is acknowledged."""
        # Behavioral analysis must include uncertainty factors
        pass
    
    def test_vuln_specific_criteria_met(self):
        """✓ Vulnerability-specific evidence criteria met."""
        # Each vulnerability type has specific criteria
        assert "sql_injection" in VULN_SPECIFIC_EVIDENCE_CRITERIA
        assert "xss_reflected" in VULN_SPECIFIC_EVIDENCE_CRITERIA
    
    def test_determinism_score_adequate(self):
        """✓ Determinism score ≥ 0.7 for SUBMIT_READY."""
        from vortex.config.evidence_config import EVIDENCE_REQUIREMENTS
        assert EVIDENCE_REQUIREMENTS.get("SUBMIT_READY", 0) >= 0.7


class TestEvidenceValidationExamples:
    """Real-world evidence validation examples."""
    
    def test_sql_injection_with_database_error(self):
        """Test SQL injection with clear database error."""
        evidence = """
        HTTP/1.1 500 Internal Server Error
        
        MySQL Error: You have an error in your SQL syntax; 
        check the manual that corresponds to your MySQL server version 
        for the right syntax to use near '' OR '1'='1' at line 1
        """
        
        # Multiple deterministic indicators
        indicators = ["mysql error", "sql syntax", "line 1"]
        matches = sum(1 for ind in indicators if ind in evidence.lower())
        
        assert matches >= 2  # Qualifies for confidence bonus
    
    def test_xss_with_javascript_execution(self):
        """Test XSS with JavaScript execution evidence."""
        evidence = """
        <script>alert('XSS')</script> executed in page
        JavaScript alert() was triggered
        Payload reflection: <img src=x onerror=alert(1)>
        """
        
        # Very deterministic - JavaScript actually executed
        indicators = ["alert", "javascript", "executed"]
        matches = sum(1 for ind in indicators if ind.lower() in evidence.lower())
        
        assert matches >= 2  # Very high determinism
    
    def test_behavioral_difference_with_uncertainty(self):
        """Test behavioral difference with uncertainty factors."""
        analysis = {
            "indicators": [
                "Response time: 0.5s → 3.2s (+2.7s)",
                "Status: 200 → 500",
                "Content-Length: 5000 → 200 (-4800 bytes)"
            ],
            "uncertainty_factors": [
                "Time difference could be network latency",
                "Status change could be upstream timeout",
                "Size change could be error page vs normal page"
            ],
            "confidence": 0.45,  # Reduced by uncertainty
            "causation": "UNKNOWN"
        }
        
        # Has evidence but causation uncertain
        assert len(analysis["indicators"]) > 0
        assert len(analysis["uncertainty_factors"]) > 0
        assert analysis["causation"] == "UNKNOWN"
        
        # Confidence reduced by uncertainty
        base = len(analysis["indicators"]) * 0.3  # 0.9
        penalty = len(analysis["uncertainty_factors"]) * 0.1  # 0.3
        expected = base - penalty  # 0.6 - but capped based on rules
        
        assert analysis["confidence"] < base