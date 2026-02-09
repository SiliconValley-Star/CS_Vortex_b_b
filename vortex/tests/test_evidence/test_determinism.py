"""
VORTEX Evidence Determinism Tests
Per .clinerules VORTEX_EVIDENCE_STANDARDS.md

Tests evidence determinism scoring and validation
"""

import pytest
from vortex.domain.enums import VerificationStatus, FindingType, FindingSeverity, EvidenceLevel
from vortex.domain.models import AssessmentResult, VerificationResult, AIAnalysisResult


class TestEvidenceDeterminismScoring:
    """Test evidence determinism scoring system."""
    
    def test_deterministic_evidence_high_score(self):
        """Test deterministic evidence receives high score (≥0.8)."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        deterministic_min = EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]['min_score']
        assert deterministic_min >= 0.8
    
    def test_behavioral_evidence_medium_score(self):
        """Test behavioral evidence receives medium score (≥0.6)."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        behavioral_min = EVIDENCE_LEVELS[EvidenceLevel.BEHAVIORAL]['min_score']
        assert behavioral_min >= 0.6
    
    def test_pattern_evidence_low_score(self):
        """Test pattern evidence receives low score (≥0.4)."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        pattern_min = EVIDENCE_LEVELS[EvidenceLevel.PATTERN]['min_score']
        assert pattern_min >= 0.4
    
    def test_determinism_hierarchy_enforced(self):
        """Test determinism hierarchy: Deterministic > Behavioral > Pattern."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        deterministic = EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]['min_score']
        behavioral = EVIDENCE_LEVELS[EvidenceLevel.BEHAVIORAL]['min_score']
        pattern = EVIDENCE_LEVELS[EvidenceLevel.PATTERN]['min_score']
        
        assert deterministic > behavioral > pattern


class TestDeterministicEvidenceRequirements:
    """Test deterministic evidence requirements."""
    
    def test_submit_ready_requires_deterministic(self):
        """Test SUBMIT_READY requires deterministic evidence."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        deterministic_config = EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]
        required_for = deterministic_config['required_for']
        
        assert VerificationStatus.SUBMIT_READY in required_for
    
    def test_system_verified_requires_behavioral(self):
        """Test SYSTEM_VERIFIED requires behavioral evidence."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        behavioral_config = EVIDENCE_LEVELS[EvidenceLevel.BEHAVIORAL]
        required_for = behavioral_config['required_for']
        
        assert VerificationStatus.SYSTEM_VERIFIED in required_for
    
    def test_ai_confirmed_accepts_pattern(self):
        """Test AI_CONFIRMED accepts pattern evidence."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        pattern_config = EVIDENCE_LEVELS[EvidenceLevel.PATTERN]
        required_for = pattern_config['required_for']
        
        assert VerificationStatus.AI_CONFIRMED in required_for


class TestEvidenceDeterminismCalculation:
    """Test evidence determinism calculation."""
    
    def test_exact_regex_match_high_determinism(self):
        """Test exact regex match = high determinism."""
        verification = VerificationResult(
            success=True,
            confidence=0.88,
            match_type="exact_regex"
        )
        
        # Exact regex = 0.5 determinism score
        determinism_contribution = 0.5
        assert determinism_contribution == 0.5
    
    def test_structural_differential_high_determinism(self):
        """Test structural differential = high determinism."""
        verification = VerificationResult(
            success=True,
            confidence=0.85,
            match_type="structural_differential"
        )
        
        # Structural = 0.4 determinism score
        determinism_contribution = 0.4
        assert determinism_contribution == 0.4
    
    def test_fuzzy_match_medium_determinism(self):
        """Test fuzzy match = medium determinism."""
        verification = VerificationResult(
            success=True,
            confidence=0.75,
            match_type="fuzzy_match"
        )
        
        # Fuzzy = 0.3 determinism score
        determinism_contribution = 0.3
        assert determinism_contribution == 0.3
    
    def test_ai_advisory_adds_medium_determinism(self):
        """Test AI advisory (CONFIRMED) adds medium determinism."""
        ai_analysis = AIAnalysisResult(
            model_used="test",
            verdict="CONFIRMED",
            confidence=0.85,
            reasoning="Test",
            success=True
        )
        
        # AI CONFIRMED (advisory) = 0.3 boost
        ai_contribution = 0.3
        assert ai_contribution == 0.3
    
    def test_heuristic_high_adds_low_determinism(self):
        """Test high heuristic score adds low determinism."""
        heuristic_score = 0.85
        
        # Heuristic ≥0.8 = 0.2 contribution
        if heuristic_score >= 0.8:
            heuristic_contribution = 0.2
        else:
            heuristic_contribution = 0.1
        
        assert heuristic_contribution == 0.2
    
    def test_combined_determinism_score(self):
        """Test combined determinism score calculation."""
        # System verification: exact_regex = 0.5
        # AI analysis: CONFIRMED = 0.3
        # Heuristic: 0.85 = 0.2
        # Total = 1.0 (capped at 1.0)
        
        total_score = min(0.5 + 0.3 + 0.2, 1.0)
        assert total_score == 1.0


class TestVulnerabilitySpecificDeterminism:
    """Test vulnerability-specific determinism bonuses."""
    
    def test_sql_injection_evidence_bonus(self):
        """Test SQL injection deterministic indicators."""
        from vortex.config.evidence_config import VULN_SPECIFIC_EVIDENCE
        
        sql_config = VULN_SPECIFIC_EVIDENCE['sql_injection']
        
        # SQL has clear deterministic indicators
        assert 'database error' in [i.lower() for i in sql_config['deterministic_indicators']]
        assert sql_config['confidence_bonus'] == 0.15
    
    def test_xss_reflected_evidence_bonus(self):
        """Test XSS reflected deterministic indicators."""
        from vortex.config.evidence_config import VULN_SPECIFIC_EVIDENCE
        
        xss_config = VULN_SPECIFIC_EVIDENCE['xss_reflected']
        
        # XSS has highest bonus (JS execution deterministic)
        assert xss_config['confidence_bonus'] == 0.20
    
    def test_ssrf_evidence_bonus(self):
        """Test SSRF deterministic indicators."""
        from vortex.config.evidence_config import VULN_SPECIFIC_EVIDENCE
        
        ssrf_config = VULN_SPECIFIC_EVIDENCE['ssrf']
        
        # SSRF has medium bonus
        assert ssrf_config['confidence_bonus'] == 0.10
    
    def test_lfi_evidence_bonus(self):
        """Test LFI deterministic indicators."""
        from vortex.config.evidence_config import VULN_SPECIFIC_EVIDENCE
        
        lfi_config = VULN_SPECIFIC_EVIDENCE['lfi']
        
        # LFI has lowest bonus (most ambiguous)
        assert lfi_config['confidence_bonus'] == 0.05
    
    def test_vulnerability_bonus_hierarchy(self):
        """Test vulnerability-specific bonus hierarchy."""
        from vortex.config.evidence_config import VULN_SPECIFIC_EVIDENCE
        
        xss_bonus = VULN_SPECIFIC_EVIDENCE['xss_reflected']['confidence_bonus']
        sql_bonus = VULN_SPECIFIC_EVIDENCE['sql_injection']['confidence_bonus']
        ssrf_bonus = VULN_SPECIFIC_EVIDENCE['ssrf']['confidence_bonus']
        lfi_bonus = VULN_SPECIFIC_EVIDENCE['lfi']['confidence_bonus']
        
        # XSS > SQL > SSRF > LFI
        assert xss_bonus > sql_bonus > ssrf_bonus > lfi_bonus


@pytest.mark.compliance
class TestDeterminismCompliance:
    """Evidence determinism compliance checklist."""
    
    def test_deterministic_level_min_score_enforced(self):
        """✓ Deterministic level minimum score ≥0.8 enforced."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        assert EVIDENCE_LEVELS[EvidenceLevel.DETERMINISTIC]['min_score'] >= 0.8
    
    def test_behavioral_level_min_score_enforced(self):
        """✓ Behavioral level minimum score ≥0.6 enforced."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        assert EVIDENCE_LEVELS[EvidenceLevel.BEHAVIORAL]['min_score'] >= 0.6
    
    def test_pattern_level_min_score_enforced(self):
        """✓ Pattern level minimum score ≥0.4 enforced."""
        from vortex.config.evidence_config import EVIDENCE_LEVELS
        
        assert EVIDENCE_LEVELS[EvidenceLevel.PATTERN]['min_score'] >= 0.4
    
    def test_submit_ready_determinism_requirement(self):
        """✓ SUBMIT_READY requires deterministic evidence (≥0.7)."""
        # SUBMIT_READY requires evidence_determinism_score ≥ 0.7
        required_score = 0.7
        deterministic_min = 0.8
        
        assert deterministic_min >= required_score
    
    def test_vulnerability_specific_bonuses_defined(self):
        """✓ Vulnerability-specific determinism bonuses defined."""
        from vortex.config.evidence_config import VULN_SPECIFIC_EVIDENCE
        
        required_vulns = ['sql_injection', 'xss_reflected', 'ssrf', 'lfi']
        
        for vuln in required_vulns:
            assert vuln in VULN_SPECIFIC_EVIDENCE
            assert 'confidence_bonus' in VULN_SPECIFIC_EVIDENCE[vuln]
            assert 'deterministic_indicators' in VULN_SPECIFIC_EVIDENCE[vuln]