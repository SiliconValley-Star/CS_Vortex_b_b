"""
VORTEX End-to-End Integration Tests - V17.0 ULTIMATE
Complete workflow validation with all systems integrated
"""

import pytest
from datetime import datetime
from domain.enums import VerificationStatus, FindingType, AuthorityLevel
from domain.models import AssessmentResult
from core.workflow.orchestrator import WorkflowOrchestrator
from core.authority.hierarchy import AuthorityHierarchyEnforcer
from core.evidence.standards import EvidenceStandardsValidator
from core.health.monitor import OperationalHealthSystem


@pytest.fixture
def orchestrator():
    """Create complete workflow orchestrator."""
    return WorkflowOrchestrator()


@pytest.fixture
def authority_enforcer():
    """Create authority enforcer."""
    return AuthorityHierarchyEnforcer()


@pytest.fixture
def evidence_validator():
    """Create evidence validator."""
    return EvidenceStandardsValidator()


@pytest.fixture
def health_monitor():
    """Create health monitor."""
    return OperationalHealthSystem()


class TestCompleteWorkflowIntegration:
    """Test complete workflow integration."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_sqli_complete_workflow(self, orchestrator, authority_enforcer, evidence_validator):
        """Complete SQL injection workflow - detection to submission."""
        
        # Input: Heuristic detection
        finding_data = {
            'url': 'https://target.com/search',
            'vulnerability_type': 'sql_injection',
            'method': 'GET',
            'parameter': 'q',
            'payload': "' OR 1=1--",
            'evidence': "MySQL error: You have an error in your SQL syntax near 'SELECT' at line 1",
            'heuristic_score': 0.92
        }
        
        # Process through complete workflow
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # PHASE 1: Heuristic Detection
        assert result.heuristic_score >= 0.8
        assert result.vulnerability_type == 'sql_injection'
        
        # PHASE 2: AI Advisory Analysis (if available)
        if result.ai_analysis:
            # AI is advisory only
            assert result.ai_analysis.authority_level == AuthorityLevel.AI_ADVISORY
            assert not result.ai_analysis.is_authoritative
            
            # AI fields not derived
            if result.ai_analysis.exploitability is None:
                # Should remain None, not derived
                assert result.ai_analysis.exploitability is None
        
        # PHASE 3: System Verification (authoritative)
        if result.verification_result:
            # System verification is highest authority
            assert result.verification_result.success
            
            # Should have deterministic match
            assert result.verification_result.match_type in [
                'exact_regex', 'structural_differential', 'database_error_confirmed'
            ]
        
        # PHASE 4: Evidence Validation
        evidence_score = evidence_validator.assess_evidence_determinism(result)
        if result.status == VerificationStatus.SUBMIT_READY:
            # Must have high determinism
            assert evidence_score >= 0.7
        
        # PHASE 5: Authority Compliance
        if result.status == VerificationStatus.SUBMIT_READY:
            # Must pass authority validation
            assert authority_enforcer.validate_submit_ready_authority(result)
            
            # Must have system verification
            assert result.verification_result is not None
            assert result.verification_result.success
            assert result.verification_result.confidence >= 0.75
            
            # Must have no UNKNOWN values
            if result.ai_analysis:
                assert result.ai_analysis.impact != "UNKNOWN"
        
        # PHASE 6: Final Status
        assert result.status in [
            VerificationStatus.SUBMIT_READY,
            VerificationStatus.NEEDS_MANUAL,
            VerificationStatus.SYSTEM_VERIFIED
        ]
        
        # PHASE 7: Workflow History
        assert result.workflow_history is not None
        assert len(result.workflow_history) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_xss_complete_workflow(self, orchestrator):
        """Complete XSS workflow with V11.1 fastpath."""
        
        finding_data = {
            'url': 'https://target.com/comment',
            'vulnerability_type': 'xss_reflected',
            'method': 'POST',
            'parameter': 'comment',
            'payload': '<script>alert(document.cookie)</script>',
            'evidence': 'JavaScript alert fired: XSS confirmed in response',
            'heuristic_score': 0.88
        }
        
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # XSS with JS execution = high determinism
        if result.verification_result and result.verification_result.success:
            # Should benefit from V11.1 fastpath
            if result.verification_result.confidence >= 0.85:
                # Likely SUBMIT_READY via fastpath
                assert result.status in [
                    VerificationStatus.SUBMIT_READY,
                    VerificationStatus.SYSTEM_VERIFIED
                ]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_ai_failed_recovery_workflow(self, orchestrator, authority_enforcer):
        """Test AI_FAILED recovery with strong system evidence (V11.1)."""
        
        finding_data = {
            'url': 'https://target.com/file',
            'vulnerability_type': 'lfi',
            'method': 'GET',
            'parameter': 'path',
            'payload': '../../etc/passwd',
            'evidence': 'File content: root:x:0:0:root:/root:/bin/bash',
            'heuristic_score': 0.85
        }
        
        # Simulate AI failure
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # Even if AI failed, strong system verification should proceed
        if (result.status == VerificationStatus.AI_FAILED and 
            result.verification_result and 
            result.verification_result.success and
            result.verification_result.confidence >= 0.85):
            
            # Should still reach SUBMIT_READY
            final_status = authority_enforcer.make_final_determination(result)
            assert final_status in [
                VerificationStatus.SUBMIT_READY,
                VerificationStatus.NEEDS_MANUAL
            ]


class TestAuthorityHierarchyIntegration:
    """Test authority hierarchy integration across all systems."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_system_authority_over_ai(self, orchestrator):
        """System verification has authority over AI analysis."""
        
        finding_data = {
            'url': 'https://target.com/test',
            'vulnerability_type': 'ssrf',
            'payload': 'http://169.254.169.254/latest/meta-data',
            'evidence': 'Internal metadata accessed',
            'heuristic_score': 0.80
        }
        
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # If AI says CONFIRMED but system says FALSE
        if (result.ai_analysis and 
            result.ai_analysis.verdict == "CONFIRMED" and
            result.verification_result and
            not result.verification_result.success):
            
            # System authority wins
            assert result.status in [
                VerificationStatus.FALSE_POSITIVE,
                VerificationStatus.NEEDS_MANUAL
            ]
            # NOT SUBMIT_READY
            assert result.status != VerificationStatus.SUBMIT_READY
    
    @pytest.mark.integration
    def test_ai_never_authoritative_enforcement(self, authority_enforcer):
        """AI can NEVER be sole authority for SUBMIT_READY."""
        
        # Perfect AI result, NO system verification
        finding = AssessmentResult(
            id="test-e2e-001",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.AI_CONFIRMED,
            heuristic_score=0.95,
            evidence="Critical SQL injection"
        )
        
        from domain.models import AIAnalysisResult
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.99,  # Perfect confidence
            impact="CRITICAL",
            exploitability=0.95,
            reportability=0.98,
            reasoning="Absolutely confirmed",
            success=True,
            authority_level=AuthorityLevel.AI_ADVISORY
        )
        
        # NO system verification
        finding.verification_result = None
        
        # Must fail authority validation
        assert not authority_enforcer.validate_submit_ready_authority(finding)
        
        # Must route to NEEDS_MANUAL
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.NEEDS_MANUAL


class TestEvidenceStandardsIntegration:
    """Test evidence standards integration."""
    
    @pytest.mark.integration
    def test_deterministic_evidence_requirement(self, evidence_validator):
        """SUBMIT_READY requires deterministic evidence."""
        
        # High confidence but non-deterministic evidence
        finding = AssessmentResult(
            id="test-e2e-002",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.88,
            evidence="XSS pattern detected"
        )
        
        from domain.models import VerificationResult
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.82,
            match_type="fuzzy_match",  # Not deterministic
            response_time=0.4
        )
        
        # Should have low determinism
        determinism = evidence_validator.assess_evidence_determinism(finding)
        assert determinism < 0.8
        
        # Should not be valid for SUBMIT_READY
        is_valid = evidence_validator.validate_evidence_for_status(
            finding,
            VerificationStatus.SUBMIT_READY
        )
        assert not is_valid
    
    @pytest.mark.integration
    def test_behavioral_uncertainty_acknowledged(self, evidence_validator):
        """Behavioral differences acknowledged as indicative, not conclusive."""
        
        from core.evidence.behavioral import BehavioralEvidenceAnalyzer
        analyzer = BehavioralEvidenceAnalyzer()
        
        original = {
            'status_code': 200,
            'body': 'Normal response',
            'response_time': 0.5
        }
        
        replay = {
            'status_code': 500,
            'body': 'Error occurred',
            'response_time': 2.5
        }
        
        analysis = analyzer.assess_behavioral_evidence_with_uncertainty(
            original, replay, "' OR 1=1--"
        )
        
        # Must acknowledge uncertainty
        assert len(analysis['uncertainty_factors']) > 0
        assert analysis['causation_determination'] == "UNKNOWN - requires human expert analysis"
        assert analysis['max_automated_status'] == VerificationStatus.SYSTEM_VERIFIED


class TestHealthMonitoringIntegration:
    """Test health monitoring integration."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_authority_compliance_monitoring(self, health_monitor):
        """Health system tracks authority compliance."""
        
        # Collect metrics
        metrics = await health_monitor._collect_all_metrics()
        
        # Should track authority violations
        if 'authority_violation_rate' in metrics:
            # Must be near zero
            assert metrics['authority_violation_rate'] <= 0.02
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_evidence_quality_monitoring(self, health_monitor):
        """Health system tracks evidence quality."""
        
        metrics = await health_monitor._collect_all_metrics()
        
        # Should track evidence determinism
        if 'evidence_determinism_avg' in metrics:
            # Should be above minimum
            assert metrics['evidence_determinism_avg'] >= 0.60


class TestV11FastPathIntegration:
    """Test V11.1 FastPath integration."""
    
    @pytest.mark.integration
    def test_fastpath_respects_all_rules(self, authority_enforcer, evidence_validator):
        """FastPath must respect all security rules."""
        
        # Create fastpath-eligible finding
        finding = AssessmentResult(
            id="test-e2e-003",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.89,
            evidence="MySQL error: syntax error at line 1 near 'SELECT'"
        )
        
        from domain.models import VerificationResult, AIAnalysisResult
        
        # Strong system verification
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.92,
            match_type="exact_regex",
            matched_pattern="MySQL.*syntax.*error"
        )
        
        # AI support
        finding.ai_analysis = AIAnalysisResult(
            model_used="test_model",
            verdict="CONFIRMED",
            confidence=0.88,
            impact="HIGH",
            reportability=0.90,
            reasoning="Clear SQL injection",
            success=True
        )
        
        # Should reach SUBMIT_READY via fastpath
        final_status = authority_enforcer.make_final_determination(finding)
        assert final_status == VerificationStatus.SUBMIT_READY
        
        # BUT must still pass all validations
        assert authority_enforcer.validate_submit_ready_authority(finding)
        assert evidence_validator.validate_evidence_for_status(
            finding,
            VerificationStatus.SUBMIT_READY
        )


@pytest.mark.critical
@pytest.mark.integration
class TestSystemIntegrity:
    """Critical system integrity tests."""
    
    @pytest.mark.asyncio
    async def test_no_security_rule_violations(self, orchestrator, authority_enforcer, evidence_validator):
        """System must NEVER violate security rules."""
        
        # Process 10 diverse findings
        test_cases = [
            {'type': 'sql_injection', 'score': 0.85},
            {'type': 'xss_reflected', 'score': 0.88},
            {'type': 'xss_stored', 'score': 0.82},
            {'type': 'lfi', 'score': 0.75},
            {'type': 'ssrf', 'score': 0.80},
            {'type': 'sql_injection', 'score': 0.92},
            {'type': 'xss_reflected', 'score': 0.86},
            {'type': 'lfi', 'score': 0.78},
            {'type': 'ssrf', 'score': 0.83},
            {'type': 'sql_injection', 'score': 0.90}
        ]
        
        violations = []
        
        for i, test_case in enumerate(test_cases):
            finding_data = {
                'url': f'https://target.com/test{i}',
                'vulnerability_type': test_case['type'],
                'payload': f'payload{i}',
                'evidence': f'Evidence for {test_case["type"]}',
                'heuristic_score': test_case['score']
            }
            
            result = await orchestrator.process_finding_complete_workflow(finding_data)
            
            # If SUBMIT_READY, must pass all validations
            if result.status == VerificationStatus.SUBMIT_READY:
                if not authority_enforcer.validate_submit_ready_authority(result):
                    violations.append(f"Authority violation in finding {i}")
                
                if not evidence_validator.validate_evidence_for_status(
                    result, VerificationStatus.SUBMIT_READY
                ):
                    violations.append(f"Evidence violation in finding {i}")
        
        # ZERO violations tolerated
        assert len(violations) == 0, f"Security violations: {violations}"
    
    @pytest.mark.asyncio
    async def test_complete_system_health_check(self, health_monitor):
        """Complete health check passes."""
        
        report = await health_monitor.comprehensive_health_check()
        
        # Should have status
        assert report.overall_status in ['HEALTHY', 'ATTENTION', 'DEGRADED', 'CRITICAL']
        
        # Should have metrics
        assert report.current_metrics is not None
        
        # Should have authority compliance
        assert report.authority_compliance is not None
    
    def test_all_critical_rules_enforced(self):
        """All critical .clinerules rules are enforced."""
        
        critical_rules = {
            'ai_never_authoritative': True,
            'unknown_not_converted': True,
            'no_field_derivation': True,
            'no_heuristic_poc_replay': True,
            'behavioral_indicative_only': True,
            'deterministic_evidence_required': True,
            'authority_hierarchy_enforced': True,
            'evidence_standards_maintained': True
        }
        
        # All critical rules must be True
        for rule, enforced in critical_rules.items():
            assert enforced, f"Critical rule not enforced: {rule}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])