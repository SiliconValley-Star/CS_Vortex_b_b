"""
VORTEX Workflow Integrity Compliance Tests - V17.0 ULTIMATE
Critical validation: Workflow state transitions never violated
"""

import pytest
from datetime import datetime, timedelta
from domain.enums import VerificationStatus, FindingType
from domain.models import AssessmentResult, AIAnalysisResult, VerificationResult
from core.workflow.state_machine import WorkflowStateMachine
from core.workflow.orchestrator import WorkflowOrchestrator


@pytest.fixture
def state_machine():
    """Create workflow state machine."""
    return WorkflowStateMachine()


@pytest.fixture
def orchestrator():
    """Create workflow orchestrator."""
    return WorkflowOrchestrator()


class TestStateMachineTransitions:
    """Test state machine transition rules."""
    
    def test_valid_transitions_defined(self, state_machine):
        """All valid transitions must be explicitly defined."""
        valid_transitions = state_machine.get_valid_transitions()
        
        # Check all statuses have transition rules
        all_statuses = [
            VerificationStatus.DETECTED,
            VerificationStatus.AI_ANALYSIS_PENDING,
            VerificationStatus.AI_CONFIRMED,
            VerificationStatus.AI_FAILED,
            VerificationStatus.SYSTEM_VERIFICATION_PENDING,
            VerificationStatus.SYSTEM_VERIFIED,
            VerificationStatus.SYSTEM_VERIFICATION_FAILED,
            VerificationStatus.SUBMIT_READY,
            VerificationStatus.NEEDS_MANUAL,
            VerificationStatus.FALSE_POSITIVE,
            VerificationStatus.ERROR_STATE
        ]
        
        for status in all_statuses:
            assert status in valid_transitions
    
    def test_terminal_states_no_transitions(self, state_machine):
        """Terminal states must have no valid transitions."""
        terminal_states = [
            VerificationStatus.SUBMIT_READY,
            VerificationStatus.FALSE_POSITIVE
        ]
        
        for terminal_state in terminal_states:
            transitions = state_machine.get_valid_transitions().get(terminal_state, [])
            assert len(transitions) == 0, f"{terminal_state} should have no transitions"
    
    def test_error_state_only_to_manual(self, state_machine):
        """ERROR_STATE should only transition to NEEDS_MANUAL."""
        error_transitions = state_machine.get_valid_transitions().get(
            VerificationStatus.ERROR_STATE, []
        )
        
        assert len(error_transitions) == 1
        assert error_transitions[0] == VerificationStatus.NEEDS_MANUAL
    
    def test_invalid_transition_rejected(self, state_machine):
        """Invalid transitions must be rejected."""
        finding = AssessmentResult(
            id="test-wf-001",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.DETECTED,
            heuristic_score=0.80,
            evidence="SQL pattern"
        )
        
        # Try invalid transition: DETECTED -> SUBMIT_READY (must go through intermediate states)
        is_valid = state_machine.validate_transition(
            finding.status,
            VerificationStatus.SUBMIT_READY
        )
        
        assert not is_valid
    
    def test_ai_failed_can_reach_submit_ready(self, state_machine):
        """V11.1: AI_FAILED can reach SUBMIT_READY with strong system evidence."""
        # AI_FAILED -> SUBMIT_READY should be valid transition
        transitions = state_machine.get_valid_transitions().get(
            VerificationStatus.AI_FAILED, []
        )
        
        assert VerificationStatus.SUBMIT_READY in transitions


class TestWorkflowPhaseProgression:
    """Test workflow phase progression rules."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_progression(self, orchestrator):
        """Test complete workflow from detection to final determination."""
        finding_data = {
            'url': 'https://target.com/test',
            'vulnerability_type': 'sql_injection',
            'method': 'GET',
            'parameter': 'id',
            'payload': "' OR 1=1--",
            'evidence': "MySQL error: syntax error"
        }
        
        # Process through complete workflow
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # Should have progressed through phases
        assert result.workflow_history is not None
        assert len(result.workflow_history) > 0
        
        # Final status should be terminal or active
        assert result.status in [
            VerificationStatus.SUBMIT_READY,
            VerificationStatus.NEEDS_MANUAL,
            VerificationStatus.FALSE_POSITIVE,
            VerificationStatus.ERROR_STATE
        ]
    
    def test_phase_order_enforced(self, orchestrator):
        """Workflow phases must execute in correct order."""
        expected_phase_order = [
            'initialize_finding',
            'heuristic_detection',
            'ai_advisory_analysis',
            'system_verification',
            'evidence_validation',
            'final_determination',
            'quality_assurance',
            'workflow_completion'
        ]
        
        # Verify orchestrator has all phases
        for phase in expected_phase_order:
            method_name = f"_{phase}_phase"
            assert hasattr(orchestrator, method_name), f"Missing phase: {phase}"
    
    @pytest.mark.asyncio
    async def test_phase_failure_handling(self, orchestrator):
        """Phase failures should be handled gracefully."""
        finding_data = {
            'url': 'invalid-url',  # Will cause validation error
            'vulnerability_type': 'unknown_type',
            'evidence': 'test'
        }
        
        # Should handle gracefully
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # Should route to error state or needs manual
        assert result.status in [
            VerificationStatus.ERROR_STATE,
            VerificationStatus.NEEDS_MANUAL
        ]


class TestManualReviewActivation:
    """Test NEEDS_MANUAL active status management."""
    
    def test_needs_manual_is_active_status(self):
        """CRITICAL: NEEDS_MANUAL is ACTIVE status, not terminal."""
        finding = AssessmentResult(
            id="test-wf-002",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            status=VerificationStatus.NEEDS_MANUAL,
            heuristic_score=0.82,
            evidence="XSS pattern"
        )
        
        # NEEDS_MANUAL should be active (requires action)
        assert not finding.is_terminal_state()
        assert finding.requires_manual_review()
    
    def test_manual_review_sla_tracking(self):
        """Manual review findings must have SLA tracking."""
        from core.workflow.manual_review import ManualReviewManager
        
        manager = ManualReviewManager()
        
        finding = AssessmentResult(
            id="test-wf-003",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.NEEDS_MANUAL,
            heuristic_score=0.88,
            evidence="SQL injection suspected"
        )
        
        # Register for manual review
        manager.register_manual_review(finding)
        
        # Should have SLA
        sla = manager.get_sla(finding.id)
        assert sla is not None
        assert sla.assigned_at is not None
        assert sla.max_age_hours > 0
    
    def test_priority_calculation(self):
        """Manual review priority should be calculated correctly."""
        from core.workflow.manual_review import ManualReviewManager
        
        manager = ManualReviewManager()
        
        # High priority finding
        high_priority = AssessmentResult(
            id="test-wf-004",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            severity="CRITICAL",
            status=VerificationStatus.NEEDS_MANUAL,
            heuristic_score=0.92,
            evidence="Critical SQL injection"
        )
        
        high_priority.verification_result = VerificationResult(
            success=True,
            confidence=0.85,
            match_type="exact_regex"
        )
        
        priority_high = manager._calculate_priority(high_priority)
        
        # Low priority finding
        low_priority = AssessmentResult(
            id="test-wf-005",
            url="https://target.com/test",
            vulnerability_type="lfi",
            severity="LOW",
            status=VerificationStatus.NEEDS_MANUAL,
            heuristic_score=0.65,
            evidence="Possible LFI"
        )
        
        priority_low = manager._calculate_priority(low_priority)
        
        # High priority should be lower number (higher priority)
        assert priority_high < priority_low


class TestSystemVerifiedFormatting:
    """Test SYSTEM_VERIFIED formatting restrictions."""
    
    def test_system_verified_neutral_tone_required(self):
        """SYSTEM_VERIFIED must use neutral technical tone only."""
        finding = AssessmentResult(
            id="test-wf-006",
            url="https://target.com/test",
            vulnerability_type="xss_stored",
            status=VerificationStatus.SYSTEM_VERIFIED,
            heuristic_score=0.84,
            evidence="Stored XSS pattern"
        )
        
        finding.verification_result = VerificationResult(
            success=True,
            confidence=0.80,
            match_type="structural_differential"
        )
        
        # Format finding
        from core.workflow.formatting import format_system_verified_finding
        formatted = format_system_verified_finding(finding)
        
        # Should contain neutral indicators
        assert "requires human" in formatted.lower() or "manual review" in formatted.lower()
        
        # Should NOT contain submission language
        forbidden_phrases = [
            "confirmed vulnerability",
            "ready for submission",
            "exploit confirmed",
            "definitely vulnerable"
        ]
        
        formatted_lower = formatted.lower()
        for phrase in forbidden_phrases:
            assert phrase not in formatted_lower
    
    def test_submission_language_validation(self):
        """Validate SYSTEM_VERIFIED content doesn't contain submission language."""
        from core.workflow.formatting import validate_system_verified_content
        
        # Valid neutral content
        valid_content = """
        Technical Finding: SQL Injection
        System Verification: exact_regex
        Status: Requires human expert causation analysis
        Next Step: Manual review for business impact determination
        """
        
        assert validate_system_verified_content(valid_content)
        
        # Invalid submission language
        invalid_content = """
        Confirmed vulnerability ready for submission.
        This is definitely vulnerable and exploit confirmed.
        """
        
        assert not validate_system_verified_content(invalid_content)


class TestWorkflowErrorRecovery:
    """Test workflow error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_ai_failure_recovery(self, orchestrator):
        """AI analysis failure should not block workflow."""
        finding_data = {
            'url': 'https://target.com/test',
            'vulnerability_type': 'sql_injection',
            'evidence': 'SQL error detected'
        }
        
        # Simulate AI failure
        with pytest.MonkeyPatch.context() as mp:
            async def mock_ai_fail(*args, **kwargs):
                raise Exception("AI service unavailable")
            
            mp.setattr(
                'core.ai.advisory.ProductionAIIntegrationEngine.ai_advisory_analysis',
                mock_ai_fail
            )
            
            result = await orchestrator.process_finding_complete_workflow(finding_data)
            
            # Should continue to system verification
            assert result.status in [
                VerificationStatus.SYSTEM_VERIFIED,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.SUBMIT_READY  # If strong system evidence
            ]
    
    @pytest.mark.asyncio
    async def test_system_verification_failure_handling(self, orchestrator):
        """System verification failure should route appropriately."""
        finding_data = {
            'url': 'https://target.com/test',
            'vulnerability_type': 'xss_reflected',
            'evidence': 'XSS pattern'
        }
        
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # If system verification fails, should consider AI confidence
        if result.verification_result and not result.verification_result.success:
            if result.ai_analysis and result.ai_analysis.reportability > 0.6:
                assert result.status == VerificationStatus.NEEDS_MANUAL
            else:
                assert result.status in [
                    VerificationStatus.FALSE_POSITIVE,
                    VerificationStatus.NEEDS_MANUAL
                ]
    
    def test_error_state_recovery(self, state_machine):
        """ERROR_STATE should allow recovery to NEEDS_MANUAL."""
        finding = AssessmentResult(
            id="test-wf-007",
            url="https://target.com/test",
            vulnerability_type="ssrf",
            status=VerificationStatus.ERROR_STATE,
            heuristic_score=0.75,
            evidence="SSRF pattern"
        )
        
        # Should allow transition to NEEDS_MANUAL
        is_valid = state_machine.validate_transition(
            VerificationStatus.ERROR_STATE,
            VerificationStatus.NEEDS_MANUAL
        )
        
        assert is_valid


class TestWorkflowMetrics:
    """Test workflow metrics and monitoring."""
    
    def test_workflow_history_recorded(self):
        """All workflow transitions should be recorded."""
        finding = AssessmentResult(
            id="test-wf-008",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.DETECTED,
            heuristic_score=0.85,
            evidence="SQL pattern"
        )
        
        # Initialize workflow history
        finding.workflow_history = []
        
        # Simulate transitions
        transitions = [
            (VerificationStatus.DETECTED, "Initial detection"),
            (VerificationStatus.AI_ANALYSIS_PENDING, "Starting AI analysis"),
            (VerificationStatus.AI_CONFIRMED, "AI confirmed"),
            (VerificationStatus.SYSTEM_VERIFICATION_PENDING, "Starting verification"),
            (VerificationStatus.SYSTEM_VERIFIED, "System verified"),
            (VerificationStatus.SUBMIT_READY, "Ready for submission")
        ]
        
        for status, reason in transitions:
            finding.transition_state(status, reason)
        
        # Should have all transitions recorded
        assert len(finding.workflow_history) == len(transitions)
        
        # Each transition should have timestamp
        for entry in finding.workflow_history:
            assert 'timestamp' in entry
            assert 'from_status' in entry
            assert 'to_status' in entry
            assert 'reason' in entry
    
    def test_workflow_duration_tracking(self):
        """Workflow duration should be tracked."""
        finding = AssessmentResult(
            id="test-wf-009",
            url="https://target.com/test",
            vulnerability_type="xss_reflected",
            status=VerificationStatus.DETECTED,
            heuristic_score=0.80,
            evidence="XSS pattern",
            created_at=datetime.utcnow()
        )
        
        # Simulate some time passing
        import time
        time.sleep(0.1)
        
        finding.status = VerificationStatus.SUBMIT_READY
        finding.completed_at = datetime.utcnow()
        
        # Should have duration
        duration = (finding.completed_at - finding.created_at).total_seconds()
        assert duration > 0


@pytest.mark.critical
class TestWorkflowIntegrityEnforcement:
    """Critical workflow integrity enforcement tests."""
    
    def test_no_invalid_state_bypass(self, state_machine):
        """Invalid states must never bypass validation."""
        # Try all possible invalid transitions
        invalid_transitions = [
            (VerificationStatus.DETECTED, VerificationStatus.SUBMIT_READY),
            (VerificationStatus.AI_ANALYSIS_PENDING, VerificationStatus.SYSTEM_VERIFIED),
            (VerificationStatus.SUBMIT_READY, VerificationStatus.DETECTED),
            (VerificationStatus.FALSE_POSITIVE, VerificationStatus.AI_CONFIRMED)
        ]
        
        for from_status, to_status in invalid_transitions:
            is_valid = state_machine.validate_transition(from_status, to_status)
            assert not is_valid, f"Invalid transition allowed: {from_status} -> {to_status}"
    
    def test_workflow_consistency_check(self, orchestrator):
        """Workflow state must remain consistent."""
        finding = AssessmentResult(
            id="test-wf-010",
            url="https://target.com/test",
            vulnerability_type="sql_injection",
            status=VerificationStatus.DETECTED,
            heuristic_score=0.85,
            evidence="SQL injection"
        )
        
        # Check consistency
        is_consistent = orchestrator.validate_workflow_consistency(finding)
        assert is_consistent
        
        # Create inconsistent state
        finding.status = VerificationStatus.SUBMIT_READY
        finding.verification_result = None  # Missing required verification
        
        # Should detect inconsistency
        is_consistent = orchestrator.validate_workflow_consistency(finding)
        assert not is_consistent
    
    @pytest.mark.asyncio
    async def test_workflow_completion_validation(self, orchestrator):
        """Workflow completion must validate all requirements."""
        finding_data = {
            'url': 'https://target.com/test',
            'vulnerability_type': 'xss_stored',
            'evidence': 'Stored XSS detected'
        }
        
        result = await orchestrator.process_finding_complete_workflow(finding_data)
        
        # If SUBMIT_READY, must pass validation
        if result.status == VerificationStatus.SUBMIT_READY:
            # Must have system verification
            assert result.verification_result is not None
            assert result.verification_result.success
            
            # Must have high confidence
            assert result.verification_result.confidence >= 0.75
            
            # Must have no UNKNOWN values
            if result.ai_analysis:
                assert result.ai_analysis.impact != "UNKNOWN"
            
            # Must have workflow history
            assert result.workflow_history is not None
            assert len(result.workflow_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])