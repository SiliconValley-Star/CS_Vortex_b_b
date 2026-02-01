"""
VORTEX Workflow Orchestrator - V17.0 ULTIMATE
Complete workflow orchestration per VORTEX_WORKFLOW_LIFECYCLE.md

CRITICAL: Orchestrates all systems with authority hierarchy enforcement
"""

import structlog
from typing import Dict, Optional
from datetime import datetime
import uuid

from domain.enums import VerificationStatus, FindingType
from domain.models import AssessmentResult, VerificationResult
from core.authority import global_authority_enforcer, global_authority_validator
from core.evidence import global_evidence_validator, global_behavioral_analyzer
from core.ai import global_ai_advisory_engine
from core.workflow.state_machine import global_state_machine

logger = structlog.get_logger()


class WorkflowOrchestrator:
    """
    Orchestrates complete finding workflow
    Per VORTEX_WORKFLOW_LIFECYCLE.md: End-to-end workflow management
    """
    
    def __init__(self):
        # Component integration
        self.authority_enforcer = global_authority_enforcer
        self.authority_validator = global_authority_validator
        self.evidence_validator = global_evidence_validator
        self.behavioral_analyzer = global_behavioral_analyzer
        self.ai_engine = global_ai_advisory_engine
        self.state_machine = global_state_machine
        
        # Workflow tracking
        self.workflow_history = []
        self.active_workflows = {}
    
    async def process_finding_workflow(
        self,
        finding_data: Dict,
        finding: Optional[AssessmentResult] = None
    ) -> AssessmentResult:
        """
        Process complete finding workflow
        Per VORTEX_WORKFLOW_LIFECYCLE.md: All phases executed
        
        Args:
            finding_data: Finding information dict
            finding: Optional existing AssessmentResult
        
        Returns: Processed AssessmentResult
        """
        workflow_id = str(uuid.uuid4())
        
        logger.info(
            "Starting complete workflow",
            workflow_id=workflow_id,
            finding_id=finding_data.get('id', 'unknown'),
            url=finding_data.get('url', 'unknown')
        )
        
        try:
            # Phase 1: Initialize or use existing finding
            if not finding:
                finding = self._initialize_finding(finding_data, workflow_id)
            
            # Track workflow
            self.active_workflows[workflow_id] = {
                'finding_id': str(finding.id),
                'start_time': datetime.utcnow(),
                'phase': 'initialization'
            }
            
            # Phase 2: Heuristic detection (already done if finding exists)
            if not finding.heuristic_score:
                await self._heuristic_detection_phase(finding)
            
            # Phase 3: AI advisory analysis
            await self._ai_advisory_analysis_phase(finding)
            
            # Phase 4: System verification
            await self._system_verification_phase(finding)
            
            # Phase 5: Evidence validation
            await self._evidence_validation_phase(finding)
            
            # Phase 6: Authority-compliant final determination
            await self._final_determination_phase(finding)
            
            # Phase 7: Quality assurance
            await self._quality_assurance_phase(finding)
            
            # Phase 8: Workflow completion
            await self._workflow_completion_phase(finding, workflow_id)
            
            logger.info(
                "Workflow completed successfully",
                workflow_id=workflow_id,
                finding_id=str(finding.id),
                final_status=finding.status.value
            )
            
            return finding
            
        except Exception as e:
            logger.error(
                "Workflow failed",
                workflow_id=workflow_id,
                error=str(e)
            )
            return await self._handle_workflow_error(finding_data, workflow_id, e)
        
        finally:
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    def _initialize_finding(self, finding_data: Dict, workflow_id: str) -> AssessmentResult:
        """Initialize new finding from data."""
        logger.debug(f"Initializing finding for workflow {workflow_id}")
        
        # Parse finding type
        finding_type_str = finding_data.get('finding_type', 'UNKNOWN')
        try:
            finding_type = FindingType[finding_type_str.upper().replace(' ', '_')]
        except KeyError:
            finding_type = FindingType.OTHER
        
        # Create AssessmentResult
        finding = AssessmentResult(
            url=finding_data.get('url', ''),
            finding_type=finding_type,
            severity=finding_data.get('severity', 'MEDIUM'),
            heuristic_score=finding_data.get('heuristic_score', 0.0),
            evidence=finding_data.get('evidence', ''),
            vulnerable_parameter=finding_data.get('parameter', ''),
            payload=finding_data.get('payload', ''),
            status=VerificationStatus.DETECTED,
            workflow_id=workflow_id
        )
        
        return finding
    
    async def _heuristic_detection_phase(self, finding: AssessmentResult) -> None:
        """Heuristic detection phase."""
        self.state_machine.transition_finding(
            finding,
            VerificationStatus.DETECTED,
            "Heuristic detection completed"
        )
    
    async def _ai_advisory_analysis_phase(self, finding: AssessmentResult) -> None:
        """
        AI advisory analysis phase
        Per VORTEX_AI_INTEGRATION.md: AI is ADVISORY ONLY
        """
        logger.info(f"Starting AI advisory analysis for {finding.id}")
        
        self.state_machine.transition_finding(
            finding,
            VerificationStatus.AI_ANALYSIS_PENDING,
            "Starting AI advisory analysis"
        )
        
        try:
            # AI advisory analysis (NOT authoritative)
            finding_data = finding.to_dict() if hasattr(finding, 'to_dict') else {}
            ai_result = await self.ai_engine.perform_advisory_analysis(finding_data, finding)
            finding.ai_analysis = ai_result
            
            # AI result determines next phase, not final verdict
            if ai_result.success and ai_result.verdict.value in ["CONFIRMED", "LIKELY"]:
                self.state_machine.transition_finding(
                    finding,
                    VerificationStatus.AI_CONFIRMED,
                    f"AI advisory: {ai_result.verdict.value}"
                )
            else:
                self.state_machine.transition_finding(
                    finding,
                    VerificationStatus.AI_FAILED,
                    f"AI advisory failed or negative: {ai_result.verdict.value if ai_result else 'unknown'}"
                )
            
            logger.debug(
                "AI advisory analysis complete",
                finding_id=str(finding.id),
                verdict=ai_result.verdict.value if ai_result else 'unknown',
                note="ADVISORY ONLY - NOT AUTHORITATIVE"
            )
            
        except Exception as e:
            logger.warning(f"AI advisory analysis failed for {finding.id}: {e}")
            finding.ai_analysis = None
            self.state_machine.transition_finding(
                finding,
                VerificationStatus.AI_FAILED,
                f"AI analysis error: {str(e)[:100]}"
            )
    
    async def _system_verification_phase(self, finding: AssessmentResult) -> None:
        """
        System verification phase
        Per VORTEX_CORE_AUTHORITY.md: THE authoritative evidence source
        """
        logger.info(f"Starting system verification for {finding.id}")
        
        self.state_machine.transition_finding(
            finding,
            VerificationStatus.SYSTEM_VERIFICATION_PENDING,
            "Starting authoritative system verification"
        )
        
        try:
            # Check if we should replay PoC
            should_replay = self.ai_engine.should_replay_poc(finding)
            
            if should_replay and finding.ai_analysis and finding.ai_analysis.poc:
                # System verification with PoC replay
                verification_result = await self._execute_poc_verification(finding)
            else:
                # System verification without PoC (pattern-based)
                verification_result = await self._execute_pattern_verification(finding)
            
            finding.verification_result = verification_result
            
            if verification_result and verification_result.success:
                self.state_machine.transition_finding(
                    finding,
                    VerificationStatus.SYSTEM_VERIFIED,
                    f"System verification successful: {verification_result.match_type.value if verification_result.match_type else 'unknown'}"
                )
                logger.info(
                    "System verification SUCCESS",
                    finding_id=str(finding.id),
                    confidence=verification_result.confidence
                )
            else:
                self.state_machine.transition_finding(
                    finding,
                    VerificationStatus.SYSTEM_VERIFICATION_FAILED,
                    "System verification failed"
                )
                logger.info(f"System verification FAILED for {finding.id}")
            
        except Exception as e:
            logger.error(f"System verification error for {finding.id}: {e}")
            finding.verification_result = None
            self.state_machine.transition_finding(
                finding,
                VerificationStatus.SYSTEM_VERIFICATION_FAILED,
                f"Verification error: {str(e)[:100]}"
            )
    
    async def _execute_poc_verification(self, finding: AssessmentResult) -> Optional[VerificationResult]:
        """
        Execute PoC verification using SystemVerificationEngine.
        
        CRITICAL: SystemVerificationEngine automatically determines:
        - If AI-generated PoC available → PoC replay
        - If not → Falls back to pattern-based verification
        """
        try:
            from core.verification import global_verification_engine
            
            logger.info(f"Executing PoC verification for {finding.id}")
            
            # SystemVerificationEngine handles PoC replay automatically
            # It checks _should_replay_poc() internally
            result = await global_verification_engine.verify_finding(finding)
            
            logger.info(
                f"PoC verification complete for {finding.id}",
                success=result.success if result else False,
                match_type=result.match_type if result else 'none'
            )
            
            return result
            
        except Exception as e:
            logger.error(f"PoC verification failed for {finding.id}: {e}")
            return None
    
    async def _execute_pattern_verification(self, finding: AssessmentResult) -> Optional[VerificationResult]:
        """
        Execute pattern-based verification using SystemVerificationEngine.
        
        CRITICAL: SystemVerificationEngine automatically uses:
        - Vulnerability-specific regex patterns
        - Behavioral differential analysis
        - Response pattern matching
        """
        try:
            from core.verification import global_verification_engine
            
            logger.info(f"Executing pattern verification for {finding.id}")
            
            # SystemVerificationEngine handles pattern verification automatically
            # When _should_replay_poc() returns False, it uses _verify_with_pattern_analysis()
            result = await global_verification_engine.verify_finding(finding)
            
            logger.info(
                f"Pattern verification complete for {finding.id}",
                success=result.success if result else False,
                confidence=result.confidence if result else 0.0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern verification failed for {finding.id}: {e}")
            return None
    
    async def _evidence_validation_phase(self, finding: AssessmentResult) -> None:
        """
        Evidence validation phase
        Per VORTEX_EVIDENCE_STANDARDS.md: Evidence must meet standards
        """
        logger.debug(f"Validating evidence standards for {finding.id}")
        
        try:
            # Calculate evidence determinism
            from core.evidence.determinism import global_determinism_scorer
            evidence_score = global_determinism_scorer.calculate_determinism_score(
                finding,
                include_breakdown=True
            )
            finding.evidence_determinism_score = evidence_score
            
            logger.debug(
                "Evidence validation complete",
                finding_id=str(finding.id),
                determinism_score=evidence_score
            )
            
        except Exception as e:
            logger.error(f"Evidence validation error for {finding.id}: {e}")
            finding.evidence_determinism_score = 0.0
    
    async def _final_determination_phase(self, finding: AssessmentResult) -> None:
        """
        Final determination phase
        Per VORTEX_CORE_AUTHORITY.md: Authority hierarchy enforcement
        """
        logger.info(f"Making final determination for {finding.id}")
        
        try:
            # Authority-compliant final determination
            final_status = self.authority_enforcer.make_final_determination(finding)
            
            # Validate status transition
            if self.state_machine.can_transition_to(finding, final_status):
                self.state_machine.transition_finding(
                    finding,
                    final_status,
                    "Final determination based on authority hierarchy"
                )
                logger.info(
                    "Final determination complete",
                    finding_id=str(finding.id),
                    final_status=final_status.value
                )
            else:
                logger.error(
                    "Invalid final determination transition",
                    finding_id=str(finding.id),
                    current=finding.status.value,
                    target=final_status.value
                )
                # Force to NEEDS_MANUAL for safety
                self.state_machine.transition_finding(
                    finding,
                    VerificationStatus.NEEDS_MANUAL,
                    "Invalid state transition detected - routing to manual"
                )
            
            # Authority compliance validation for SUBMIT_READY
            if final_status == VerificationStatus.SUBMIT_READY:
                authority_valid, reason = self.authority_enforcer.validate_submit_ready_authority(finding)
                if not authority_valid:
                    logger.error(
                        "AUTHORITY VIOLATION",
                        finding_id=str(finding.id),
                        reason=reason
                    )
                    self.state_machine.transition_finding(
                        finding,
                        VerificationStatus.NEEDS_MANUAL,
                        f"Authority validation failed: {reason}"
                    )
            
        except Exception as e:
            logger.error(f"Final determination error for {finding.id}: {e}")
            self.state_machine.transition_finding(
                finding,
                VerificationStatus.ERROR_STATE,
                f"Final determination error: {str(e)[:100]}"
            )
    
    async def _quality_assurance_phase(self, finding: AssessmentResult) -> None:
        """
        Quality assurance phase
        Per VORTEX_WORKFLOW_LIFECYCLE.md: Ensure submission readiness
        """
        if finding.status != VerificationStatus.SUBMIT_READY:
            return  # QA only for submit-ready findings
        
        logger.info(f"Quality assurance check for SUBMIT_READY finding {finding.id}")
        
        try:
            # Evidence quality check
            evidence_valid, reason = self.evidence_validator.validate_evidence_for_status(
                finding,
                VerificationStatus.SUBMIT_READY
            )
            if not evidence_valid:
                logger.warning(f"Evidence quality insufficient for {finding.id}: {reason}")
                self.state_machine.transition_finding(
                    finding,
                    VerificationStatus.NEEDS_MANUAL,
                    f"Evidence quality below SUBMIT_READY threshold: {reason}"
                )
                return
            
            # Authority compliance double-check
            authority_valid, auth_reason = self.authority_enforcer.validate_submit_ready_authority(finding)
            if not authority_valid:
                logger.error(f"Authority compliance failed for {finding.id}: {auth_reason}")
                self.state_machine.transition_finding(
                    finding,
                    VerificationStatus.NEEDS_MANUAL,
                    f"Authority compliance validation failed: {auth_reason}"
                )
                return
            
            logger.info(f"Quality assurance PASSED for {finding.id}")
            
        except Exception as e:
            logger.error(f"Quality assurance error for {finding.id}: {e}")
            self.state_machine.transition_finding(
                finding,
                VerificationStatus.NEEDS_MANUAL,
                f"QA error: {str(e)[:100]}"
            )
    
    async def _workflow_completion_phase(
        self,
        finding: AssessmentResult,
        workflow_id: str
    ) -> None:
        """
        Workflow completion phase
        Record completion and update metrics
        """
        logger.info(
            "Workflow completion",
            workflow_id=workflow_id,
            finding_id=str(finding.id),
            final_status=finding.status.value
        )
        
        # Record workflow completion
        self._record_workflow_completion(finding, workflow_id)
    
    async def _handle_workflow_error(
        self,
        finding_data: Dict,
        workflow_id: str,
        error: Exception
    ) -> AssessmentResult:
        """Handle workflow error."""
        logger.error(
            "Workflow error handler",
            workflow_id=workflow_id,
            error=str(error)
        )
        
        # Create error result
        finding = self._initialize_finding(finding_data, workflow_id)
        finding.status = VerificationStatus.ERROR_STATE
        finding.error_message = str(error)
        
        return finding
    
    def _record_workflow_completion(
        self,
        finding: AssessmentResult,
        workflow_id: str
    ) -> None:
        """Record workflow completion for audit."""
        record = {
            'timestamp': datetime.utcnow(),
            'workflow_id': workflow_id,
            'finding_id': str(finding.id),
            'final_status': finding.status.value,
            'workflow_duration_seconds': (
                datetime.utcnow() - self.active_workflows.get(workflow_id, {}).get('start_time', datetime.utcnow())
            ).total_seconds() if workflow_id in self.active_workflows else 0
        }
        
        self.workflow_history.append(record)
    
    def get_workflow_stats(self) -> Dict:
        """Get workflow statistics."""
        if not self.workflow_history:
            return {
                'total_workflows': 0,
                'avg_duration_seconds': 0.0,
                'status_distribution': {}
            }
        
        from collections import Counter
        
        total = len(self.workflow_history)
        avg_duration = sum(w['workflow_duration_seconds'] for w in self.workflow_history) / total
        
        statuses = [w['final_status'] for w in self.workflow_history]
        status_dist = dict(Counter(statuses))
        
        return {
            'total_workflows': total,
            'avg_duration_seconds': avg_duration,
            'status_distribution': status_dist,
            'active_workflows': len(self.active_workflows)
        }


async def process_finding_workflow(
    finding_data: Dict,
    finding: Optional[AssessmentResult] = None
) -> AssessmentResult:
    """
    Convenience function for workflow processing
    """
    orchestrator = WorkflowOrchestrator()
    return await orchestrator.process_finding_workflow(finding_data, finding)


# Global workflow orchestrator
global_workflow_orchestrator = WorkflowOrchestrator()