"""
VORTEX State Machine - V17.0 ULTIMATE
State machine per VORTEX_WORKFLOW_LIFECYCLE.md

CRITICAL: State transitions must follow these rules exactly
"""

import structlog
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from domain.enums import VerificationStatus
from domain.models import AssessmentResult

logger = structlog.get_logger()


class StateMachine:
    """
    State machine for finding lifecycle
    Per VORTEX_WORKFLOW_LIFECYCLE.md: Enforces valid transitions
    """
    
    def __init__(self):
        # Define valid state transitions per VORTEX_WORKFLOW_LIFECYCLE.md
        self.valid_transitions = {
            VerificationStatus.DETECTED: [
                VerificationStatus.AI_ANALYSIS_PENDING,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE,
                VerificationStatus.ERROR_STATE
            ],
            VerificationStatus.AI_ANALYSIS_PENDING: [
                VerificationStatus.AI_CONFIRMED,
                VerificationStatus.AI_FAILED,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.ERROR_STATE
            ],
            VerificationStatus.AI_CONFIRMED: [
                VerificationStatus.SYSTEM_VERIFICATION_PENDING,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE
            ],
            VerificationStatus.AI_FAILED: [
                VerificationStatus.SYSTEM_VERIFICATION_PENDING,
                VerificationStatus.SUBMIT_READY,  # V11.1: If strong system evidence
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE
            ],
            VerificationStatus.SYSTEM_VERIFICATION_PENDING: [
                VerificationStatus.SYSTEM_VERIFIED,
                VerificationStatus.SYSTEM_VERIFICATION_FAILED,
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.ERROR_STATE
            ],
            VerificationStatus.SYSTEM_VERIFIED: [
                VerificationStatus.SUBMIT_READY,  # V11.1: Fastpath enabled
                VerificationStatus.NEEDS_MANUAL
            ],
            VerificationStatus.SYSTEM_VERIFICATION_FAILED: [
                VerificationStatus.NEEDS_MANUAL,
                VerificationStatus.FALSE_POSITIVE
            ],
            # Terminal states have no valid transitions
            VerificationStatus.SUBMIT_READY: [],
            VerificationStatus.NEEDS_MANUAL: [],
            VerificationStatus.FALSE_POSITIVE: [],
            VerificationStatus.ERROR_STATE: [VerificationStatus.NEEDS_MANUAL]  # Can recover to manual
        }
        
        # Terminal states
        self.terminal_states = [
            VerificationStatus.SUBMIT_READY,
            VerificationStatus.NEEDS_MANUAL,
            VerificationStatus.FALSE_POSITIVE
        ]
        
        # Track transitions for audit
        self.transition_history = []
    
    def validate_transition(
        self,
        from_status: VerificationStatus,
        to_status: VerificationStatus
    ) -> Tuple[bool, str]:
        """
        Validate if state transition is allowed
        
        Returns: (is_valid, reason)
        """
        # Get allowed transitions for current state
        allowed = self.valid_transitions.get(from_status, [])
        
        if to_status not in allowed:
            reason = (
                f"Invalid state transition: {from_status.value} → {to_status.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )
            logger.error(
                "Invalid state transition",
                from_status=from_status.value,
                to_status=to_status.value,
                allowed=[s.value for s in allowed]
            )
            return False, reason
        
        logger.debug(
            "Valid state transition",
            from_status=from_status.value,
            to_status=to_status.value
        )
        return True, "Valid transition"
    
    def transition_finding(
        self,
        finding: AssessmentResult,
        new_status: VerificationStatus,
        reason: str
    ) -> bool:
        """
        Perform state transition on finding
        
        Returns: True if successful
        """
        current_status = finding.status
        
        # Validate transition
        is_valid, validation_reason = self.validate_transition(current_status, new_status)
        if not is_valid:
            logger.error(
                "Transition validation failed",
                finding_id=str(finding.id),
                reason=validation_reason
            )
            return False
        
        # Perform transition
        finding.status = new_status
        finding.status_updated_at = datetime.utcnow()
        
        # Add to history
        if not hasattr(finding, 'status_history'):
            finding.status_history = []
        
        finding.status_history.append({
            'timestamp': datetime.utcnow(),
            'from_status': current_status.value,
            'to_status': new_status.value,
            'reason': reason
        })
        
        # Record transition
        self._record_transition(finding, current_status, new_status, reason)
        
        logger.info(
            "State transition completed",
            finding_id=str(finding.id),
            from_status=current_status.value,
            to_status=new_status.value,
            reason=reason
        )
        
        return True
    
    def get_valid_transitions(self, status: VerificationStatus) -> List[VerificationStatus]:
        """Get list of valid transitions from current status."""
        return self.valid_transitions.get(status, [])
    
    def is_terminal_state(self, status: VerificationStatus) -> bool:
        """Check if status is a terminal state."""
        return status in self.terminal_states
    
    def can_transition_to(
        self,
        finding: AssessmentResult,
        target_status: VerificationStatus
    ) -> bool:
        """Check if finding can transition to target status."""
        valid, _ = self.validate_transition(finding.status, target_status)
        return valid
    
    def get_transition_path(
        self,
        from_status: VerificationStatus,
        to_status: VerificationStatus
    ) -> Optional[List[VerificationStatus]]:
        """
        Find a valid path between two states
        Returns None if no path exists
        """
        if from_status == to_status:
            return [from_status]
        
        # BFS to find path
        from collections import deque
        
        queue = deque([(from_status, [from_status])])
        visited = {from_status}
        
        while queue:
            current, path = queue.popleft()
            
            # Get valid next states
            next_states = self.valid_transitions.get(current, [])
            
            for next_state in next_states:
                if next_state == to_status:
                    return path + [next_state]
                
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [next_state]))
        
        # No path found
        return None
    
    def _record_transition(
        self,
        finding: AssessmentResult,
        from_status: VerificationStatus,
        to_status: VerificationStatus,
        reason: str
    ) -> None:
        """Record transition for audit."""
        record = {
            'timestamp': datetime.utcnow(),
            'finding_id': str(finding.id),
            'from_status': from_status.value,
            'to_status': to_status.value,
            'reason': reason
        }
        
        self.transition_history.append(record)
    
    def get_transition_stats(self) -> Dict:
        """Get transition statistics."""
        if not self.transition_history:
            return {
                'total_transitions': 0,
                'transitions_by_type': {},
                'most_common_transitions': []
            }
        
        from collections import Counter
        
        total = len(self.transition_history)
        
        # Count transitions by type
        transitions = [
            f"{t['from_status']}→{t['to_status']}"
            for t in self.transition_history
        ]
        transition_counts = Counter(transitions)
        
        return {
            'total_transitions': total,
            'transitions_by_type': dict(transition_counts),
            'most_common_transitions': transition_counts.most_common(10)
        }


def validate_state_transition(
    from_status: VerificationStatus,
    to_status: VerificationStatus
) -> Tuple[bool, str]:
    """
    Convenience function for state validation
    """
    sm = StateMachine()
    return sm.validate_transition(from_status, to_status)


def get_valid_transitions(status: VerificationStatus) -> List[VerificationStatus]:
    """
    Convenience function to get valid transitions
    """
    sm = StateMachine()
    return sm.get_valid_transitions(status)


# Global state machine instance
global_state_machine = StateMachine()