"""
VORTEX Lifecycle Domain - V17.0 ULTIMATE
Finding lifecycle and state machine management per VORTEX_WORKFLOW_LIFECYCLE.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID

from .enums import VerificationStatus, ManualReviewPriority
from .models import AssessmentResult, ManualReviewSLA


# Valid state transitions per VORTEX_WORKFLOW_LIFECYCLE.md
VALID_STATE_TRANSITIONS: Dict[VerificationStatus, List[VerificationStatus]] = {
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
    VerificationStatus.NEEDS_MANUAL: [],  # Terminal but active
    VerificationStatus.FALSE_POSITIVE: [],
    VerificationStatus.ERROR_STATE: [VerificationStatus.NEEDS_MANUAL]  # Can recover
}


def is_valid_transition(
    current_status: VerificationStatus,
    new_status: VerificationStatus
) -> bool:
    """
    Check if state transition is valid
    Per VORTEX_WORKFLOW_LIFECYCLE.md: All transitions must be validated
    """
    allowed_transitions = VALID_STATE_TRANSITIONS.get(current_status, [])
    return new_status in allowed_transitions


def get_valid_transitions(current_status: VerificationStatus) -> List[VerificationStatus]:
    """Get list of valid transitions from current status."""
    return VALID_STATE_TRANSITIONS.get(current_status, [])


def is_terminal_state(status: VerificationStatus) -> bool:
    """Check if status is a terminal state (no further transitions)."""
    return len(VALID_STATE_TRANSITIONS.get(status, [])) == 0


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_status: VerificationStatus
    to_status: VerificationStatus
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Validation
    is_valid: bool = True
    validation_error: Optional[str] = None
    
    # Context
    triggered_by: str = "system"  # system, manual, ai, error
    metadata: Dict = field(default_factory=dict)


@dataclass
class FindingLifecycle:
    """
    Complete lifecycle tracking for a finding
    Per VORTEX_WORKFLOW_LIFECYCLE.md requirements
    """
    finding_id: UUID
    current_status: VerificationStatus = VerificationStatus.DETECTED
    
    # History
    transitions: List[StateTransition] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_transition_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Phase durations (in seconds)
    detection_duration: float = 0.0
    ai_analysis_duration: float = 0.0
    system_verification_duration: float = 0.0
    total_duration: float = 0.0
    
    # Status tracking
    is_terminal: bool = False
    terminal_reason: Optional[str] = None
    
    def transition(
        self,
        new_status: VerificationStatus,
        reason: str,
        triggered_by: str = "system"
    ) -> StateTransition:
        """
        Transition to new status with validation
        Returns the StateTransition record
        """
        transition = StateTransition(
            from_status=self.current_status,
            to_status=new_status,
            reason=reason,
            triggered_by=triggered_by
        )
        
        # Validate transition
        if not is_valid_transition(self.current_status, new_status):
            transition.is_valid = False
            transition.validation_error = (
                f"Invalid transition: {self.current_status} -> {new_status}"
            )
            return transition
        
        # Record transition
        self.transitions.append(transition)
        self.current_status = new_status
        self.last_transition_at = datetime.utcnow()
        
        # Check if terminal
        if is_terminal_state(new_status):
            self.is_terminal = True
            self.terminal_reason = reason
            self.completed_at = datetime.utcnow()
            self.total_duration = (
                self.completed_at - self.created_at
            ).total_seconds()
        
        return transition
    
    def get_time_in_status(self, status: VerificationStatus) -> float:
        """Get total time spent in a specific status (in seconds)."""
        total_time = 0.0
        current_start = None
        
        for transition in self.transitions:
            if transition.to_status == status:
                current_start = transition.timestamp
            elif current_start and transition.from_status == status:
                duration = (transition.timestamp - current_start).total_seconds()
                total_time += duration
                current_start = None
        
        # If still in the status
        if current_start and self.current_status == status:
            duration = (datetime.utcnow() - current_start).total_seconds()
            total_time += duration
        
        return total_time
    
    def get_transition_count(self) -> int:
        """Get total number of transitions."""
        return len(self.transitions)
    
    def get_phase_summary(self) -> Dict[str, float]:
        """Get summary of time spent in each phase."""
        return {
            'detection': self.get_time_in_status(VerificationStatus.DETECTED),
            'ai_analysis': (
                self.get_time_in_status(VerificationStatus.AI_ANALYSIS_PENDING) +
                self.get_time_in_status(VerificationStatus.AI_CONFIRMED)
            ),
            'system_verification': (
                self.get_time_in_status(VerificationStatus.SYSTEM_VERIFICATION_PENDING) +
                self.get_time_in_status(VerificationStatus.SYSTEM_VERIFIED)
            ),
            'manual_review': self.get_time_in_status(VerificationStatus.NEEDS_MANUAL),
            'total': self.total_duration if self.is_terminal else (
                datetime.utcnow() - self.created_at
            ).total_seconds()
        }


@dataclass
class ManualReviewQueue:
    """
    Manual review queue management
    Per VORTEX_WORKFLOW_LIFECYCLE.md: NEEDS_MANUAL is ACTIVE status
    """
    queue: Dict[str, ManualReviewSLA] = field(default_factory=dict)
    
    # Queue stats
    total_in_queue: int = 0
    overdue_count: int = 0
    escalated_count: int = 0
    
    # Priority distribution
    priority_distribution: Dict[ManualReviewPriority, int] = field(default_factory=dict)
    
    def add(self, sla: ManualReviewSLA) -> None:
        """Add finding to manual review queue."""
        self.queue[sla.finding_id] = sla
        self.total_in_queue = len(self.queue)
        
        # Update priority distribution
        self.priority_distribution[sla.priority_level] = (
            self.priority_distribution.get(sla.priority_level, 0) + 1
        )
    
    def remove(self, finding_id: str) -> Optional[ManualReviewSLA]:
        """Remove finding from queue (completed or rejected)."""
        sla = self.queue.pop(finding_id, None)
        if sla:
            self.total_in_queue = len(self.queue)
            self.priority_distribution[sla.priority_level] = (
                self.priority_distribution.get(sla.priority_level, 1) - 1
            )
        return sla
    
    def get(self, finding_id: str) -> Optional[ManualReviewSLA]:
        """Get SLA for finding."""
        return self.queue.get(finding_id)
    
    def get_overdue(self) -> List[ManualReviewSLA]:
        """Get all overdue items."""
        return [sla for sla in self.queue.values() if sla.check_overdue()]
    
    def get_escalated(self) -> List[ManualReviewSLA]:
        """Get all escalated items."""
        return [sla for sla in self.queue.values() if sla.is_escalated]
    
    def get_by_priority(self, priority: ManualReviewPriority) -> List[ManualReviewSLA]:
        """Get all items of specific priority."""
        return [sla for sla in self.queue.values() if sla.priority_level == priority]
    
    def update_stats(self) -> None:
        """Update queue statistics."""
        self.overdue_count = len(self.get_overdue())
        self.escalated_count = len(self.get_escalated())
        self.total_in_queue = len(self.queue)
    
    def get_queue_status(self) -> Dict:
        """Get comprehensive queue status."""
        self.update_stats()
        
        ages = [sla.get_age_hours() for sla in self.queue.values()]
        avg_age = sum(ages) / len(ages) if ages else 0.0
        
        return {
            'queue_size': self.total_in_queue,
            'overdue_count': self.overdue_count,
            'escalated_count': self.escalated_count,
            'average_age_hours': avg_age,
            'priority_distribution': {
                priority.name: count
                for priority, count in self.priority_distribution.items()
            },
            'high_priority_count': sum(
                count for priority, count in self.priority_distribution.items()
                if priority.value <= 2
            )
        }


def calculate_manual_review_priority(finding: AssessmentResult) -> ManualReviewPriority:
    """
    Calculate manual review priority
    Per VORTEX_WORKFLOW_LIFECYCLE.md priority calculation
    """
    priority_score = 3  # Default: MEDIUM
    
    # Severity adjustment
    severity_map = {
        'CRITICAL': -2,
        'HIGH': -1,
        'MEDIUM': 0,
        'LOW': +1,
        'INFO': +2
    }
    priority_score += severity_map.get(finding.severity.value, 0)
    
    # Confidence adjustment
    if finding.heuristic_score >= 0.8:
        priority_score -= 1
    
    # System verification attempt
    if finding.verification_result:
        priority_score -= 1
    
    # AI impact suggestion
    if finding.ai_analysis and finding.ai_analysis.impact in ["HIGH", "CRITICAL"]:
        priority_score -= 1
    
    # Clamp to valid range
    priority_score = max(1, min(5, priority_score))
    
    return ManualReviewPriority(priority_score)


def calculate_sla_hours(priority: ManualReviewPriority) -> tuple:
    """
    Calculate SLA hours for priority level
    Returns (max_age_hours, escalation_threshold_hours)
    """
    sla_map = {
        ManualReviewPriority.CRITICAL: (24, 12),
        ManualReviewPriority.HIGH: (48, 24),
        ManualReviewPriority.MEDIUM: (72, 48),
        ManualReviewPriority.LOW: (168, 96),  # 1 week
        ManualReviewPriority.BACKLOG: (720, 480)  # 30 days
    }
    return sla_map.get(priority, (72, 48))