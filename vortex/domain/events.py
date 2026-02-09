"""
VORTEX Domain Events - V17.0 ULTIMATE
Event system for workflow state transitions and system monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from .enums import VerificationStatus, HealthStatus, AuthorityLevel


@dataclass
class DomainEvent:
    """Base class for all domain events."""
    event_id: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow, kw_only=True)
    metadata: Dict[str, Any] = field(default_factory=dict, kw_only=True)


@dataclass
class FindingDetectedEvent(DomainEvent):
    """Fired when a new finding is detected."""
    finding_id: UUID
    url: str
    finding_type: str
    heuristic_score: float
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "finding_detected"


@dataclass
class StateTransitionEvent(DomainEvent):
    """
    Fired when finding state transitions
    Per VORTEX_WORKFLOW_LIFECYCLE.md: All transitions must be audited
    """
    finding_id: UUID
    from_status: VerificationStatus
    to_status: VerificationStatus
    reason: str
    authority_level: AuthorityLevel
    
    # Validation flags
    is_valid_transition: bool = True
    authority_compliant: bool = True
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "state_transition"


@dataclass
class AIAnalysisCompleteEvent(DomainEvent):
    """
    Fired when AI analysis completes
    Per VORTEX_AI_INTEGRATION.md: AI is advisory only
    """
    finding_id: UUID
    model_used: str
    verdict: str
    confidence: float
    is_authoritative: bool = False  # ALWAYS False
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "ai_analysis_complete"


@dataclass
class SystemVerificationCompleteEvent(DomainEvent):
    """
    Fired when system verification completes
    Per VORTEX_CORE_AUTHORITY.md: System verification is Level 1 authority
    """
    finding_id: UUID
    success: bool
    confidence: float
    match_type: str
    is_authoritative: bool = True  # System verification IS authoritative
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "system_verification_complete"


@dataclass
class AuthorityViolationEvent(DomainEvent):
    """
    Fired when authority hierarchy is violated
    Per VORTEX_CORE_AUTHORITY.md: Violations must be logged and prevented
    """
    finding_id: UUID
    violation_type: str
    violation_description: str
    attempted_status: VerificationStatus
    blocked: bool = True
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "authority_violation"


@dataclass
class FastPathPromotionEvent(DomainEvent):
    """
    Fired when finding is promoted via fastpath
    Per VORTEX_FASTPATH_V11.md: Track fastpath promotions
    """
    finding_id: UUID
    fastpath_score: float
    qualifying_factors: list
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "fastpath_promotion"


@dataclass
class ManualReviewRegisteredEvent(DomainEvent):
    """
    Fired when finding is registered for manual review
    Per VORTEX_WORKFLOW_LIFECYCLE.md: NEEDS_MANUAL is active status
    """
    finding_id: UUID
    priority_level: int
    max_age_hours: int
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "manual_review_registered"


@dataclass
class ManualReviewOverdueEvent(DomainEvent):
    """Fired when manual review becomes overdue."""
    finding_id: UUID
    age_hours: float
    max_age_hours: int
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "manual_review_overdue"


@dataclass
class HealthThresholdViolationEvent(DomainEvent):
    """
    Fired when health threshold is violated
    Per VORTEX_OPERATIONAL_HEALTH.md: Monitor and alert on violations
    """
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # WARNING, CRITICAL
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "health_threshold_violation"


@dataclass
class AutoTuningExecutedEvent(DomainEvent):
    """Fired when auto-tuning is executed."""
    tuning_category: str
    actions_taken: list
    estimated_impact: str
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "auto_tuning_executed"


@dataclass
class ComplianceViolationEvent(DomainEvent):
    """Fired when legal compliance violation is detected."""
    violation_type: str
    severity: str
    finding_id: Optional[UUID] = None
    url: Optional[str] = None
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "compliance_violation"


@dataclass
class EvidenceIntegrityEvent(DomainEvent):
    """Fired for evidence integrity operations."""
    finding_id: UUID
    operation: str  # stored, verified, tampered
    integrity_valid: bool
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "evidence_integrity"


@dataclass
class WorkflowErrorEvent(DomainEvent):
    """Fired when workflow error occurs."""
    finding_id: UUID
    phase: str
    error_message: str
    recoverable: bool
    
    def __post_init__(self):
        if not hasattr(self, 'event_type'):
            self.event_type = "workflow_error"


class EventBus:
    """
    Simple event bus for domain events
    Allows decoupled components to react to domain events
    """
    
    def __init__(self):
        self._handlers: Dict[str, list] = {}
    
    def subscribe(self, event_type: str, handler):
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def publish(self, event: DomainEvent):
        """Publish an event to all subscribers."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log but don't fail on handler errors
                print(f"Event handler error: {e}")
    
    def clear(self):
        """Clear all event handlers."""
        self._handlers.clear()


# Global event bus instance
event_bus = EventBus()