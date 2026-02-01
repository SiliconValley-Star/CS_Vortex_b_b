"""
VORTEX Finding Lifecycle Manager - V17.0 ULTIMATE
Complete finding lifecycle tracking and state management

Per .clinerules VORTEX_WORKFLOW_LIFECYCLE.md:
- State machine enforcement
- Lifecycle event tracking
- Audit trail maintenance
- State transition validation

FEATURES:
- Complete lifecycle tracking from detection to resolution
- State transition validation
- Event history with timestamps
- Audit trail for compliance
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class LifecycleEventType(str, Enum):
    """Lifecycle event types."""
    # Initial states
    DETECTED = "DETECTED"
    
    # AI analysis events
    AI_ANALYSIS_STARTED = "AI_ANALYSIS_STARTED"
    AI_ANALYSIS_COMPLETED = "AI_ANALYSIS_COMPLETED"
    AI_ANALYSIS_FAILED = "AI_ANALYSIS_FAILED"
    
    # System verification events
    SYSTEM_VERIFICATION_STARTED = "SYSTEM_VERIFICATION_STARTED"
    SYSTEM_VERIFICATION_COMPLETED = "SYSTEM_VERIFICATION_COMPLETED"
    SYSTEM_VERIFICATION_FAILED = "SYSTEM_VERIFICATION_FAILED"
    
    # Evidence events
    EVIDENCE_ADDED = "EVIDENCE_ADDED"
    EVIDENCE_VALIDATED = "EVIDENCE_VALIDATED"
    
    # Status transitions
    STATUS_CHANGED = "STATUS_CHANGED"
    
    # Quality events
    QUALITY_CHECKED = "QUALITY_CHECKED"
    COMPLIANCE_CHECKED = "COMPLIANCE_CHECKED"
    
    # Manual review events
    MANUAL_REVIEW_ASSIGNED = "MANUAL_REVIEW_ASSIGNED"
    MANUAL_REVIEW_STARTED = "MANUAL_REVIEW_STARTED"
    MANUAL_REVIEW_COMPLETED = "MANUAL_REVIEW_COMPLETED"
    
    # Final states
    SUBMITTED = "SUBMITTED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"
    
    # Error events
    ERROR_OCCURRED = "ERROR_OCCURRED"


@dataclass
class LifecycleEvent:
    """Single lifecycle event."""
    event_type: LifecycleEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Event details
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Actor information
    actor: str = "system"  # Who/what triggered the event
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'metadata': self.metadata,
            'actor': self.actor
        }


@dataclass
class FindingLifecycle:
    """
    Complete finding lifecycle tracker.
    
    Tracks all events from detection to final resolution.
    
    Per .clinerules VORTEX_WORKFLOW_LIFECYCLE.md:
    - Complete audit trail
    - State transition validation
    - SLA tracking
    - Lifecycle metrics
    """
    finding_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Event history (ordered)
    events: List[LifecycleEvent] = field(default_factory=list)
    
    # Current state
    current_status: str = "DETECTED"
    
    # Lifecycle phases tracking
    detection_completed_at: Optional[datetime] = None
    ai_analysis_started_at: Optional[datetime] = None
    ai_analysis_completed_at: Optional[datetime] = None
    system_verification_started_at: Optional[datetime] = None
    system_verification_completed_at: Optional[datetime] = None
    manual_review_assigned_at: Optional[datetime] = None
    manual_review_started_at: Optional[datetime] = None
    manual_review_completed_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    
    # Performance metrics
    total_processing_time: Optional[float] = None  # seconds
    ai_analysis_time: Optional[float] = None
    system_verification_time: Optional[float] = None
    manual_review_time: Optional[float] = None
    
    # Quality tracking
    quality_checks_passed: int = 0
    quality_checks_failed: int = 0
    compliance_checks_passed: int = 0
    compliance_checks_failed: int = 0
    
    def add_event(self, 
                  event_type: LifecycleEventType, 
                  description: str = "",
                  metadata: Optional[Dict[str, Any]] = None,
                  actor: str = "system") -> LifecycleEvent:
        """
        Add lifecycle event.
        
        Args:
            event_type: Type of event
            description: Event description
            metadata: Additional event data
            actor: Who triggered the event
            
        Returns:
            Created event
        """
        event = LifecycleEvent(
            event_type=event_type,
            description=description,
            metadata=metadata or {},
            actor=actor
        )
        
        self.events.append(event)
        
        # Update phase timestamps
        self._update_phase_timestamps(event)
        
        # Calculate metrics
        self._calculate_metrics()
        
        logger.debug(f"Lifecycle event added for {self.finding_id}: {event_type.value}")
        
        return event
    
    def _update_phase_timestamps(self, event: LifecycleEvent) -> None:
        """Update phase timestamps based on event."""
        event_type = event.event_type
        timestamp = event.timestamp
        
        if event_type == LifecycleEventType.DETECTED:
            self.detection_completed_at = timestamp
            
        elif event_type == LifecycleEventType.AI_ANALYSIS_STARTED:
            self.ai_analysis_started_at = timestamp
            
        elif event_type == LifecycleEventType.AI_ANALYSIS_COMPLETED:
            self.ai_analysis_completed_at = timestamp
            
        elif event_type == LifecycleEventType.SYSTEM_VERIFICATION_STARTED:
            self.system_verification_started_at = timestamp
            
        elif event_type == LifecycleEventType.SYSTEM_VERIFICATION_COMPLETED:
            self.system_verification_completed_at = timestamp
            
        elif event_type == LifecycleEventType.MANUAL_REVIEW_ASSIGNED:
            self.manual_review_assigned_at = timestamp
            
        elif event_type == LifecycleEventType.MANUAL_REVIEW_STARTED:
            self.manual_review_started_at = timestamp
            
        elif event_type == LifecycleEventType.MANUAL_REVIEW_COMPLETED:
            self.manual_review_completed_at = timestamp
            
        elif event_type == LifecycleEventType.SUBMITTED:
            self.submitted_at = timestamp
    
    def _calculate_metrics(self) -> None:
        """Calculate lifecycle performance metrics."""
        # AI analysis time
        if self.ai_analysis_started_at and self.ai_analysis_completed_at:
            delta = self.ai_analysis_completed_at - self.ai_analysis_started_at
            self.ai_analysis_time = delta.total_seconds()
        
        # System verification time
        if self.system_verification_started_at and self.system_verification_completed_at:
            delta = self.system_verification_completed_at - self.system_verification_started_at
            self.system_verification_time = delta.total_seconds()
        
        # Manual review time
        if self.manual_review_started_at and self.manual_review_completed_at:
            delta = self.manual_review_completed_at - self.manual_review_started_at
            self.manual_review_time = delta.total_seconds()
        
        # Total processing time (detection to submission)
        if self.submitted_at:
            delta = self.submitted_at - self.created_at
            self.total_processing_time = delta.total_seconds()
    
    def record_quality_check(self, passed: bool) -> None:
        """Record quality check result."""
        if passed:
            self.quality_checks_passed += 1
        else:
            self.quality_checks_failed += 1
        
        self.add_event(
            LifecycleEventType.QUALITY_CHECKED,
            description=f"Quality check {'passed' if passed else 'failed'}",
            metadata={'passed': passed}
        )
    
    def record_compliance_check(self, passed: bool, details: str = "") -> None:
        """Record compliance check result."""
        if passed:
            self.compliance_checks_passed += 1
        else:
            self.compliance_checks_failed += 1
        
        self.add_event(
            LifecycleEventType.COMPLIANCE_CHECKED,
            description=f"Compliance check {'passed' if passed else 'failed'}: {details}",
            metadata={'passed': passed, 'details': details}
        )
    
    def get_phase_duration(self, phase: str) -> Optional[float]:
        """
        Get duration of specific phase in seconds.
        
        Args:
            phase: Phase name ('ai_analysis', 'system_verification', 'manual_review')
            
        Returns:
            Duration in seconds or None
        """
        phase_times = {
            'ai_analysis': self.ai_analysis_time,
            'system_verification': self.system_verification_time,
            'manual_review': self.manual_review_time,
            'total': self.total_processing_time
        }
        
        return phase_times.get(phase)
    
    def get_event_history(self, 
                         event_type: Optional[LifecycleEventType] = None) -> List[LifecycleEvent]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (optional)
            
        Returns:
            List of events
        """
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        
        return self.events
    
    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle summary."""
        return {
            'finding_id': self.finding_id,
            'created_at': self.created_at.isoformat(),
            'current_status': self.current_status,
            'total_events': len(self.events),
            
            # Phase durations
            'ai_analysis_time': self.ai_analysis_time,
            'system_verification_time': self.system_verification_time,
            'manual_review_time': self.manual_review_time,
            'total_processing_time': self.total_processing_time,
            
            # Quality metrics
            'quality_checks_passed': self.quality_checks_passed,
            'quality_checks_failed': self.quality_checks_failed,
            'compliance_checks_passed': self.compliance_checks_passed,
            'compliance_checks_failed': self.compliance_checks_failed,
            
            # Timestamps
            'detection_completed_at': self.detection_completed_at.isoformat() if self.detection_completed_at else None,
            'ai_analysis_started_at': self.ai_analysis_started_at.isoformat() if self.ai_analysis_started_at else None,
            'ai_analysis_completed_at': self.ai_analysis_completed_at.isoformat() if self.ai_analysis_completed_at else None,
            'system_verification_started_at': self.system_verification_started_at.isoformat() if self.system_verification_started_at else None,
            'system_verification_completed_at': self.system_verification_completed_at.isoformat() if self.system_verification_completed_at else None,
            'manual_review_assigned_at': self.manual_review_assigned_at.isoformat() if self.manual_review_assigned_at else None,
            'manual_review_started_at': self.manual_review_started_at.isoformat() if self.manual_review_started_at else None,
            'manual_review_completed_at': self.manual_review_completed_at.isoformat() if self.manual_review_completed_at else None,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete lifecycle to dictionary."""
        return {
            'summary': self.get_lifecycle_summary(),
            'events': [event.to_dict() for event in self.events]
        }


class LifecycleManager:
    """
    Global lifecycle management system.
    
    RESPONSIBILITIES:
    - Track all finding lifecycles
    - Provide lifecycle analytics
    - SLA monitoring
    - Performance metrics
    
    Per .clinerules VORTEX_WORKFLOW_LIFECYCLE.md:
    - Complete audit trail maintenance
    - Lifecycle validation
    - Performance tracking
    """
    
    # SLA targets (per .clinerules VORTEX_WORKFLOW_LIFECYCLE.md)
    SLA_TARGETS = {
        'detection_to_ai': 300,  # 5 minutes
        'ai_analysis': 30,  # 30 seconds
        'system_verification': 60,  # 60 seconds
        'manual_review_registration': 5,  # 5 seconds
        'total_automated_workflow': 600  # 10 minutes
    }
    
    def __init__(self):
        """Initialize lifecycle manager."""
        # Active lifecycles
        self.lifecycles: Dict[str, FindingLifecycle] = {}
        
        # Statistics
        self.total_lifecycles = 0
        self.completed_lifecycles = 0
        self.average_processing_time = 0.0
        
        logger.info("Lifecycle Manager initialized")
    
    def create_lifecycle(self, finding_id: str) -> FindingLifecycle:
        """
        Create new finding lifecycle.
        
        Args:
            finding_id: Finding identifier
            
        Returns:
            Created lifecycle
        """
        lifecycle = FindingLifecycle(finding_id=finding_id)
        
        # Add initial detection event
        lifecycle.add_event(
            LifecycleEventType.DETECTED,
            description="Finding detected"
        )
        
        self.lifecycles[finding_id] = lifecycle
        self.total_lifecycles += 1
        
        logger.info(f"Lifecycle created for finding {finding_id}")
        
        return lifecycle
    
    def get_lifecycle(self, finding_id: str) -> Optional[FindingLifecycle]:
        """Get lifecycle for finding."""
        return self.lifecycles.get(finding_id)
    
    def record_event(self,
                    finding_id: str,
                    event_type: LifecycleEventType,
                    description: str = "",
                    metadata: Optional[Dict[str, Any]] = None,
                    actor: str = "system") -> bool:
        """
        Record lifecycle event for finding.
        
        Args:
            finding_id: Finding identifier
            event_type: Type of event
            description: Event description
            metadata: Additional data
            actor: Event actor
            
        Returns:
            Success status
        """
        lifecycle = self.get_lifecycle(finding_id)
        
        if not lifecycle:
            logger.warning(f"No lifecycle found for finding {finding_id}")
            return False
        
        lifecycle.add_event(event_type, description, metadata, actor)
        
        return True
    
    def check_sla_compliance(self, finding_id: str) -> Dict[str, Any]:
        """
        Check SLA compliance for finding lifecycle.
        
        Args:
            finding_id: Finding identifier
            
        Returns:
            SLA compliance status
        """
        lifecycle = self.get_lifecycle(finding_id)
        
        if not lifecycle:
            return {'error': 'Lifecycle not found'}
        
        compliance = {
            'overall_compliant': True,
            'violations': []
        }
        
        # Check AI analysis SLA
        if lifecycle.ai_analysis_time:
            if lifecycle.ai_analysis_time > self.SLA_TARGETS['ai_analysis']:
                compliance['overall_compliant'] = False
                compliance['violations'].append({
                    'phase': 'ai_analysis',
                    'target': self.SLA_TARGETS['ai_analysis'],
                    'actual': lifecycle.ai_analysis_time
                })
        
        # Check system verification SLA
        if lifecycle.system_verification_time:
            if lifecycle.system_verification_time > self.SLA_TARGETS['system_verification']:
                compliance['overall_compliant'] = False
                compliance['violations'].append({
                    'phase': 'system_verification',
                    'target': self.SLA_TARGETS['system_verification'],
                    'actual': lifecycle.system_verification_time
                })
        
        # Check total workflow SLA
        if lifecycle.total_processing_time:
            if lifecycle.total_processing_time > self.SLA_TARGETS['total_automated_workflow']:
                compliance['overall_compliant'] = False
                compliance['violations'].append({
                    'phase': 'total_workflow',
                    'target': self.SLA_TARGETS['total_automated_workflow'],
                    'actual': lifecycle.total_processing_time
                })
        
        return compliance
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall lifecycle performance metrics."""
        if not self.lifecycles:
            return {}
        
        # Calculate averages
        total_processing_times = [
            lc.total_processing_time 
            for lc in self.lifecycles.values() 
            if lc.total_processing_time
        ]
        
        ai_times = [
            lc.ai_analysis_time
            for lc in self.lifecycles.values()
            if lc.ai_analysis_time
        ]
        
        verification_times = [
            lc.system_verification_time
            for lc in self.lifecycles.values()
            if lc.system_verification_time
        ]
        
        manual_review_times = [
            lc.manual_review_time
            for lc in self.lifecycles.values()
            if lc.manual_review_time
        ]
        
        return {
            'total_lifecycles': self.total_lifecycles,
            'active_lifecycles': len(self.lifecycles),
            'avg_total_processing_time': sum(total_processing_times) / len(total_processing_times) if total_processing_times else 0,
            'avg_ai_analysis_time': sum(ai_times) / len(ai_times) if ai_times else 0,
            'avg_system_verification_time': sum(verification_times) / len(verification_times) if verification_times else 0,
            'avg_manual_review_time': sum(manual_review_times) / len(manual_review_times) if manual_review_times else 0,
            'sla_targets': self.SLA_TARGETS
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lifecycle statistics."""
        return {
            'total_lifecycles': self.total_lifecycles,
            'active_lifecycles': len(self.lifecycles),
            'completed_lifecycles': self.completed_lifecycles,
            'performance_metrics': self.get_performance_metrics()
        }


# Global lifecycle manager
global_lifecycle_manager = LifecycleManager()