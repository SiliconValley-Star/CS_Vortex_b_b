"""
VORTEX Health Alert System - V17.0 ULTIMATE
System health alerting with escalation levels

Per .clinerules VORTEX_OPERATIONAL_HEALTH.md:
- Multi-level alerting (INFO, WARNING, CRITICAL)
- Automated action recommendations
- SLA-based response requirements
- Alert history and tracking

FEATURES:
- Severity-based alert levels
- Actionable recommendations
- Response time tracking
- Alert acknowledgment
- Historical analysis
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthAlertLevel(str, Enum):
    """Health alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class HealthAlert:
    """
    System health alert with response requirements.
    
    Per .clinerules alert levels:
    - INFO: Informational, 24h response time
    - WARNING: Attention needed, 4h response time
    - CRITICAL: Immediate action required, 1h response time
    """
    level: HealthAlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Action requirements
    requires_immediate_response: bool = False
    suggested_actions: List[str] = field(default_factory=list)
    
    # Tracking
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Response SLA
    response_deadline: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate response deadline based on level."""
        if self.response_deadline is None:
            # Per .clinerules response times
            response_hours = {
                HealthAlertLevel.INFO: 24,
                HealthAlertLevel.WARNING: 4,
                HealthAlertLevel.CRITICAL: 1
            }
            
            hours = response_hours.get(self.level, 24)
            self.response_deadline = self.timestamp + timedelta(hours=hours)
    
    def acknowledge(self, acknowledged_by: str = "system") -> None:
        """Acknowledge alert."""
        self.acknowledged = True
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = acknowledged_by
        
        logger.info(f"Alert acknowledged: {self.message} by {acknowledged_by}")
    
    def resolve(self, resolution_notes: str = "") -> None:
        """Resolve alert."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolution_notes = resolution_notes
        
        logger.info(f"Alert resolved: {self.message}")
    
    def is_overdue(self) -> bool:
        """Check if alert response is overdue."""
        if self.acknowledged or self.resolved:
            return False
        
        return datetime.utcnow() > self.response_deadline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'requires_immediate_response': self.requires_immediate_response,
            'suggested_actions': self.suggested_actions,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes,
            'response_deadline': self.response_deadline.isoformat() if self.response_deadline else None,
            'is_overdue': self.is_overdue()
        }


class HealthAlertSystem:
    """
    System health alerting with escalation management.
    
    RESPONSIBILITIES:
    - Generate alerts from health assessments
    - Provide actionable recommendations
    - Track alert lifecycle
    - Monitor SLA compliance
    
    Per .clinerules VORTEX_OPERATIONAL_HEALTH.md:
    Alert levels with response requirements:
    - INFO: 24h response time, urgency 1
    - WARNING: 4h response time, urgency 2
    - CRITICAL: 1h response time, urgency 3
    """
    
    # Alert level configurations (per .clinerules)
    ALERT_LEVELS = {
        HealthAlertLevel.INFO: {
            "urgency": 1,
            "response_time_hours": 24,
            "requires_immediate": False
        },
        HealthAlertLevel.WARNING: {
            "urgency": 2,
            "response_time_hours": 4,
            "requires_immediate": False
        },
        HealthAlertLevel.CRITICAL: {
            "urgency": 3,
            "response_time_hours": 1,
            "requires_immediate": True
        }
    }
    
    def __init__(self, max_alert_history: int = 1000):
        """
        Initialize health alert system.
        
        Args:
            max_alert_history: Maximum alerts to keep in history
        """
        self.alert_history: List[HealthAlert] = []
        self.max_alert_history = max_alert_history
        
        # Active alerts (unresolved)
        self.active_alerts: Dict[str, HealthAlert] = {}
        
        # Statistics
        self.total_alerts = 0
        self.alerts_by_level = {
            HealthAlertLevel.INFO: 0,
            HealthAlertLevel.WARNING: 0,
            HealthAlertLevel.CRITICAL: 0
        }
        
        logger.info("Health Alert System initialized")
    
    def generate_health_alerts(self, health_assessment: Dict[str, Any]) -> List[HealthAlert]:
        """
        Generate appropriate alerts based on health assessment.
        
        Per .clinerules VORTEX_OPERATIONAL_HEALTH.md alerting logic.
        
        Args:
            health_assessment: Health assessment containing status and issues
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Extract assessment data
        status = health_assessment.get('status', 'HEALTHY')
        warnings = health_assessment.get('warnings', [])
        critical_issues = health_assessment.get('critical_issues', [])
        
        # Generate critical alerts
        for issue in critical_issues:
            alert = HealthAlert(
                level=HealthAlertLevel.CRITICAL,
                message=f"CRITICAL: {issue}",
                requires_immediate_response=True,
                suggested_actions=self.get_critical_issue_actions(issue)
            )
            alerts.append(alert)
            self._record_alert(alert)
        
        # Generate warning alerts
        for warning in warnings:
            alert = HealthAlert(
                level=HealthAlertLevel.WARNING,
                message=f"WARNING: {warning}",
                requires_immediate_response=False,
                suggested_actions=self.get_warning_actions(warning)
            )
            alerts.append(alert)
            self._record_alert(alert)
        
        # System status alert if degraded
        if status in ['DEGRADED', 'CRITICAL'] and not critical_issues:
            alert = HealthAlert(
                level=HealthAlertLevel.WARNING,
                message=f"System health status: {status}",
                requires_immediate_response=False,
                suggested_actions=["Review system metrics", "Check for resource constraints"]
            )
            alerts.append(alert)
            self._record_alert(alert)
        
        if alerts:
            logger.info(f"Generated {len(alerts)} health alerts")
        
        return alerts
    
    def get_critical_issue_actions(self, issue: str) -> List[str]:
        """
        Get suggested actions for critical issues.
        
        Per .clinerules VORTEX_OPERATIONAL_HEALTH.md action recommendations.
        
        Args:
            issue: Critical issue description
            
        Returns:
            List of suggested actions
        """
        actions = []
        issue_lower = issue.lower()
        
        if "manual review rate too high" in issue_lower:
            actions.extend([
                "Check fastpath promotion eligibility criteria",
                "Review AI model availability and performance",
                "Consider temporary threshold adjustments",
                "Scale manual review capacity if needed"
            ])
        
        elif "false positive rate too high" in issue_lower:
            actions.extend([
                "URGENT: Tighten confidence thresholds immediately",
                "Review recent SUBMIT_READY findings for quality",
                "Temporarily disable fastpath until investigation complete",
                "Analyze root cause of false positives"
            ])
        
        elif "memory usage too high" in issue_lower or "memory" in issue_lower:
            actions.extend([
                "Trigger emergency memory cleanup immediately",
                "Check for memory leaks in finding processing",
                "Consider temporary scan rate reduction",
                "Monitor system stability closely"
            ])
        
        elif "error rate too high" in issue_lower or "error" in issue_lower:
            actions.extend([
                "Review error logs for patterns",
                "Check AI model and database connectivity",
                "Consider enabling additional error recovery",
                "Monitor system components individually"
            ])
        
        elif "authority violation" in issue_lower:
            actions.extend([
                "URGENT: Review authority enforcement logic",
                "Check for bypass conditions in code",
                "Validate evidence validation pipeline",
                "Audit recent SUBMIT_READY findings"
            ])
        
        elif "overdue review" in issue_lower:
            actions.extend([
                "Escalate overdue reviews to high priority",
                "Allocate additional manual review resources",
                "Review SLA compliance procedures",
                "Consider automated retry for stalled reviews"
            ])
        
        else:
            # Generic critical actions
            actions.extend([
                "Investigate issue immediately",
                "Review recent system changes",
                "Check system logs for errors",
                "Monitor system metrics closely"
            ])
        
        return actions
    
    def get_warning_actions(self, warning: str) -> List[str]:
        """
        Get suggested actions for warnings.
        
        Args:
            warning: Warning description
            
        Returns:
            List of suggested actions
        """
        actions = []
        warning_lower = warning.lower()
        
        if "submit ready rate" in warning_lower:
            actions.extend([
                "Review evidence thresholds for over-strictness",
                "Check AI model availability",
                "Analyze SYSTEM_VERIFIED findings not progressing"
            ])
        
        elif "manual review time" in warning_lower:
            actions.extend([
                "Review manual review queue prioritization",
                "Check for bottlenecks in review process",
                "Consider process optimization"
            ])
        
        elif "ai availability" in warning_lower:
            actions.extend([
                "Check AI model service status",
                "Review API rate limits and quotas",
                "Enable fallback models if available"
            ])
        
        elif "evidence determinism" in warning_lower:
            actions.extend([
                "Review system verification patterns",
                "Enhance vulnerability-specific evidence criteria",
                "Check for evidence quality degradation"
            ])
        
        else:
            # Generic warning actions
            actions.extend([
                "Monitor metric trend",
                "Review related system components",
                "Consider preventive measures"
            ])
        
        return actions
    
    def _record_alert(self, alert: HealthAlert) -> None:
        """Record alert in history and active alerts."""
        # Add to history
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history = self.alert_history[-self.max_alert_history:]
        
        # Add to active alerts
        alert_key = f"{alert.level}_{alert.timestamp.isoformat()}"
        self.active_alerts[alert_key] = alert
        
        # Update statistics
        self.total_alerts += 1
        self.alerts_by_level[alert.level] += 1
        
        logger.info(f"Alert recorded: {alert.level} - {alert.message}")
    
    def get_active_alerts(self, level: Optional[HealthAlertLevel] = None) -> List[HealthAlert]:
        """
        Get active (unresolved) alerts.
        
        Args:
            level: Filter by alert level (optional)
            
        Returns:
            List of active alerts
        """
        alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_overdue_alerts(self) -> List[HealthAlert]:
        """Get alerts that are overdue for response."""
        return [
            alert for alert in self.active_alerts.values()
            if alert.is_overdue()
        ]
    
    def acknowledge_alert(self, alert_key: str, acknowledged_by: str = "system") -> bool:
        """
        Acknowledge alert.
        
        Args:
            alert_key: Alert identifier
            acknowledged_by: User/system acknowledging
            
        Returns:
            Success status
        """
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key].acknowledge(acknowledged_by)
            return True
        
        return False
    
    def resolve_alert(self, alert_key: str, resolution_notes: str = "") -> bool:
        """
        Resolve alert.
        
        Args:
            alert_key: Alert identifier
            resolution_notes: Resolution description
            
        Returns:
            Success status
        """
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key].resolve(resolution_notes)
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        active_count = len([a for a in self.active_alerts.values() if not a.resolved])
        overdue_count = len(self.get_overdue_alerts())
        
        # Calculate resolution rate
        resolved_count = len([a for a in self.alert_history if a.resolved])
        resolution_rate = resolved_count / self.total_alerts if self.total_alerts > 0 else 0.0
        
        return {
            'total_alerts': self.total_alerts,
            'active_alerts': active_count,
            'overdue_alerts': overdue_count,
            'resolved_alerts': resolved_count,
            'resolution_rate': resolution_rate,
            'alerts_by_level': {
                level.value: count
                for level, count in self.alerts_by_level.items()
            }
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary."""
        active = self.get_active_alerts()
        overdue = self.get_overdue_alerts()
        
        critical_active = [a for a in active if a.level == HealthAlertLevel.CRITICAL]
        warning_active = [a for a in active if a.level == HealthAlertLevel.WARNING]
        
        return {
            'total_active': len(active),
            'critical_active': len(critical_active),
            'warning_active': len(warning_active),
            'overdue': len(overdue),
            'requires_immediate_attention': len([a for a in active if a.requires_immediate_response]),
            'statistics': self.get_statistics()
        }


# Global health alert system
global_health_alert_system = HealthAlertSystem()