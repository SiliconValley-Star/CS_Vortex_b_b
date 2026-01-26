"""
VORTEX Authority Compliance - V17.0 ULTIMATE
Authority compliance tracking and audit per VORTEX_OPERATIONAL_HEALTH.md

CRITICAL: Tracks and reports authority hierarchy violations
"""

import structlog
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from domain.enums import VerificationStatus, AuthorityLevel
from domain.models import AssessmentResult
from config.constants import HEALTH_THRESHOLDS as OPERATIONAL_HEALTH_THRESHOLDS

logger = structlog.get_logger()


class AuthorityComplianceTracker:
    """
    Tracks authority compliance metrics and violations
    Per VORTEX_OPERATIONAL_HEALTH.md: Authority violation rate < 1%
    """
    
    def __init__(self):
        self.violations = []
        self.compliance_checks = []
        self.metrics_window_hours = 24  # Rolling 24h window
    
    def record_compliance_check(
        self,
        finding: AssessmentResult,
        check_type: str,
        compliant: bool,
        details: Optional[str] = None
    ) -> None:
        """Record a compliance check result."""
        record = {
            'timestamp': datetime.utcnow(),
            'finding_id': str(finding.id),
            'check_type': check_type,
            'compliant': compliant,
            'status': finding.status,
            'details': details or "",
            'authority_level': self._get_authority_level(finding)
        }
        
        self.compliance_checks.append(record)
        
        if not compliant:
            self._record_violation(finding, check_type, details)
    
    def _record_violation(
        self,
        finding: AssessmentResult,
        violation_type: str,
        details: Optional[str]
    ) -> None:
        """Record an authority violation."""
        violation = {
            'timestamp': datetime.utcnow(),
            'finding_id': str(finding.id),
            'violation_type': violation_type,
            'details': details or "",
            'status': finding.status,
            'severity': self._calculate_violation_severity(violation_type, finding)
        }
        
        self.violations.append(violation)
        
        logger.error(
            "Authority compliance violation",
            finding_id=str(finding.id),
            violation_type=violation_type,
            severity=violation['severity'],
            details=details
        )
    
    def _calculate_violation_severity(
        self,
        violation_type: str,
        finding: AssessmentResult
    ) -> str:
        """Calculate violation severity."""
        # CRITICAL: Violations in SUBMIT_READY findings
        if finding.status == VerificationStatus.SUBMIT_READY:
            return "CRITICAL"
        
        # HIGH: Violations bypassing system verification
        if "system_verification" in violation_type.lower():
            return "HIGH"
        
        # MEDIUM: AI authority violations
        if "ai_authority" in violation_type.lower():
            return "MEDIUM"
        
        # LOW: Other violations
        return "LOW"
    
    def _get_authority_level(self, finding: AssessmentResult) -> str:
        """Get authority level for finding."""
        if finding.verification_result and finding.verification_result.success:
            return "SYSTEM_VERIFICATION"
        elif finding.ai_analysis and finding.ai_analysis.success:
            return "AI_ADVISORY"
        elif finding.heuristic_score > 0:
            return "HEURISTIC"
        else:
            return "NONE"
    
    def get_compliance_rate(self, hours: Optional[int] = None) -> float:
        """
        Get authority compliance rate
        Per VORTEX_OPERATIONAL_HEALTH.md: Target > 99%
        
        Args:
            hours: Time window (default: 24 hours)
        
        Returns: Compliance rate (0.0-1.0)
        """
        hours = hours or self.metrics_window_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_checks = [
            c for c in self.compliance_checks
            if c['timestamp'] >= cutoff_time
        ]
        
        if not recent_checks:
            return 1.0  # No checks = perfect compliance
        
        compliant_count = sum(1 for c in recent_checks if c['compliant'])
        return compliant_count / len(recent_checks)
    
    def get_violation_rate(self, hours: Optional[int] = None) -> float:
        """
        Get authority violation rate
        Per VORTEX_OPERATIONAL_HEALTH.md: Critical threshold > 2%
        
        Args:
            hours: Time window (default: 24 hours)
        
        Returns: Violation rate (0.0-1.0)
        """
        return 1.0 - self.get_compliance_rate(hours)
    
    def get_violation_count_by_type(self, hours: Optional[int] = None) -> Dict[str, int]:
        """Get violation counts by type."""
        hours = hours or self.metrics_window_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_violations = [
            v for v in self.violations
            if v['timestamp'] >= cutoff_time
        ]
        
        counts = defaultdict(int)
        for violation in recent_violations:
            counts[violation['violation_type']] += 1
        
        return dict(counts)
    
    def get_violation_count_by_severity(self, hours: Optional[int] = None) -> Dict[str, int]:
        """Get violation counts by severity."""
        hours = hours or self.metrics_window_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_violations = [
            v for v in self.violations
            if v['timestamp'] >= cutoff_time
        ]
        
        counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for violation in recent_violations:
            severity = violation.get('severity', 'LOW')
            counts[severity] += 1
        
        return counts
    
    def get_compliance_report(self, hours: Optional[int] = None) -> Dict[str, any]:
        """
        Generate comprehensive compliance report
        Per VORTEX_OPERATIONAL_HEALTH.md: For health monitoring
        """
        hours = hours or self.metrics_window_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_checks = [
            c for c in self.compliance_checks
            if c['timestamp'] >= cutoff_time
        ]
        recent_violations = [
            v for v in self.violations
            if v['timestamp'] >= cutoff_time
        ]
        
        compliance_rate = self.get_compliance_rate(hours)
        violation_rate = self.get_violation_rate(hours)
        
        # Health status per VORTEX_OPERATIONAL_HEALTH.md thresholds
        if violation_rate > OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_critical:
            health_status = "CRITICAL"
        elif violation_rate > OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_max:
            health_status = "DEGRADED"
        elif violation_rate > OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_target:
            health_status = "ATTENTION"
        else:
            health_status = "HEALTHY"
        
        report = {
            'timestamp': datetime.utcnow(),
            'time_window_hours': hours,
            'health_status': health_status,
            
            # Core metrics
            'compliance_rate': compliance_rate,
            'violation_rate': violation_rate,
            'total_checks': len(recent_checks),
            'total_violations': len(recent_violations),
            
            # Breakdown by type
            'violations_by_type': self.get_violation_count_by_type(hours),
            'violations_by_severity': self.get_violation_count_by_severity(hours),
            
            # Compliance by check type
            'compliance_by_check_type': self._get_compliance_by_check_type(recent_checks),
            
            # Trends
            'violation_trend': self._calculate_violation_trend(hours),
            
            # Recommendations
            'recommendations': self._generate_compliance_recommendations(
                violation_rate,
                recent_violations
            )
        }
        
        logger.info(
            "Compliance report generated",
            health_status=health_status,
            compliance_rate=compliance_rate,
            violation_count=len(recent_violations)
        )
        
        return report
    
    def _get_compliance_by_check_type(self, checks: List[Dict]) -> Dict[str, Dict]:
        """Get compliance rate by check type."""
        by_type = defaultdict(lambda: {'total': 0, 'compliant': 0})
        
        for check in checks:
            check_type = check['check_type']
            by_type[check_type]['total'] += 1
            if check['compliant']:
                by_type[check_type]['compliant'] += 1
        
        # Calculate rates
        result = {}
        for check_type, counts in by_type.items():
            result[check_type] = {
                'total': counts['total'],
                'compliant': counts['compliant'],
                'rate': counts['compliant'] / counts['total'] if counts['total'] > 0 else 1.0
            }
        
        return result
    
    def _calculate_violation_trend(self, hours: int) -> str:
        """Calculate violation trend (increasing/stable/decreasing)."""
        if hours < 4:
            return "INSUFFICIENT_DATA"
        
        # Split window into two halves
        midpoint = datetime.utcnow() - timedelta(hours=hours/2)
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        first_half = [
            v for v in self.violations
            if cutoff <= v['timestamp'] < midpoint
        ]
        second_half = [
            v for v in self.violations
            if v['timestamp'] >= midpoint
        ]
        
        if not first_half and not second_half:
            return "STABLE"
        
        first_rate = len(first_half) / (hours/2)
        second_rate = len(second_half) / (hours/2)
        
        if second_rate > first_rate * 1.2:
            return "INCREASING"
        elif second_rate < first_rate * 0.8:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _generate_compliance_recommendations(
        self,
        violation_rate: float,
        recent_violations: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on compliance status."""
        recommendations = []
        
        # Critical violation rate
        if violation_rate > OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_critical:
            recommendations.append(
                "URGENT: Authority violation rate CRITICAL - Review authority enforcement logic immediately"
            )
            recommendations.append(
                "URGENT: Audit recent SUBMIT_READY findings for compliance violations"
            )
            recommendations.append(
                "Consider temporarily increasing manual review rate until issue resolved"
            )
        
        # High violation rate
        elif violation_rate > OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_max:
            recommendations.append(
                "Authority violation rate above acceptable threshold - Investigation required"
            )
            recommendations.append(
                "Review authority validation pipeline for potential bypasses"
            )
        
        # Check for specific violation patterns
        violation_types = defaultdict(int)
        for v in recent_violations:
            violation_types[v['violation_type']] += 1
        
        # AI authority violations
        if violation_types.get('ai_authority', 0) > 3:
            recommendations.append(
                "Multiple AI authority violations detected - Review AI integration limits"
            )
        
        # System verification bypasses
        if violation_types.get('system_verification', 0) > 2:
            recommendations.append(
                "System verification bypasses detected - Review verification requirements"
            )
        
        # Unknown value violations
        if violation_types.get('unknown_values', 0) > 2:
            recommendations.append(
                "Multiple UNKNOWN value violations - Review AI field handling"
            )
        
        # If no violations, provide positive feedback
        if not recommendations and violation_rate <= OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_target:
            recommendations.append(
                "Authority compliance excellent - System maintaining security standards"
            )
        
        return recommendations
    
    def check_threshold_alerts(self) -> List[Dict[str, any]]:
        """
        Check if any thresholds are breached
        Returns list of alerts to be raised
        """
        alerts = []
        
        violation_rate = self.get_violation_rate()
        
        # Critical threshold
        if violation_rate > OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_critical:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'AUTHORITY_VIOLATION_RATE',
                'message': f'Authority violation rate {violation_rate:.1%} exceeds critical threshold',
                'current_value': violation_rate,
                'threshold': OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_critical,
                'action_required': True,
                'suggested_actions': [
                    'Review authority enforcement immediately',
                    'Audit recent SUBMIT_READY findings',
                    'Consider increasing manual review rate'
                ]
            })
        
        # Warning threshold
        elif violation_rate > OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_max:
            alerts.append({
                'level': 'WARNING',
                'type': 'AUTHORITY_VIOLATION_RATE',
                'message': f'Authority violation rate {violation_rate:.1%} above acceptable threshold',
                'current_value': violation_rate,
                'threshold': OPERATIONAL_HEALTH_THRESHOLDS.authority_violation_max,
                'action_required': True,
                'suggested_actions': [
                    'Investigate violation patterns',
                    'Review validation pipeline'
                ]
            })
        
        # Check for critical severity violations
        severity_counts = self.get_violation_count_by_severity()
        if severity_counts.get('CRITICAL', 0) > 0:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'CRITICAL_VIOLATIONS',
                'message': f'{severity_counts["CRITICAL"]} CRITICAL authority violations detected',
                'current_value': severity_counts['CRITICAL'],
                'threshold': 0,
                'action_required': True,
                'suggested_actions': [
                    'Review CRITICAL violations immediately',
                    'Check for systematic issues'
                ]
            })
        
        return alerts
    
    def reset_metrics(self) -> None:
        """Reset all metrics (for testing or fresh start)."""
        self.violations.clear()
        self.compliance_checks.clear()
        logger.info("Authority compliance metrics reset")


class ComplianceAuditor:
    """
    Audit trail manager for authority compliance
    Provides immutable audit records
    """
    
    def __init__(self):
        self.audit_log = []
    
    def record_authority_decision(
        self,
        finding: AssessmentResult,
        decision_type: str,
        decision: str,
        authority_level: AuthorityLevel,
        rationale: str
    ) -> None:
        """Record an authority decision for audit trail."""
        entry = {
            'timestamp': datetime.utcnow(),
            'finding_id': str(finding.id),
            'decision_type': decision_type,
            'decision': decision,
            'authority_level': authority_level.value,
            'status': finding.status.value,
            'rationale': rationale,
            'system_confidence': finding.verification_result.confidence if finding.verification_result else None,
            'ai_confidence': finding.ai_analysis.confidence if finding.ai_analysis else None
        }
        
        self.audit_log.append(entry)
    
    def get_audit_trail(
        self,
        finding_id: Optional[str] = None,
        hours: Optional[int] = None
    ) -> List[Dict]:
        """Get audit trail, optionally filtered."""
        log = self.audit_log
        
        # Filter by finding ID
        if finding_id:
            log = [e for e in log if e['finding_id'] == finding_id]
        
        # Filter by time window
        if hours:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            log = [e for e in log if e['timestamp'] >= cutoff]
        
        return log
    
    def export_audit_trail(self, hours: Optional[int] = None) -> str:
        """Export audit trail as formatted text."""
        entries = self.get_audit_trail(hours=hours)
        
        lines = ["VORTEX Authority Compliance Audit Trail"]
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        if hours:
            lines.append(f"Time window: Last {hours} hours")
        lines.append(f"Total entries: {len(entries)}")
        lines.append("")
        
        for entry in entries:
            lines.append(f"[{entry['timestamp'].isoformat()}]")
            lines.append(f"  Finding: {entry['finding_id']}")
            lines.append(f"  Decision: {entry['decision_type']} → {entry['decision']}")
            lines.append(f"  Authority: {entry['authority_level']}")
            lines.append(f"  Status: {entry['status']}")
            lines.append(f"  Rationale: {entry['rationale']}")
            lines.append("")
        
        return "\n".join(lines)


# Global instances
global_compliance_tracker = AuthorityComplianceTracker()
global_compliance_auditor = ComplianceAuditor()