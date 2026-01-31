"""
VORTEX Health Monitoring System - V17.0 ULTIMATE
Per VORTEX_OPERATIONAL_HEALTH.md

CRITICAL: This system ensures VORTEX remains operationally viable.
Health thresholds prevent degradation and enable predictive maintenance.

HEALTH PHILOSOPHY:
- Monitor continuously
- Alert promptly  
- Tune conservatively
- Preserve quality always
"""

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from domain.enums import VerificationStatus, FindingSeverity
from domain.models import AssessmentResult
from config.constants import OperationalHealthThresholds

logger = logging.getLogger(__name__)

# WebSocket support for real-time updates
try:
    from web.websockets import global_websocket_manager
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("WebSocket module not available - health updates will not be broadcasted")


@dataclass
class SystemHealthMetrics:
    """Current system health metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Core effectiveness metrics
    submit_ready_rate: float = 0.0
    manual_review_rate: float = 0.0
    false_positive_rate: float = 0.0
    
    # System performance
    ai_availability: float = 0.0
    ai_success_rate: float = 0.0
    system_verification_success: float = 0.0
    memory_usage_mb: float = 0.0
    error_rate: float = 0.0
    
    # Manual review efficiency
    manual_queue_size: int = 0
    overdue_reviews: int = 0
    avg_manual_hours: float = 0.0
    manual_sla_compliance: float = 0.0
    manual_conversion_rate: float = 0.0
    
    # Authority compliance (CRITICAL)
    authority_violation_rate: float = 0.0
    evidence_determinism_avg: float = 0.0
    unknown_value_rate: float = 0.0
    
    # Finding distribution
    total_findings: int = 0
    
    # Metadata
    collection_duration_ms: float = 0.0


@dataclass
class HealthAssessment:
    """Health assessment result."""
    status: str  # HEALTHY, ATTENTION, DEGRADED, CRITICAL
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HealthAlert:
    """System health alert."""
    level: str  # INFO, WARNING, CRITICAL
    category: str
    message: str
    timestamp: datetime
    requires_immediate_response: bool
    suggested_actions: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class TuningRecommendation:
    """Automated tuning recommendation."""
    category: str
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    actions: List[str]
    estimated_impact: str
    auto_executable: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class HealthMonitor:
    """
    Comprehensive system health monitoring.
    
    RESPONSIBILITIES:
    - Collect metrics from all system components
    - Validate against operational thresholds
    - Track authority compliance
    - Generate health assessments and alerts
    - Provide tuning recommendations
    
    THRESHOLDS (V11.1):
    - Submit Ready Rate: >3% (target: 5-8%)
    - Manual Review Rate: <75%
    - False Positive Rate: <15%
    - Authority Violation Rate: <1%
    """
    
    def __init__(self):
        self.thresholds = OperationalHealthThresholds()
        self.metrics_history: List[SystemHealthMetrics] = []
        self.alert_history: List[HealthAlert] = []
        self.monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Component references (set by integration)
        self.authority_enforcer = None
        self.evidence_validator = None
        self.ai_engine = None
        self.workflow_orchestrator = None
        self.database = None
        
        # WebSocket manager for real-time updates
        self.websocket_manager = global_websocket_manager if WEBSOCKET_AVAILABLE else None
        
    async def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous health monitoring (default: every 5 minutes)."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        logger.info(f"Health monitoring started (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current health metrics synchronously.
        Returns the latest metrics from history or empty dict.
        """
        if self.metrics_history:
            latest = self.metrics_history[-1]
            return {
                'memory_usage_mb': latest.memory_usage_mb,
                'ai_availability': latest.ai_availability,
                'authority_violation_rate': latest.authority_violation_rate,
                'evidence_determinism_avg': latest.evidence_determinism_avg,
                'unknown_value_rate': latest.unknown_value_rate,
                'submit_ready_rate': latest.submit_ready_rate,
                'manual_review_rate': latest.manual_review_rate,
                'false_positive_rate': latest.false_positive_rate,
                'error_rate': latest.error_rate,
                'total_findings': latest.total_findings,
                'timestamp': latest.timestamp.isoformat()
            }
        return {}
    
    async def _monitoring_loop(self, interval_seconds: int):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics and assess health
                await self.comprehensive_health_check()
                
                # Sleep until next check
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def comprehensive_health_check(self) -> 'SystemHealthReport':
        """
        Perform comprehensive health check.
        
        Returns complete health report with:
        - Current metrics
        - Health assessment
        - Authority compliance
        - Tuning recommendations
        - Alerts
        """
        start_time = datetime.utcnow()
        logger.debug("Starting comprehensive health check")
        
        try:
            # Collect all metrics
            current_metrics = await self._collect_all_metrics()
            current_metrics.collection_duration_ms = (
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            # Store in history (keep last 168 = 1 week at 5min intervals)
            self.metrics_history.append(current_metrics)
            if len(self.metrics_history) > 168:
                self.metrics_history.pop(0)
            
            # Assess health against thresholds
            health_assessment = self._assess_health_status(current_metrics)
            
            # Check authority compliance
            authority_compliance = self._assess_authority_compliance(current_metrics)
            
            # Generate tuning recommendations
            tuning_recommendations = self._generate_tuning_recommendations(current_metrics)
            
            # Generate alerts
            alerts = self._generate_health_alerts(health_assessment, current_metrics)
            self.alert_history.extend(alerts)
            
            # Analyze trends if enough history
            trend_analysis = self._analyze_performance_trends()
            
            # Create report
            report = SystemHealthReport(
                timestamp=datetime.utcnow(),
                overall_status=health_assessment.status,
                current_metrics=current_metrics,
                threshold_violations=self._get_threshold_violations(current_metrics),
                authority_compliance=authority_compliance,
                tuning_recommendations=tuning_recommendations,
                alerts=alerts,
                trend_analysis=trend_analysis
            )
            
            # Log summary
            self._log_health_summary(report)
            
            # Broadcast health update via WebSocket
            self._broadcast_health_update(report)
            
            # Broadcast critical alerts
            for alert in alerts:
                if alert.level == 'CRITICAL':
                    self._broadcast_critical_alert(alert)
            
            return report
            
        except Exception as e:
            logger.error(f"Comprehensive health check failed: {e}", exc_info=True)
            raise
    
    async def _collect_all_metrics(self) -> SystemHealthMetrics:
        """Collect comprehensive system metrics."""
        metrics = SystemHealthMetrics()
        
        try:
            # Finding distribution stats
            finding_stats = await self._get_finding_distribution_stats()
            if finding_stats and finding_stats.get('total', 0) > 0:
                total = finding_stats['total']
                metrics.total_findings = total
                metrics.submit_ready_rate = finding_stats.get('submit_ready', 0) / total
                metrics.manual_review_rate = finding_stats.get('needs_manual', 0) / total
                metrics.false_positive_rate = finding_stats.get('false_positive', 0) / total
            
            # AI system metrics
            if self.ai_engine:
                ai_stats = await self._get_ai_stats()
                metrics.ai_availability = ai_stats.get('availability_rate', 0.0)
                metrics.ai_success_rate = ai_stats.get('success_rate', 0.0)
            
            # System verification metrics
            verification_stats = await self._get_verification_stats()
            metrics.system_verification_success = verification_stats.get('success_rate', 0.0)
            
            # Memory metrics
            memory_stats = self._get_memory_stats()
            metrics.memory_usage_mb = memory_stats.get('usage_mb', 0.0)
            
            # Error rate
            metrics.error_rate = await self._get_error_rate()
            
            # Manual review metrics
            manual_stats = await self._get_manual_review_stats()
            metrics.manual_queue_size = manual_stats.get('queue_size', 0)
            metrics.overdue_reviews = manual_stats.get('overdue_count', 0)
            metrics.avg_manual_hours = manual_stats.get('average_age_hours', 0.0)
            metrics.manual_sla_compliance = manual_stats.get('sla_compliance', 0.0)
            metrics.manual_conversion_rate = manual_stats.get('conversion_rate', 0.0)
            
            # Authority compliance metrics (CRITICAL)
            authority_metrics = await self._collect_authority_metrics()
            metrics.authority_violation_rate = authority_metrics.get('violation_rate', 0.0)
            metrics.evidence_determinism_avg = authority_metrics.get('determinism_avg', 0.0)
            metrics.unknown_value_rate = authority_metrics.get('unknown_value_rate', 0.0)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    async def _get_finding_distribution_stats(self) -> Dict[str, int]:
        """Get finding distribution statistics."""
        if not self.database:
            return {'total': 0}
        
        try:
            # Count findings by status (last 24 hours for relevance)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # This would query database - placeholder implementation
            return {
                'total': 0,
                'submit_ready': 0,
                'needs_manual': 0,
                'false_positive': 0,
                'ai_confirmed': 0,
                'system_verified': 0
            }
        except Exception as e:
            logger.error(f"Error getting finding stats: {e}")
            return {'total': 0}
    
    async def _get_ai_stats(self) -> Dict[str, float]:
        """Get AI system statistics."""
        if not self.ai_engine:
            return {'availability_rate': 0.0, 'success_rate': 0.0}
        
        try:
            # Get AI engine stats
            if hasattr(self.ai_engine, 'get_stats'):
                return self.ai_engine.get_stats()
        except Exception as e:
            logger.error(f"Error getting AI stats: {e}")
        
        return {'availability_rate': 0.0, 'success_rate': 0.0}
    
    async def _get_verification_stats(self) -> Dict[str, float]:
        """Get system verification statistics."""
        # Placeholder - would query verification system
        return {'success_rate': 0.0}
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'usage_mb': memory_info.rss / (1024 * 1024),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'usage_mb': 0.0}
    
    async def _get_error_rate(self) -> float:
        """Get system error rate (last hour)."""
        # Placeholder - would analyze error logs
        return 0.0
    
    async def _get_manual_review_stats(self) -> Dict[str, Any]:
        """Get manual review queue statistics."""
        # Placeholder - would query manual review manager
        return {
            'queue_size': 0,
            'overdue_count': 0,
            'average_age_hours': 0.0,
            'sla_compliance': 0.0,
            'conversion_rate': 0.0
        }
    
    async def _collect_authority_metrics(self) -> Dict[str, float]:
        """
        Collect authority compliance metrics.
        
        CRITICAL: These metrics ensure system integrity.
        Authority violations indicate fundamental system failure.
        """
        if not self.authority_enforcer or not self.evidence_validator:
            return {
                'violation_rate': 0.0,
                'determinism_avg': 0.0,
                'unknown_value_rate': 0.0
            }
        
        try:
            # Get recent findings for analysis
            recent_findings = await self._get_recent_findings(limit=100)
            if not recent_findings:
                return {
                    'violation_rate': 0.0,
                    'determinism_avg': 0.0,
                    'unknown_value_rate': 0.0
                }
            
            authority_violations = 0
            evidence_scores = []
            unknown_value_count = 0
            
            for finding in recent_findings:
                # Check authority compliance for SUBMIT_READY
                if finding.status == VerificationStatus.SUBMIT_READY:
                    if not self.authority_enforcer.validate_submit_ready_authority(finding):
                        authority_violations += 1
                        logger.error(f"AUTHORITY VIOLATION detected in finding {finding.id}")
                
                # Track evidence determinism
                if self.evidence_validator:
                    determinism = self.evidence_validator.assess_evidence_determinism(finding)
                    evidence_scores.append(determinism)
                
                # Track UNKNOWN values
                if self.authority_enforcer._has_unknown_values(finding):
                    unknown_value_count += 1
            
            total = len(recent_findings)
            
            return {
                'violation_rate': authority_violations / total if total > 0 else 0.0,
                'determinism_avg': statistics.mean(evidence_scores) if evidence_scores else 0.0,
                'unknown_value_rate': unknown_value_count / total if total > 0 else 0.0,
                'total_findings_checked': total
            }
            
        except Exception as e:
            logger.error(f"Error collecting authority metrics: {e}")
            return {
                'violation_rate': 0.0,
                'determinism_avg': 0.0,
                'unknown_value_rate': 0.0
            }
    
    async def _get_recent_findings(self, limit: int = 100) -> List[AssessmentResult]:
        """Get recent findings for analysis."""
        if not self.database:
            return []
        
        try:
            # Would query database for recent findings
            return []
        except Exception as e:
            logger.error(f"Error getting recent findings: {e}")
            return []
    
    def _assess_health_status(self, metrics: SystemHealthMetrics) -> HealthAssessment:
        """
        Assess overall system health against thresholds.
        
        STATUS LEVELS:
        - HEALTHY: All metrics within target ranges
        - ATTENTION: 2+ warnings but no critical issues
        - DEGRADED: 1 critical issue
        - CRITICAL: 2+ critical issues
        """
        warnings = []
        critical_issues = []
        
        # Submit ready rate
        if metrics.submit_ready_rate < self.thresholds.min_submit_ready_rate:
            # Critical if less than half of minimum
            if metrics.submit_ready_rate < (self.thresholds.min_submit_ready_rate / 2):
                critical_issues.append(
                    f"Submit ready rate CRITICAL: {metrics.submit_ready_rate:.1%} "
                    f"(minimum: {self.thresholds.min_submit_ready_rate:.1%})"
                )
            else:
                warnings.append(
                    f"Submit ready rate low: {metrics.submit_ready_rate:.1%} "
                    f"(target: {self.thresholds.target_submit_ready_rate:.1%})"
                )
        
        # Manual review rate
        if metrics.manual_review_rate > self.thresholds.max_manual_review_rate:
            # Critical if double the maximum
            if metrics.manual_review_rate > (self.thresholds.max_manual_review_rate * 2):
                critical_issues.append(
                    f"Manual review rate CRITICAL: {metrics.manual_review_rate:.1%} "
                    f"(maximum: {self.thresholds.max_manual_review_rate:.1%})"
                )
            else:
                warnings.append(
                    f"Manual review rate high: {metrics.manual_review_rate:.1%} "
                    f"(target: {self.thresholds.target_manual_review_rate:.1%})"
                )
        
        # False positive rate
        if metrics.false_positive_rate > self.thresholds.max_false_positive_rate:
            critical_issues.append(
                f"False positive rate CRITICAL: {metrics.false_positive_rate:.1%} "
                f"(maximum: {self.thresholds.max_false_positive_rate:.1%})"
            )
        
        # Authority violation rate (CRITICAL - system integrity)
        if metrics.authority_violation_rate > self.thresholds.max_authority_violation_rate:
            critical_issues.append(
                f"AUTHORITY VIOLATIONS detected: {metrics.authority_violation_rate:.1%} "
                f"(maximum: {self.thresholds.max_authority_violation_rate:.1%}) - "
                "SYSTEM INTEGRITY COMPROMISED"
            )
        
        # Evidence determinism
        if metrics.evidence_determinism_avg < self.thresholds.min_evidence_determinism_avg:
            warnings.append(
                f"Evidence determinism low: {metrics.evidence_determinism_avg:.2f} "
                f"(minimum: {self.thresholds.min_evidence_determinism_avg:.2f})"
            )
        
        # AI availability
        if metrics.ai_availability < self.thresholds.min_ai_availability:
            warnings.append(
                f"AI availability low: {metrics.ai_availability:.1%} "
                f"(minimum: {self.thresholds.min_ai_availability:.1%})"
            )
        
        # Memory usage
        if metrics.memory_usage_mb > self.thresholds.max_memory_usage_mb:
            critical_issues.append(
                f"Memory usage CRITICAL: {metrics.memory_usage_mb:.0f}MB "
                f"(maximum: {self.thresholds.max_memory_usage_mb:.0f}MB)"
            )
        
        # Error rate
        if metrics.error_rate > self.thresholds.max_error_rate:
            critical_issues.append(
                f"Error rate CRITICAL: {metrics.error_rate:.1%} "
                f"(maximum: {self.thresholds.max_error_rate:.1%})"
            )
        
        # Manual review efficiency
        if metrics.overdue_reviews > 10:  # Absolute threshold
            critical_issues.append(
                f"Too many overdue reviews: {metrics.overdue_reviews}"
            )
        
        # Determine overall status
        if len(critical_issues) >= 2:
            status = "CRITICAL"
        elif len(critical_issues) >= 1:
            status = "DEGRADED"
        elif len(warnings) >= 2:
            status = "ATTENTION"
        else:
            status = "HEALTHY"
        
        return HealthAssessment(
            status=status,
            warnings=warnings,
            critical_issues=critical_issues,
            timestamp=datetime.utcnow()
        )
    
    def _assess_authority_compliance(self, metrics: SystemHealthMetrics) -> Dict[str, Any]:
        """Assess authority hierarchy compliance."""
        compliance_status = "COMPLIANT"
        issues = []
        
        if metrics.authority_violation_rate > 0.0:
            compliance_status = "VIOLATIONS_DETECTED"
            issues.append(f"Authority violations: {metrics.authority_violation_rate:.1%}")
        
        if metrics.unknown_value_rate > self.thresholds.max_unknown_value_rate:
            issues.append(f"High UNKNOWN value rate: {metrics.unknown_value_rate:.1%}")
        
        return {
            'status': compliance_status,
            'violation_rate': metrics.authority_violation_rate,
            'evidence_quality': metrics.evidence_determinism_avg,
            'unknown_value_rate': metrics.unknown_value_rate,
            'issues': issues
        }
    
    def _get_threshold_violations(self, metrics: SystemHealthMetrics) -> List[str]:
        """Get list of threshold violations."""
        violations = []
        
        if metrics.submit_ready_rate < self.thresholds.min_submit_ready_rate:
            violations.append("submit_ready_rate")
        
        if metrics.manual_review_rate > self.thresholds.max_manual_review_rate:
            violations.append("manual_review_rate")
        
        if metrics.false_positive_rate > self.thresholds.max_false_positive_rate:
            violations.append("false_positive_rate")
        
        if metrics.authority_violation_rate > self.thresholds.max_authority_violation_rate:
            violations.append("authority_violation_rate")
        
        if metrics.memory_usage_mb > self.thresholds.max_memory_usage_mb:
            violations.append("memory_usage")
        
        if metrics.error_rate > self.thresholds.max_error_rate:
            violations.append("error_rate")
        
        return violations
    
    def _generate_tuning_recommendations(self, 
                                        metrics: SystemHealthMetrics) -> List[TuningRecommendation]:
        """Generate automated tuning recommendations."""
        recommendations = []
        
        # Submit ready rate optimization
        if metrics.submit_ready_rate < self.thresholds.min_submit_ready_rate:
            recommendations.append(TuningRecommendation(
                category='SUBMIT_READY_OPTIMIZATION',
                priority='HIGH',
                description=f'Submit ready rate {metrics.submit_ready_rate:.1%} below minimum',
                actions=[
                    'Review evidence thresholds for over-strictness',
                    'Check AI model availability and performance',
                    'Analyze SYSTEM_VERIFIED findings not progressing',
                    'Consider vulnerability-specific threshold adjustments'
                ],
                estimated_impact='+1-2% submit ready rate',
                auto_executable=True
            ))
        
        # Authority violation (CRITICAL)
        if metrics.authority_violation_rate > self.thresholds.max_authority_violation_rate:
            recommendations.append(TuningRecommendation(
                category='AUTHORITY_COMPLIANCE',
                priority='CRITICAL',
                description=f'Authority violations detected: {metrics.authority_violation_rate:.1%}',
                actions=[
                    'URGENT: Review authority enforcement logic',
                    'Check for bypass conditions in code',
                    'Validate evidence validation pipeline',
                    'Audit recent SUBMIT_READY findings'
                ],
                estimated_impact='Restore authority hierarchy compliance',
                auto_executable=False  # Requires manual investigation
            ))
        
        # Evidence quality
        if metrics.evidence_determinism_avg < self.thresholds.min_evidence_determinism_avg:
            recommendations.append(TuningRecommendation(
                category='EVIDENCE_QUALITY',
                priority='MEDIUM',
                description=f'Evidence determinism {metrics.evidence_determinism_avg:.2f} below minimum',
                actions=[
                    'Review system verification patterns',
                    'Enhance vulnerability-specific evidence criteria',
                    'Improve behavioral analysis accuracy',
                    'Check for evidence quality degradation causes'
                ],
                estimated_impact='+0.05-0.10 evidence determinism',
                auto_executable=True
            ))
        
        # Memory pressure
        if metrics.memory_usage_mb > self.thresholds.max_memory_usage_mb:
            recommendations.append(TuningRecommendation(
                category='RESOURCE_OPTIMIZATION',
                priority='HIGH',
                description=f'Memory usage {metrics.memory_usage_mb:.0f}MB exceeds limit',
                actions=[
                    'Trigger emergency memory cleanup',
                    'Check for memory leaks in processing',
                    'Reduce concurrent processing limits',
                    'Enable aggressive response truncation'
                ],
                estimated_impact='-1000-2000MB memory usage',
                auto_executable=True
            ))
        
        return recommendations
    
    def _generate_health_alerts(self, 
                               assessment: HealthAssessment,
                               metrics: SystemHealthMetrics) -> List[HealthAlert]:
        """Generate health alerts based on assessment."""
        alerts = []
        
        # Critical alerts
        for issue in assessment.critical_issues:
            alerts.append(HealthAlert(
                level='CRITICAL',
                category=self._categorize_issue(issue),
                message=f"System health critical: {issue}",
                timestamp=datetime.utcnow(),
                requires_immediate_response=True,
                suggested_actions=self._get_critical_actions(issue)
            ))
        
        # Warning alerts
        for warning in assessment.warnings:
            alerts.append(HealthAlert(
                level='WARNING',
                category=self._categorize_issue(warning),
                message=f"System health warning: {warning}",
                timestamp=datetime.utcnow(),
                requires_immediate_response=False,
                suggested_actions=self._get_warning_actions(warning)
            ))
        
        return alerts
    
    def _categorize_issue(self, issue: str) -> str:
        """Categorize health issue."""
        issue_lower = issue.lower()
        
        if 'submit ready' in issue_lower:
            return 'SUBMIT_READY_RATE'
        elif 'manual review' in issue_lower:
            return 'MANUAL_QUEUE'
        elif 'false positive' in issue_lower:
            return 'QUALITY'
        elif 'authority' in issue_lower:
            return 'AUTHORITY_COMPLIANCE'
        elif 'memory' in issue_lower:
            return 'RESOURCE'
        elif 'error' in issue_lower:
            return 'SYSTEM_ERRORS'
        else:
            return 'GENERAL'
    
    def _get_critical_actions(self, issue: str) -> List[str]:
        """Get suggested actions for critical issue."""
        issue_lower = issue.lower()
        
        if 'authority violation' in issue_lower:
            return [
                'URGENT: Review authority enforcement immediately',
                'Check for code bypassing authority validation',
                'Audit recent SUBMIT_READY findings',
                'Temporarily disable automated submission'
            ]
        elif 'false positive' in issue_lower:
            return [
                'URGENT: Tighten confidence thresholds',
                'Review recent submissions for quality',
                'Temporarily disable fastpath',
                'Analyze root cause of false positives'
            ]
        elif 'memory' in issue_lower:
            return [
                'Trigger emergency memory cleanup',
                'Check for memory leaks',
                'Reduce processing rate temporarily',
                'Monitor system stability'
            ]
        else:
            return ['Investigate issue immediately', 'Review system logs']
    
    def _get_warning_actions(self, warning: str) -> List[str]:
        """Get suggested actions for warning."""
        warning_lower = warning.lower()
        
        if 'submit ready rate' in warning_lower:
            return [
                'Review evidence thresholds',
                'Check AI availability',
                'Analyze blocked findings'
            ]
        elif 'ai availability' in warning_lower:
            return [
                'Check AI model status',
                'Review fallback configuration',
                'Consider alternative models'
            ]
        else:
            return ['Monitor situation', 'Review metrics']
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from history."""
        if len(self.metrics_history) < 12:  # Need at least 1 hour of data
            return {'status': 'insufficient_data'}
        
        recent = self.metrics_history[-12:]  # Last hour
        
        # Calculate trends
        submit_rates = [m.submit_ready_rate for m in recent]
        manual_rates = [m.manual_review_rate for m in recent]
        
        return {
            'status': 'analyzed',
            'submit_ready_trend': self._calculate_trend(submit_rates),
            'manual_review_trend': self._calculate_trend(manual_rates),
            'data_points': len(recent)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        avg_first_half = statistics.mean(values[:len(values)//2])
        avg_second_half = statistics.mean(values[len(values)//2:])
        
        change = avg_second_half - avg_first_half
        
        if abs(change) < 0.01:  # Less than 1% change
            return 'stable'
        elif change > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _log_health_summary(self, report: 'SystemHealthReport'):
        """Log health check summary."""
        logger.info(
            f"Health Check: {report.overall_status} | "
            f"Submit Ready: {report.current_metrics.submit_ready_rate:.1%} | "
            f"Manual: {report.current_metrics.manual_review_rate:.1%} | "
            f"FP Rate: {report.current_metrics.false_positive_rate:.1%} | "
            f"Authority Violations: {report.current_metrics.authority_violation_rate:.1%}"
        )
        
        if report.threshold_violations:
            logger.warning(f"Threshold violations: {', '.join(report.threshold_violations)}")
        
        if report.tuning_recommendations:
            logger.info(f"Tuning recommendations: {len(report.tuning_recommendations)}")
    
    def _broadcast_health_update(self, report: 'SystemHealthReport'):
        """
        Broadcast health update via WebSocket for real-time monitoring.
        
        Sends comprehensive health data to connected clients.
        """
        if not self.websocket_manager or not self.websocket_manager.is_available():
            return
        
        try:
            # Prepare health data for broadcast
            health_data = {
                'overall_status': report.overall_status,
                'timestamp': report.timestamp.isoformat(),
                'metrics': {
                    'submit_ready_rate': report.current_metrics.submit_ready_rate,
                    'manual_review_rate': report.current_metrics.manual_review_rate,
                    'false_positive_rate': report.current_metrics.false_positive_rate,
                    'authority_violation_rate': report.current_metrics.authority_violation_rate,
                    'evidence_determinism_avg': report.current_metrics.evidence_determinism_avg,
                    'ai_availability': report.current_metrics.ai_availability,
                    'ai_success_rate': report.current_metrics.ai_success_rate,
                    'memory_usage_mb': report.current_metrics.memory_usage_mb,
                    'error_rate': report.current_metrics.error_rate,
                    'total_findings': report.current_metrics.total_findings,
                    'manual_queue_size': report.current_metrics.manual_queue_size,
                    'overdue_reviews': report.current_metrics.overdue_reviews
                },
                'threshold_violations': report.threshold_violations,
                'authority_compliance': report.authority_compliance,
                'recommendations_count': len(report.tuning_recommendations),
                'alerts_count': len(report.alerts),
                'trend_analysis': report.trend_analysis
            }
            
            # Broadcast to system_health room
            self.websocket_manager.broadcast_system_health_update(health_data)
            
            logger.debug("Health update broadcasted via WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to broadcast health update: {e}")
    
    def _broadcast_critical_alert(self, alert: HealthAlert):
        """
        Broadcast critical alert via WebSocket.
        
        Sends immediate alert to all connected clients for critical issues.
        """
        if not self.websocket_manager or not self.websocket_manager.is_available():
            return
        
        try:
            # Prepare alert data
            alert_data = {
                'level': alert.level,
                'category': alert.category,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'requires_immediate_response': alert.requires_immediate_response,
                'suggested_actions': alert.suggested_actions
            }
            
            # Broadcast to all clients (broadcast room)
            self.websocket_manager.broadcast_system_alert(
                alert_level=alert.level,
                message=alert.message,
                details=alert_data
            )
            
            # If authority violation, use special broadcast
            if 'authority' in alert.category.lower():
                self.websocket_manager.broadcast_authority_violation({
                    'alert': alert_data,
                    'requires_immediate_action': True
                })
            
            logger.info(f"Critical alert broadcasted: {alert.category}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast critical alert: {e}")


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    timestamp: datetime
    overall_status: str
    current_metrics: SystemHealthMetrics
    threshold_violations: List[str]
    authority_compliance: Dict[str, Any]
    tuning_recommendations: List[TuningRecommendation]
    alerts: List[HealthAlert]
    trend_analysis: Dict[str, Any]


# Global health monitor instance
global_health_monitor = HealthMonitor()


async def check_system_health() -> SystemHealthReport:
    """
    Quick health check function.
    
    Convenience function for one-time health checks.
    For continuous monitoring, use global_health_monitor.start_monitoring()
    """
    return await global_health_monitor.comprehensive_health_check()


def get_health_monitor() -> HealthMonitor:
    """
    Get global health monitor instance.
    Called by web/api.py for health status endpoints.
    """
    return global_health_monitor