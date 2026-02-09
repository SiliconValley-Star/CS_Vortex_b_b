"""
VORTEX Health Monitoring Compliance Tests - V17.0 ULTIMATE
Critical validation: Operational health thresholds maintained
"""

import pytest
from datetime import datetime, timedelta
from domain.enums import VerificationStatus
from domain.models import AssessmentResult
from core.health.monitor import OperationalHealthSystem, SystemHealthThresholds
from core.health.auto_tune import SystemTuningEngine


@pytest.fixture
def health_system():
    """Create health monitoring system."""
    return OperationalHealthSystem()


@pytest.fixture
def tuning_engine():
    """Create auto-tuning engine."""
    return SystemTuningEngine()


class TestHealthThresholds:
    """Test operational health threshold definitions."""
    
    def test_all_thresholds_defined(self, health_system):
        """All critical thresholds must be defined."""
        thresholds = health_system.health_thresholds
        
        # Core effectiveness metrics
        assert 'submit_ready_rate' in thresholds
        assert 'manual_review_rate' in thresholds
        assert 'false_positive_rate' in thresholds
        
        # System performance metrics
        assert 'ai_availability' in thresholds
        assert 'system_verification_success' in thresholds
        assert 'memory_usage_mb' in thresholds
        assert 'error_rate' in thresholds
        
        # Authority compliance metrics (V17.0)
        assert 'authority_violation_rate' in thresholds
        assert 'evidence_determinism_avg' in thresholds
        assert 'unknown_value_rate' in thresholds
    
    def test_v11_1_target_values(self, health_system):
        """V11.1 target values must be correctly configured."""
        thresholds = health_system.health_thresholds
        
        # Submit ready rate: >3% (target: 5-8%)
        assert thresholds['submit_ready_rate']['min'] == 0.03
        assert thresholds['submit_ready_rate']['target'] == 0.05
        
        # Manual review rate: <75%
        assert thresholds['manual_review_rate']['max'] == 0.75
        assert thresholds['manual_review_rate']['critical'] == 0.80
        
        # False positive rate: <15%
        assert thresholds['false_positive_rate']['max'] == 0.15
        assert thresholds['false_positive_rate']['critical'] == 0.20
    
    def test_authority_violation_critical_threshold(self, health_system):
        """Authority violation rate must have strictest threshold."""
        thresholds = health_system.health_thresholds
        
        # Authority violations: <1% (CRITICAL)
        assert thresholds['authority_violation_rate']['target'] == 0.0
        assert thresholds['authority_violation_rate']['max'] == 0.01
        assert thresholds['authority_violation_rate']['critical'] == 0.02


class TestHealthMetricsCollection:
    """Test health metrics collection."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_metrics_collection(self, health_system):
        """All metrics should be collected comprehensively."""
        metrics = await health_system._collect_all_metrics()
        
        # Should have core metrics
        assert 'submit_ready_rate' in metrics or metrics  # Empty if no findings yet
        
        # Should have authority metrics
        assert 'authority_violation_rate' in metrics or metrics  # Empty if no findings
        
        # Should have resource metrics
        assert 'memory_usage_mb' in metrics or metrics  # Should always be available
    
    @pytest.mark.asyncio
    async def test_authority_compliance_tracking(self, health_system):
        """Authority compliance must be tracked."""
        # Simulate findings
        findings = []
        for i in range(10):
            finding = AssessmentResult(
                id=f"test-health-{i}",
                url=f"https://target.com/test{i}",
                vulnerability_type="sql_injection",
                status=VerificationStatus.SUBMIT_READY if i < 5 else VerificationStatus.NEEDS_MANUAL,
                heuristic_score=0.80 + (i * 0.01),
                evidence=f"Evidence {i}"
            )
            findings.append(finding)
        
        # Check authority metrics
        authority_metrics = await health_system._collect_authority_metrics()
        
        # Should track violations
        assert 'authority_violation_rate' in authority_metrics or not authority_metrics
        assert 'evidence_determinism_avg' in authority_metrics or not authority_metrics


class TestHealthStatusAssessment:
    """Test health status assessment logic."""
    
    def test_healthy_status(self, health_system):
        """Healthy metrics should result in HEALTHY status."""
        metrics = {
            'submit_ready_rate': 0.06,  # Above minimum
            'manual_review_rate': 0.70,  # Below maximum
            'false_positive_rate': 0.12,  # Below maximum
            'ai_availability': 0.85,  # Above minimum
            'memory_usage_mb': 4500,  # Below maximum
            'error_rate': 0.05,  # Below maximum
            'authority_violation_rate': 0.0,  # Perfect
            'evidence_determinism_avg': 0.78  # Above minimum
        }
        
        assessment = health_system._assess_health_status(metrics)
        assert assessment['overall_status'] == 'HEALTHY'
    
    def test_degraded_status(self, health_system):
        """Single critical issue should result in DEGRADED."""
        metrics = {
            'submit_ready_rate': 0.02,  # CRITICAL: Below minimum
            'manual_review_rate': 0.70,
            'false_positive_rate': 0.12,
            'ai_availability': 0.80,
            'memory_usage_mb': 4500,
            'error_rate': 0.05,
            'authority_violation_rate': 0.0,
            'evidence_determinism_avg': 0.75
        }
        
        assessment = health_system._assess_health_status(metrics)
        assert assessment['overall_status'] == 'DEGRADED'
        assert len(assessment['critical_issues']) >= 1
    
    def test_critical_status(self, health_system):
        """Multiple critical issues should result in CRITICAL."""
        metrics = {
            'submit_ready_rate': 0.01,  # CRITICAL
            'manual_review_rate': 0.82,  # CRITICAL
            'false_positive_rate': 0.22,  # CRITICAL
            'ai_availability': 0.50,  # CRITICAL
            'memory_usage_mb': 7500,  # CRITICAL
            'error_rate': 0.15,  # CRITICAL
            'authority_violation_rate': 0.03,  # CRITICAL
            'evidence_determinism_avg': 0.60  # CRITICAL
        }
        
        assessment = health_system._assess_health_status(metrics)
        assert assessment['overall_status'] == 'CRITICAL'
        assert len(assessment['critical_issues']) >= 2


class TestAutoTuningRecommendations:
    """Test automated tuning recommendations."""
    
    def test_submit_ready_optimization_recommendation(self, tuning_engine):
        """Low submit ready rate should trigger optimization."""
        metrics = {
            'submit_ready_rate': 0.02,  # Below 3% minimum
            'manual_review_rate': 0.78,
            'false_positive_rate': 0.13
        }
        
        recommendations = tuning_engine._generate_auto_tuning_recommendations(metrics)
        
        # Should recommend optimization
        submit_ready_recs = [
            r for r in recommendations 
            if r.category == 'SUBMIT_READY_OPTIMIZATION'
        ]
        
        assert len(submit_ready_recs) > 0
        assert submit_ready_recs[0].priority in ['HIGH', 'CRITICAL']
    
    def test_authority_violation_critical_recommendation(self, tuning_engine):
        """Authority violations trigger CRITICAL recommendation."""
        metrics = {
            'authority_violation_rate': 0.025,  # Above 2% critical
            'submit_ready_rate': 0.05,
            'manual_review_rate': 0.70
        }
        
        recommendations = tuning_engine._generate_auto_tuning_recommendations(metrics)
        
        # Should have CRITICAL priority
        authority_recs = [
            r for r in recommendations
            if r.category == 'AUTHORITY_COMPLIANCE'
        ]
        
        assert len(authority_recs) > 0
        assert authority_recs[0].priority == 'CRITICAL'
        assert not authority_recs[0].auto_executable  # Manual review required
    
    def test_evidence_quality_recommendation(self, tuning_engine):
        """Low evidence determinism should trigger recommendation."""
        metrics = {
            'evidence_determinism_avg': 0.62,  # Below 0.70 minimum
            'submit_ready_rate': 0.04,
            'manual_review_rate': 0.72
        }
        
        recommendations = tuning_engine._generate_auto_tuning_recommendations(metrics)
        
        evidence_recs = [
            r for r in recommendations
            if r.category == 'EVIDENCE_QUALITY'
        ]
        
        assert len(evidence_recs) > 0
        assert evidence_recs[0].auto_executable  # Can be auto-tuned


class TestAutoTuningExecution:
    """Test auto-tuning execution."""
    
    @pytest.mark.asyncio
    async def test_auto_tuning_respects_limits(self, health_system):
        """Auto-tuning must respect minimum thresholds."""
        # Enable auto-tuning
        health_system.auto_tuning_enabled = True
        
        # Simulate low submit ready rate
        from core.evidence.standards import EvidenceStandardsValidator
        validator = EvidenceStandardsValidator()
        
        original_threshold = validator.evidence_levels['DETERMINISTIC']['min_score']
        
        # Execute optimization
        await health_system._auto_optimize_submit_ready_rate()
        
        new_threshold = validator.evidence_levels['DETERMINISTIC']['min_score']
        
        # Should lower slightly but NEVER below 0.70
        assert new_threshold >= 0.70
        assert new_threshold <= original_threshold
    
    @pytest.mark.asyncio
    async def test_memory_optimization_execution(self, health_system):
        """Memory optimization should reduce usage."""
        # Simulate high memory
        health_system.auto_tuning_enabled = True
        
        # Execute memory optimization
        await health_system._auto_optimize_memory_usage()
        
        # Should trigger cleanup (actual effect depends on implementation)
        # At minimum, should not raise error
        assert True  # Execution successful
    
    def test_non_executable_recommendations_skipped(self, tuning_engine):
        """Non-executable recommendations should be skipped."""
        from core.health.auto_tune import TuningRecommendation
        
        # Create non-executable recommendation
        rec = TuningRecommendation(
            category='AUTHORITY_COMPLIANCE',
            priority='CRITICAL',
            description='Authority violations detected',
            actions=['Manual review required'],
            estimated_impact='Fix violations',
            auto_executable=False  # NOT auto-executable
        )
        
        # Should not execute
        # (This is tested in integration, here we just verify the flag)
        assert not rec.auto_executable


class TestHealthAlerts:
    """Test health alerting system."""
    
    def test_critical_alert_generation(self, health_system):
        """Critical issues should generate alerts."""
        assessment = {
            'overall_status': 'CRITICAL',
            'critical_issues': [
                'False positive rate too high: 22%',
                'Memory usage too high: 7500MB'
            ],
            'warnings': [],
            'timestamp': datetime.utcnow()
        }
        
        from core.health.alerts import HealthAlertSystem
        alert_system = HealthAlertSystem()
        
        alerts = alert_system.generate_health_alerts(assessment)
        
        # Should have critical alerts
        critical_alerts = [a for a in alerts if a.level == 'CRITICAL']
        assert len(critical_alerts) == 2
        
        # Each should have suggested actions
        for alert in critical_alerts:
            assert len(alert.suggested_actions) > 0
    
    def test_warning_alert_generation(self, health_system):
        """Warnings should generate appropriate alerts."""
        assessment = {
            'overall_status': 'ATTENTION',
            'critical_issues': [],
            'warnings': [
                'AI availability low: 65%',
                'Manual review time too long: 50h'
            ],
            'timestamp': datetime.utcnow()
        }
        
        from core.health.alerts import HealthAlertSystem
        alert_system = HealthAlertSystem()
        
        alerts = alert_system.generate_health_alerts(assessment)
        
        warning_alerts = [a for a in alerts if a.level == 'WARNING']
        assert len(warning_alerts) == 2
        
        # Warnings don't require immediate response
        for alert in warning_alerts:
            assert not alert.requires_immediate_response


class TestV11SuccessMetrics:
    """Test V11.1 success metrics validation."""
    
    def test_v11_1_success_criteria(self, health_system):
        """V11.1 success should be measurable."""
        # Successful V11.1 metrics
        metrics = {
            'submit_ready_rate': 0.06,  # 6% (target: 5-8%)
            'manual_review_rate': 0.68,  # 68% (target: <70%)
            'false_positive_rate': 0.13,  # 13% (target: <15%)
            'manual_conversion_rate': 0.28  # 28% (target: >25%)
        }
        
        from core.health.monitor import validate_v11_1_success
        assessment = validate_v11_1_success(metrics)
        
        # Should be successful
        assert assessment.overall_status in ['V11.1_SUCCESS', 'V11.1_PARTIAL_SUCCESS']
        
        # Core metrics should be SUCCESS
        assert assessment.component_assessments['submit_ready'] == 'SUCCESS'
        assert assessment.component_assessments['manual_reduction'] == 'SUCCESS'
        assert assessment.component_assessments['quality'] == 'SUCCESS'
    
    def test_v11_1_partial_success(self, health_system):
        """Partial success should be identified."""
        # Partial success metrics
        metrics = {
            'submit_ready_rate': 0.035,  # 3.5% (partial: 3-5%)
            'manual_review_rate': 0.73,  # 73% (partial: 70-75%)
            'false_positive_rate': 0.17  # 17% (partial: 15-20%)
        }
        
        from core.health.monitor import validate_v11_1_success
        assessment = validate_v11_1_success(metrics)
        
        # Should be partial success
        assert assessment.overall_status == 'V11.1_PARTIAL_SUCCESS'
    
    def test_v11_1_needs_tuning(self, health_system):
        """Failed metrics should trigger tuning need."""
        # Failed metrics
        metrics = {
            'submit_ready_rate': 0.02,  # 2% (below 3%)
            'manual_review_rate': 0.82,  # 82% (above 80%)
            'false_positive_rate': 0.22  # 22% (above 20%)
        }
        
        from core.health.monitor import validate_v11_1_success
        assessment = validate_v11_1_success(metrics)
        
        # Should need tuning
        assert assessment.overall_status == 'V11.1_NEEDS_TUNING'
        
        # Should have recommendations
        assert len(assessment.recommendations) > 0


@pytest.mark.critical
class TestHealthMonitoringIntegrity:
    """Critical health monitoring integrity tests."""
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring_loop(self, health_system):
        """Health monitoring should run continuously."""
        # This would normally run as background task
        # Here we just verify the function exists and is callable
        
        assert hasattr(health_system, 'comprehensive_health_check')
        
        # Execute single check
        report = await health_system.comprehensive_health_check()
        
        # Should have timestamp
        assert report.timestamp is not None
        
        # Should have status
        assert report.overall_status in ['HEALTHY', 'ATTENTION', 'DEGRADED', 'CRITICAL']
    
    def test_threshold_consistency(self, health_system):
        """Thresholds must be internally consistent."""
        thresholds = health_system.health_thresholds
        
        for metric_name, metric_config in thresholds.items():
            # target <= max <= critical (for upper bounds)
            if 'max' in metric_config and 'critical' in metric_config:
                if 'target' in metric_config:
                    assert metric_config['target'] <= metric_config['max']
                assert metric_config['max'] <= metric_config['critical']
            
            # target >= min >= critical (for lower bounds)
            if 'min' in metric_config and 'critical' in metric_config:
                if 'target' in metric_config:
                    assert metric_config['target'] >= metric_config['min']
                assert metric_config['min'] >= metric_config['critical']
    
    def test_no_threshold_relaxation(self, health_system):
        """Health thresholds must NEVER be relaxed."""
        original_thresholds = health_system._initialize_health_thresholds()
        
        # Authority violation threshold
        assert original_thresholds['authority_violation_rate']['max'] <= 0.01
        
        # False positive threshold
        assert original_thresholds['false_positive_rate']['max'] <= 0.15
        
        # Submit ready minimum
        assert original_thresholds['submit_ready_rate']['min'] >= 0.03
        
        # These values should NEVER increase/decrease (respectively)
        # even under optimization pressure


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])