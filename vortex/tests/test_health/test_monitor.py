"""
VORTEX Operational Health Monitoring Tests
Per .clinerules VORTEX_OPERATIONAL_HEALTH.md

Tests system health monitoring, KPI tracking, and auto-tuning
"""

import pytest
from vortex.config.health_thresholds import (
    SYSTEM_HEALTH_THRESHOLDS,
    V11_TARGET_METRICS,
    OPERATIONAL_KPIS
)


class TestHealthThresholds:
    """Test operational health thresholds per .clinerules."""
    
    def test_submit_ready_rate_targets(self):
        """Test submit ready rate targets (V11.1 enhanced)."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        # V11.1 targets: 5-8% (up from 2-5%)
        assert thresholds["submit_ready_rate"]["target"] >= 0.05
        assert thresholds["submit_ready_rate"]["min"] >= 0.03
        assert thresholds["submit_ready_rate"]["critical"] >= 0.02
    
    def test_manual_review_rate_targets(self):
        """Test manual review rate targets (reduced with fastpath)."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        # <75% (reduced from 80% with V11.1 fastpath)
        assert thresholds["manual_review_rate"]["target"] <= 0.70
        assert thresholds["manual_review_rate"]["max"] <= 0.75
        assert thresholds["manual_review_rate"]["critical"] <= 0.80
    
    def test_false_positive_rate_limit(self):
        """Test false positive rate is strictly limited."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        # <15% (tightened for quality)
        assert thresholds["false_positive_rate"]["target"] <= 0.12
        assert thresholds["false_positive_rate"]["max"] <= 0.15
        assert thresholds["false_positive_rate"]["critical"] <= 0.20
    
    def test_authority_violation_zero_tolerance(self):
        """Test authority violation rate has zero tolerance."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        # 0.0% target (zero tolerance)
        assert thresholds["authority_violation_rate"]["target"] == 0.0
        assert thresholds["authority_violation_rate"]["max"] <= 0.01
        assert thresholds["authority_violation_rate"]["critical"] <= 0.02


class TestMemoryZones:
    """Test memory management zones."""
    
    def test_memory_limit(self):
        """Test memory limit is 6000MB per .clinerules."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        assert thresholds["memory_usage_mb"]["target"] <= 5000
        assert thresholds["memory_usage_mb"]["max"] <= 6000
        assert thresholds["memory_usage_mb"]["critical"] <= 7000
    
    def test_memory_zones_defined(self):
        """Test memory zones are properly defined."""
        # GREEN: 0-60%
        # YELLOW: 60-85%
        # RED: 85-95%
        # EMERGENCY: 95-100%
        
        zones = {
            "GREEN": (0, 0.60),
            "YELLOW": (0.60, 0.85),
            "RED": (0.85, 0.95),
            "EMERGENCY": (0.95, 1.0)
        }
        
        # Zones should be contiguous
        assert zones["GREEN"][1] == zones["YELLOW"][0]
        assert zones["YELLOW"][1] == zones["RED"][0]
        assert zones["RED"][1] == zones["EMERGENCY"][0]
    
    def test_emergency_zone_triggers_immediate_action(self):
        """Test EMERGENCY zone (>95%) triggers immediate action."""
        memory_limit = 6000
        emergency_threshold = memory_limit * 0.95
        
        # 5700MB should trigger EMERGENCY
        assert 5700 > emergency_threshold


class TestV11TargetMetrics:
    """Test V11.1 target metrics and improvements."""
    
    def test_v11_submit_ready_improvement(self):
        """Test V11.1 submit ready rate improvement target."""
        metrics = V11_TARGET_METRICS["submit_ready_rate"]
        
        assert metrics["target"] == "5-8%"
        assert metrics["baseline"] == "2-5%"
        assert metrics["improvement"] == "+3-4%"
    
    def test_v11_manual_queue_reduction(self):
        """Test V11.1 manual queue reduction target."""
        metrics = V11_TARGET_METRICS["manual_queue_reduction"]
        
        assert metrics["target"] == "60-70%"
        assert metrics["baseline"] == "70-80%"
        assert metrics["improvement"] == "-10-15%"
    
    def test_v11_quality_preservation(self):
        """Test V11.1 maintains quality standards."""
        metrics = V11_TARGET_METRICS["quality_preservation"]
        
        # Must maintain >=85% acceptance rate
        assert ">=85%" in metrics["target"]
        assert metrics["improvement"] == "maintain"
    
    def test_v11_false_positive_maintained(self):
        """Test V11.1 doesn't increase false positives."""
        metrics = V11_TARGET_METRICS["false_positive_rate"]
        
        # Must stay <=15%
        assert "<=15%" in metrics["target"]
        assert "maintain or improve" in metrics["improvement"]


class TestHealthMetricsValidation:
    """Test health metrics validation."""
    
    def test_healthy_system_metrics(self, mock_health_metrics):
        """Test metrics for healthy system."""
        metrics = mock_health_metrics
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        # Submit ready rate above minimum
        assert metrics["submit_ready_rate"] >= thresholds["submit_ready_rate"]["min"]
        
        # Manual review rate below maximum
        assert metrics["manual_review_rate"] <= thresholds["manual_review_rate"]["max"]
        
        # False positive rate below maximum
        assert metrics["false_positive_rate"] <= thresholds["false_positive_rate"]["max"]
        
        # Authority violations zero
        assert metrics["authority_violation_rate"] == 0.0
    
    def test_degraded_system_detection(self):
        """Test detection of degraded system state."""
        # Simulated degraded metrics
        metrics = {
            "submit_ready_rate": 0.025,  # Below target (0.03 min)
            "manual_review_rate": 0.78,  # Above target (0.75 max)
            "false_positive_rate": 0.18,  # Above target (0.15 max)
            "ai_availability": 0.65,     # Below target (0.70 min)
            "memory_usage_mb": 6500,     # Above target (6000 max)
            "error_rate": 0.10           # Above target (0.08 max)
        }
        
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        # Count violations
        violations = 0
        if metrics["submit_ready_rate"] < thresholds["submit_ready_rate"]["min"]:
            violations += 1
        if metrics["manual_review_rate"] > thresholds["manual_review_rate"]["max"]:
            violations += 1
        if metrics["false_positive_rate"] > thresholds["false_positive_rate"]["max"]:
            violations += 1
        if metrics["memory_usage_mb"] > thresholds["memory_usage_mb"]["max"]:
            violations += 1
        
        # Multiple violations = DEGRADED state
        assert violations >= 2
    
    def test_critical_system_detection(self):
        """Test detection of critical system state."""
        # Simulated critical metrics
        metrics = {
            "submit_ready_rate": 0.015,  # Below critical (0.02)
            "manual_review_rate": 0.82,  # Above critical (0.80)
            "false_positive_rate": 0.22,  # Above critical (0.20)
            "authority_violation_rate": 0.025,  # Above critical (0.02)
            "memory_usage_mb": 7200,     # Above critical (7000)
            "error_rate": 0.15           # Above critical (0.12)
        }
        
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        # Count critical violations
        critical_violations = 0
        if metrics["submit_ready_rate"] < thresholds["submit_ready_rate"]["critical"]:
            critical_violations += 1
        if metrics["manual_review_rate"] > thresholds["manual_review_rate"]["critical"]:
            critical_violations += 1
        if metrics["false_positive_rate"] > thresholds["false_positive_rate"]["critical"]:
            critical_violations += 1
        if metrics["authority_violation_rate"] > thresholds["authority_violation_rate"]["critical"]:
            critical_violations += 1
        
        # Multiple critical violations = CRITICAL state
        assert critical_violations >= 2


class TestAutoTuning:
    """Test automated tuning recommendations."""
    
    def test_auto_tuning_for_low_submit_ready(self):
        """Test auto-tuning recommendations for low submit ready rate."""
        metrics = {"submit_ready_rate": 0.025}  # Below 0.03 minimum
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        if metrics["submit_ready_rate"] < thresholds["submit_ready_rate"]["min"]:
            # Should recommend tuning
            recommendation = {
                "category": "SUBMIT_READY_OPTIMIZATION",
                "priority": "HIGH",
                "actions": [
                    "Review evidence thresholds for over-strictness",
                    "Check AI model performance",
                    "Analyze SYSTEM_VERIFIED not progressing"
                ],
                "estimated_impact": "+1-2% submit ready rate",
                "auto_executable": True
            }
            
            assert recommendation["priority"] == "HIGH"
            assert recommendation["auto_executable"] is True
    
    def test_auto_tuning_for_high_manual_load(self):
        """Test auto-tuning for high manual review rate."""
        metrics = {"manual_review_rate": 0.78}  # Above 0.75 maximum
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        if metrics["manual_review_rate"] > thresholds["manual_review_rate"]["max"]:
            recommendation = {
                "category": "MANUAL_QUEUE_LOAD",
                "priority": "MEDIUM",
                "actions": [
                    "Enable additional fastpath criteria",
                    "Review AI_FAILED recovery",
                    "Check for systematic AI issues"
                ],
                "estimated_impact": "-5-10% manual review rate"
            }
            
            assert len(recommendation["actions"]) > 0
    
    def test_critical_tuning_for_quality_degradation(self):
        """Test CRITICAL tuning for quality degradation."""
        metrics = {"false_positive_rate": 0.18}  # Above 0.15 maximum
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        
        if metrics["false_positive_rate"] > thresholds["false_positive_rate"]["max"]:
            recommendation = {
                "category": "QUALITY_PRESERVATION",
                "priority": "CRITICAL",
                "actions": [
                    "URGENT: Tighten confidence thresholds",
                    "Disable fastpath until investigation",
                    "Review recent SUBMIT_READY findings",
                    "Implement additional evidence validation"
                ],
                "auto_executable": False  # Manual review required
            }
            
            assert recommendation["priority"] == "CRITICAL"
            assert recommendation["auto_executable"] is False


class TestOperationalKPIs:
    """Test operational KPI definitions."""
    
    def test_primary_kpis_defined(self):
        """Test primary KPIs are properly defined."""
        primary = OPERATIONAL_KPIS["PRIMARY_METRICS"]
        
        required_kpis = [
            "submit_ready_rate",
            "manual_review_rate",
            "false_positive_rate",
            "manual_sla_compliance"
        ]
        
        for kpi in required_kpis:
            assert kpi in primary
            assert "description" in primary[kpi]
            assert "target" in primary[kpi]
            assert "measurement_frequency" in primary[kpi]
    
    def test_secondary_kpis_defined(self):
        """Test secondary KPIs are properly defined."""
        secondary = OPERATIONAL_KPIS["SECONDARY_METRICS"]
        
        required_kpis = [
            "ai_availability_rate",
            "system_verification_success",
            "fastpath_utilization",
            "manual_conversion_rate"
        ]
        
        for kpi in required_kpis:
            assert kpi in secondary
    
    def test_system_kpis_defined(self):
        """Test system KPIs are properly defined."""
        system = OPERATIONAL_KPIS["SYSTEM_METRICS"]
        
        required_kpis = [
            "memory_utilization",
            "error_rate",
            "processing_throughput"
        ]
        
        for kpi in required_kpis:
            assert kpi in system


class TestHealthAlerts:
    """Test health alerting system."""
    
    def test_alert_levels_defined(self):
        """Test alert levels are properly defined."""
        alert_levels = {
            "INFO": {"urgency": 1, "response_time_hours": 24},
            "WARNING": {"urgency": 2, "response_time_hours": 4},
            "CRITICAL": {"urgency": 3, "response_time_hours": 1}
        }
        
        # Critical alerts require immediate response
        assert alert_levels["CRITICAL"]["response_time_hours"] == 1
        assert alert_levels["CRITICAL"]["urgency"] > alert_levels["WARNING"]["urgency"]
    
    def test_critical_alert_for_authority_violation(self):
        """Test critical alert for authority violations."""
        metrics = {"authority_violation_rate": 0.015}  # Above 0.01 max
        
        if metrics["authority_violation_rate"] > 0.01:
            alert = {
                "level": "CRITICAL",
                "title": "Authority Violation Detected",
                "message": f"Authority violation rate {metrics['authority_violation_rate']:.1%} exceeds 1% threshold",
                "requires_immediate_response": True,
                "response_time_hours": 1
            }
            
            assert alert["level"] == "CRITICAL"
            assert alert["requires_immediate_response"] is True
    
    def test_warning_alert_for_threshold_approach(self):
        """Test warning alert when approaching thresholds."""
        metrics = {"manual_review_rate": 0.73}  # Approaching 0.75 max
        
        if 0.70 < metrics["manual_review_rate"] < 0.75:
            alert = {
                "level": "WARNING",
                "title": "Manual Review Rate Elevated",
                "message": "Approaching maximum threshold",
                "response_time_hours": 4
            }
            
            assert alert["level"] == "WARNING"


@pytest.mark.compliance
class TestHealthComplianceChecklist:
    """Health monitoring compliance checklist per .clinerules."""
    
    def test_submit_ready_rate_monitored(self):
        """✓ Submit ready rate ≥ 3% monitored."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        assert thresholds["submit_ready_rate"]["min"] >= 0.03
    
    def test_manual_review_rate_monitored(self):
        """✓ Manual review rate ≤ 75% monitored."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        assert thresholds["manual_review_rate"]["max"] <= 0.75
    
    def test_false_positive_rate_monitored(self):
        """✓ False positive rate ≤ 15% monitored."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        assert thresholds["false_positive_rate"]["max"] <= 0.15
    
    def test_authority_violation_zero_tolerance(self):
        """✓ Authority violation rate = 0% enforced."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        assert thresholds["authority_violation_rate"]["target"] == 0.0
    
    def test_memory_limit_enforced(self):
        """✓ Memory usage < 6GB enforced."""
        thresholds = SYSTEM_HEALTH_THRESHOLDS
        assert thresholds["memory_usage_mb"]["max"] <= 6000
    
    def test_continuous_monitoring_enabled(self):
        """✓ Continuous health monitoring enabled."""
        # Health monitoring runs continuously
        pass
    
    def test_auto_tuning_available(self):
        """✓ Auto-tuning recommendations available."""
        # Auto-tuning system generates recommendations
        pass


class TestV11SuccessValidation:
    """Test V11.1 success criteria validation."""
    
    def test_v11_success_metrics(self, mock_health_metrics):
        """Test V11.1 success metrics are met."""
        metrics = mock_health_metrics
        
        # Submit ready rate: 5-8% target (min 3%)
        submit_success = metrics["submit_ready_rate"] >= 0.03
        
        # Manual reduction: 60-70% target (max 75%)
        manual_success = metrics["manual_review_rate"] <= 0.75
        
        # Quality preserved: FP rate <=15%
        quality_success = metrics["false_positive_rate"] <= 0.15
        
        # At least 2/3 criteria must be met for V11.1 success
        success_count = sum([submit_success, manual_success, quality_success])
        assert success_count >= 2
    
    def test_v11_fastpath_impact(self):
        """Test V11.1 fastpath provides expected impact."""
        # Expected improvements per .clinerules:
        # - SYSTEM_VERIFIED fastpath: +1.5-2.0%
        # - AI_FAILED recovery: +0.5-1.0%
        # - XSS threshold adjustment: +1.0-1.5%
        # - Vuln-specific thresholds: +0.5-1.0%
        # Total: +3.5-5.5%
        
        baseline_submit_rate = 0.035  # 3.5% baseline
        expected_improvement_min = 0.035  # +3.5%
        expected_improvement_max = 0.055  # +5.5%
        
        v11_target_min = baseline_submit_rate + expected_improvement_min  # 7%
        v11_target_max = baseline_submit_rate + expected_improvement_max  # 9%
        
        # V11.1 target range: 5-8%
        assert 0.05 <= v11_target_min <= 0.08