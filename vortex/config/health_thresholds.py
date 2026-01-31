"""
VORTEX Health Monitoring Thresholds - V17.0 ULTIMATE
System health monitoring configuration and thresholds

Per .clinerules VORTEX_OPERATIONAL_HEALTH.md:
- System health limits
- Performance thresholds
- Alert triggers
- Auto-tuning parameters

CONFIGURATION:
- Finding distribution thresholds
- Performance metrics
- Resource limits
- SLA requirements
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HealthThresholds:
    """
    System health thresholds configuration.
    
    Per .clinerules VORTEX_OPERATIONAL_HEALTH.md:
    These thresholds ensure system remains viable and effective.
    """
    
    # ============================================================================
    # FINDING DISTRIBUTION THRESHOLDS (V11.1 adjusted for fastpath)
    # ============================================================================
    
    # Submit ready rate (increased target post-fastpath)
    submit_ready_rate_target: float = 0.05       # 5% target
    submit_ready_rate_min: float = 0.03          # 3% minimum
    submit_ready_rate_critical: float = 0.02     # 2% critical threshold
    
    # Manual review rate (reduced target post-fastpath)
    manual_review_rate_target: float = 0.70      # 70% target
    manual_review_rate_max: float = 0.75         # 75% maximum
    manual_review_rate_critical: float = 0.80    # 80% critical threshold
    
    # False positive rate (tightened for quality)
    false_positive_rate_target: float = 0.12     # 12% target
    false_positive_rate_max: float = 0.15        # 15% maximum
    false_positive_rate_critical: float = 0.20   # 20% critical threshold
    
    # ============================================================================
    # MANUAL REVIEW EFFICIENCY THRESHOLDS
    # ============================================================================
    
    # Average review time
    manual_review_hours_target: float = 24.0     # 24h target
    manual_review_hours_max: float = 48.0        # 48h maximum
    manual_review_hours_critical: float = 72.0   # 72h critical
    
    # Manual conversion rate (reviews resulting in submission)
    manual_conversion_target: float = 0.30       # 30% target
    manual_conversion_min: float = 0.25          # 25% minimum
    manual_conversion_critical: float = 0.20     # 20% critical
    
    # Queue growth rate
    queue_growth_rate_max: float = 0.15          # 15% daily growth maximum
    queue_growth_rate_critical: float = 0.25     # 25% critical
    
    # SLA compliance
    manual_sla_compliance_target: float = 0.85   # 85% target
    manual_sla_compliance_min: float = 0.80      # 80% minimum
    manual_sla_compliance_critical: float = 0.70  # 70% critical
    
    # ============================================================================
    # SYSTEM PERFORMANCE THRESHOLDS
    # ============================================================================
    
    # AI availability
    ai_availability_target: float = 0.80         # 80% target
    ai_availability_min: float = 0.70            # 70% minimum
    ai_availability_critical: float = 0.60       # 60% critical
    
    # System verification success rate
    verification_success_target: float = 0.70    # 70% target
    verification_success_min: float = 0.60       # 60% minimum
    verification_success_critical: float = 0.50  # 50% critical
    
    # System accuracy (verification correctness)
    system_accuracy_target: float = 0.85         # 85% target
    system_accuracy_min: float = 0.80            # 80% minimum
    system_accuracy_critical: float = 0.75       # 75% critical
    
    # ============================================================================
    # RESOURCE LIMITS
    # ============================================================================
    
    # Memory usage (per .clinerules memory_manager.py)
    memory_usage_target_mb: float = 5000.0       # 5GB target
    memory_usage_max_mb: float = 6000.0          # 6GB maximum
    memory_usage_critical_mb: float = 7000.0     # 7GB critical (emergency cleanup)
    
    # Error rate
    error_rate_target: float = 0.05              # 5% target
    error_rate_max: float = 0.08                 # 8% maximum (tightened)
    error_rate_critical: float = 0.12            # 12% critical
    
    # Processing throughput
    throughput_target_rps: float = 2.0           # 2 requests/second target
    throughput_min_rps: float = 1.0              # 1 request/second minimum
    throughput_critical_rps: float = 0.5         # 0.5 request/second critical
    
    # ============================================================================
    # AUTHORITY COMPLIANCE THRESHOLDS (NEW - per .clinerules)
    # ============================================================================
    
    # Authority violation rate
    authority_violation_target: float = 0.0      # 0% target (zero tolerance)
    authority_violation_max: float = 0.01        # 1% maximum
    authority_violation_critical: float = 0.02   # 2% critical
    
    # Evidence determinism average
    evidence_determinism_target: float = 0.75    # 75% target
    evidence_determinism_min: float = 0.70       # 70% minimum
    evidence_determinism_critical: float = 0.65  # 65% critical
    
    # Unknown value rate
    unknown_value_target: float = 0.05           # 5% target
    unknown_value_max: float = 0.10              # 10% maximum
    unknown_value_critical: float = 0.15         # 15% critical


class HealthThresholdsConfig:
    """
    Health thresholds configuration manager.
    
    Provides thresholds and validation for system health monitoring.
    """
    
    # Default thresholds instance
    DEFAULT = HealthThresholds()
    
    # ============================================================================
    # V11.1 TARGET METRICS (post-fastpath implementation)
    # ============================================================================
    
    V11_1_TARGETS = {
        'submit_ready_rate': {
            'target': '5-8%',
            'baseline': '2-5%',
            'improvement': '+3-4%',
            'measurement': 'Weekly average of findings reaching SUBMIT_READY status'
        },
        'manual_queue_reduction': {
            'target': '60-70%',
            'baseline': '70-80%',
            'improvement': '-10-15%',
            'measurement': 'Percentage of findings requiring manual review'
        },
        'quality_preservation': {
            'target': '>=85%',
            'baseline': '85-90%',
            'improvement': 'maintain',
            'measurement': 'Bug bounty platform acceptance rate'
        },
        'false_positive_rate': {
            'target': '<=15%',
            'baseline': '12-18%',
            'improvement': 'maintain or improve',
            'measurement': 'Findings determined false positive after review'
        },
        'manual_conversion_rate': {
            'target': '>=25%',
            'baseline': '20-30%',
            'improvement': 'maintain or improve',
            'measurement': 'Manual reviews resulting in submissions'
        }
    }
    
    # ============================================================================
    # ALERT TRIGGER CONDITIONS
    # ============================================================================
    
    ALERT_TRIGGERS = {
        'CRITICAL': {
            'submit_ready_rate_below': 0.02,      # Below 2%
            'manual_review_rate_above': 0.80,     # Above 80%
            'false_positive_rate_above': 0.20,    # Above 20%
            'memory_usage_above_mb': 7000,        # Above 7GB
            'error_rate_above': 0.12,             # Above 12%
            'authority_violations_above': 0.02,    # Above 2%
            'overdue_reviews_above': 10           # More than 10 overdue
        },
        'WARNING': {
            'submit_ready_rate_below': 0.03,      # Below 3%
            'manual_review_rate_above': 0.75,     # Above 75%
            'false_positive_rate_above': 0.15,    # Above 15%
            'memory_usage_above_mb': 6000,        # Above 6GB
            'error_rate_above': 0.08,             # Above 8%
            'ai_availability_below': 0.70,        # Below 70%
            'manual_review_hours_above': 48,      # Above 48h
            'evidence_determinism_below': 0.70    # Below 70%
        },
        'INFO': {
            'submit_ready_rate_approaching_target': 0.045,  # Approaching 5%
            'manual_review_rate_approaching_target': 0.72,  # Approaching 70%
            'performance_trend_positive': True
        }
    }
    
    # ============================================================================
    # AUTO-TUNING PARAMETERS
    # ============================================================================
    
    AUTO_TUNING = {
        'enabled': True,
        'conservative_adjustment': True,         # Make small adjustments
        'max_threshold_adjustment': 0.02,        # Maximum 2% adjustment
        'min_data_points': 7,                    # Need 7 days of data
        'adjustment_frequency_hours': 24,        # Adjust daily at most
        
        # Adjustment triggers
        'trigger_submit_ready_low': {
            'condition': 'submit_ready_rate < 0.03',
            'action': 'lower_evidence_threshold',
            'max_adjustment': -0.02
        },
        'trigger_false_positive_high': {
            'condition': 'false_positive_rate > 0.15',
            'action': 'raise_confidence_threshold',
            'max_adjustment': +0.02
        },
        'trigger_memory_pressure': {
            'condition': 'memory_usage > 6000MB',
            'action': 'reduce_batch_size',
            'adjustment': -2
        }
    }
    
    # ============================================================================
    # SLA REQUIREMENTS
    # ============================================================================
    
    SLA_REQUIREMENTS = {
        'detection_to_ai_analysis_seconds': 300,     # 5 minutes
        'ai_analysis_seconds': 30,                   # 30 seconds average
        'system_verification_seconds': 60,           # 60 seconds average
        'manual_review_registration_seconds': 5,     # 5 seconds
        'total_automated_workflow_seconds': 600,     # 10 minutes
        
        # Manual review SLAs by priority
        'manual_review_sla': {
            'priority_1': {'hours': 24, 'escalation_hours': 12},
            'priority_2': {'hours': 48, 'escalation_hours': 24},
            'priority_3': {'hours': 72, 'escalation_hours': 48},
            'priority_4': {'hours': 120, 'escalation_hours': 72},
            'priority_5': {'hours': 168, 'escalation_hours': 120}
        }
    }
    
    # ============================================================================
    # PERFORMANCE BENCHMARKS
    # ============================================================================
    
    PERFORMANCE_BENCHMARKS = {
        'excellent': {
            'submit_ready_rate': 0.08,           # 8%
            'manual_review_rate': 0.60,          # 60%
            'false_positive_rate': 0.10,         # 10%
            'manual_conversion_rate': 0.35,      # 35%
            'system_accuracy': 0.90              # 90%
        },
        'good': {
            'submit_ready_rate': 0.05,           # 5%
            'manual_review_rate': 0.70,          # 70%
            'false_positive_rate': 0.12,         # 12%
            'manual_conversion_rate': 0.30,      # 30%
            'system_accuracy': 0.85              # 85%
        },
        'acceptable': {
            'submit_ready_rate': 0.03,           # 3%
            'manual_review_rate': 0.75,          # 75%
            'false_positive_rate': 0.15,         # 15%
            'manual_conversion_rate': 0.25,      # 25%
            'system_accuracy': 0.80              # 80%
        },
        'needs_improvement': {
            'submit_ready_rate': 0.02,           # 2%
            'manual_review_rate': 0.80,          # 80%
            'false_positive_rate': 0.20,         # 20%
            'manual_conversion_rate': 0.20,      # 20%
            'system_accuracy': 0.75              # 75%
        }
    }
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def get_threshold(metric_name: str, level: str = 'target') -> float:
        """
        Get threshold value for metric.
        
        Args:
            metric_name: Metric name (e.g., 'submit_ready_rate')
            level: Threshold level ('target', 'min/max', 'critical')
            
        Returns:
            Threshold value
        """
        thresholds = HealthThresholdsConfig.DEFAULT
        attr_name = f"{metric_name}_{level}"
        return getattr(thresholds, attr_name, 0.0)
    
    @staticmethod
    def assess_metric_status(metric_name: str, current_value: float) -> str:
        """
        Assess metric status against thresholds.
        
        Args:
            metric_name: Metric name
            current_value: Current metric value
            
        Returns:
            Status: 'EXCELLENT', 'GOOD', 'ACCEPTABLE', 'WARNING', 'CRITICAL'
        """
        # Get threshold levels
        target = HealthThresholdsConfig.get_threshold(metric_name, 'target')
        
        # Try to get max/min
        try:
            max_threshold = HealthThresholdsConfig.get_threshold(metric_name, 'max')
            critical = HealthThresholdsConfig.get_threshold(metric_name, 'critical')
        except:
            try:
                max_threshold = HealthThresholdsConfig.get_threshold(metric_name, 'min')
                critical = HealthThresholdsConfig.get_threshold(metric_name, 'critical')
            except:
                return 'UNKNOWN'
        
        # Determine if higher or lower is better based on metric name
        higher_is_better = any(x in metric_name for x in ['submit_ready', 'conversion', 'availability', 'accuracy', 'compliance'])
        
        if higher_is_better:
            if current_value >= target * 1.1:
                return 'EXCELLENT'
            elif current_value >= target:
                return 'GOOD'
            elif current_value >= max_threshold:
                return 'ACCEPTABLE'
            elif current_value >= critical:
                return 'WARNING'
            else:
                return 'CRITICAL'
        else:
            if current_value <= target * 0.9:
                return 'EXCELLENT'
            elif current_value <= target:
                return 'GOOD'
            elif current_value <= max_threshold:
                return 'ACCEPTABLE'
            elif current_value <= critical:
                return 'WARNING'
            else:
                return 'CRITICAL'
    
    @staticmethod
    def get_v11_1_target(metric_name: str) -> Dict[str, str]:
        """Get V11.1 target for metric."""
        return HealthThresholdsConfig.V11_1_TARGETS.get(metric_name, {})
    
    @staticmethod
    def validate_configuration() -> tuple[bool, List[str]]:
        """
        Validate health thresholds configuration.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        thresholds = HealthThresholdsConfig.DEFAULT
        
        # Validate submit ready rate
        if thresholds.submit_ready_rate_min < 0.03:
            errors.append("Submit ready rate minimum must be >= 3% per .clinerules")
        
        # Validate manual review rate
        if thresholds.manual_review_rate_max > 0.75:
            errors.append("Manual review rate maximum should be <= 75% per .clinerules")
        
        # Validate false positive rate
        if thresholds.false_positive_rate_max > 0.15:
            errors.append("False positive rate maximum should be <= 15% per .clinerules")
        
        # Validate memory limits
        if thresholds.memory_usage_max_mb > 6000:
            errors.append("Memory usage maximum should be <= 6000MB per .clinerules")
        
        # Validate authority violation tolerance
        if thresholds.authority_violation_target != 0.0:
            errors.append("Authority violation target must be 0% (zero tolerance) per .clinerules")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.error(f"Health thresholds validation FAILED: {errors}")
        else:
            logger.info("Health thresholds validated successfully")
        
        return is_valid, errors


# Validate configuration on import
_is_valid, _errors = HealthThresholdsConfig.validate_configuration()
if not _is_valid:
    logger.warning(f"Health thresholds validation warnings: {_errors}")
    # Don't raise - these are operational thresholds that can be adjusted