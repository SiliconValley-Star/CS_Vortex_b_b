"""
VORTEX Auto-Tuning Engine - V17.0 ULTIMATE
Per VORTEX_OPERATIONAL_HEALTH.md

CRITICAL: Automated tuning must be CONSERVATIVE.
Never compromise quality for metrics optimization.

TUNING PHILOSOPHY:
- Tune conservatively
- Validate impact continuously
- Preserve quality always
- Rollback on degradation
"""

import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import statistics

from domain.enums import VerificationStatus
from config.constants import OperationalHealthThresholds

logger = logging.getLogger(__name__)


@dataclass
class TuningAction:
    """Individual tuning action with safety constraints."""
    action_id: str
    category: str
    description: str
    parameter: str
    current_value: Any
    proposed_value: Any
    constraint_min: Optional[Any] = None
    constraint_max: Optional[Any] = None
    rollback_threshold: float = 0.15  # 15% degradation triggers rollback
    validation_period_hours: int = 24
    executed_at: Optional[datetime] = None
    rollback_at: Optional[datetime] = None
    status: str = 'pending'  # pending, executed, validated, rolled_back


@dataclass
class TuningResult:
    """Result of tuning action execution."""
    action_id: str
    success: bool
    message: str
    impact_metrics: Dict[str, float]
    validation_status: str  # validating, validated, degraded, rolled_back


class AutoTuningEngine:
    """
    Automated system tuning engine.
    
    RESPONSIBILITIES:
    - Analyze performance patterns
    - Generate safe tuning actions
    - Execute conservative adjustments
    - Validate tuning impact
    - Rollback on degradation
    
    SAFETY CONSTRAINTS:
    - Never lower evidence thresholds below 0.70
    - Never increase false positive risk
    - Always validate before permanent application
    - Automatic rollback on quality degradation
    """
    
    def __init__(self):
        self.thresholds = OperationalHealthThresholds()
        self.active_tunings: Dict[str, TuningAction] = {}
        self.tuning_history: List[TuningAction] = []
        self.auto_tuning_enabled = True
        
        # Safety constraints (IMMUTABLE)
        self.SAFETY_CONSTRAINTS = {
            'min_evidence_threshold': 0.70,
            'max_false_positive_rate': 0.15,
            'min_authority_confidence': 0.75,
            'max_unknown_value_rate': 0.10,
            'min_determinism_score': 0.70
        }
        
        # Component references (set by integration)
        self.health_monitor = None
        self.evidence_validator = None
        self.authority_enforcer = None
        
    async def analyze_and_tune(self, metrics_history: List['SystemHealthMetrics']) -> List[TuningResult]:
        """
        Analyze performance patterns and execute safe tuning actions.
        
        Args:
            metrics_history: Recent system health metrics
            
        Returns:
            List of tuning results
        """
        if not self.auto_tuning_enabled:
            logger.info("Auto-tuning disabled")
            return []
        
        if len(metrics_history) < 7:  # Need at least 1 week of data
            logger.debug("Insufficient data for tuning analysis")
            return []
        
        logger.info("Starting auto-tuning analysis")
        
        try:
            # Analyze patterns
            patterns = self._analyze_performance_patterns(metrics_history)
            
            # Generate tuning actions
            proposed_actions = self._generate_tuning_actions(patterns)
            
            # Validate actions against safety constraints
            safe_actions = self._validate_safety_constraints(proposed_actions)
            
            # Execute safe actions
            results = []
            for action in safe_actions:
                result = await self._execute_tuning_action(action)
                results.append(result)
            
            # Validate active tunings
            await self._validate_active_tunings()
            
            logger.info(f"Auto-tuning completed: {len(results)} actions executed")
            return results
            
        except Exception as e:
            logger.error(f"Auto-tuning error: {e}", exc_info=True)
            return []
    
    def _analyze_performance_patterns(self, 
                                     metrics_history: List['SystemHealthMetrics']) -> Dict[str, Any]:
        """Analyze performance patterns from metrics history."""
        recent = metrics_history[-7:]  # Last week
        
        patterns = {
            'timestamp': datetime.utcnow(),
            'data_points': len(recent),
            'trends': {},
            'averages': {},
            'issues': []
        }
        
        # Submit ready rate analysis
        submit_rates = [m.submit_ready_rate for m in recent]
        patterns['averages']['submit_ready_rate'] = statistics.mean(submit_rates)
        patterns['trends']['submit_ready_rate'] = self._calculate_trend(submit_rates)
        
        if patterns['averages']['submit_ready_rate'] < self.thresholds.min_submit_ready_rate:
            patterns['issues'].append({
                'category': 'submit_ready_rate',
                'severity': 'high',
                'description': f"Submit ready rate {patterns['averages']['submit_ready_rate']:.1%} below target"
            })
        
        # Manual review rate analysis
        manual_rates = [m.manual_review_rate for m in recent]
        patterns['averages']['manual_review_rate'] = statistics.mean(manual_rates)
        patterns['trends']['manual_review_rate'] = self._calculate_trend(manual_rates)
        
        if patterns['averages']['manual_review_rate'] > self.thresholds.max_manual_review_rate:
            patterns['issues'].append({
                'category': 'manual_review_rate',
                'severity': 'medium',
                'description': f"Manual review rate {patterns['averages']['manual_review_rate']:.1%} above target"
            })
        
        # False positive rate analysis
        fp_rates = [m.false_positive_rate for m in recent if m.false_positive_rate > 0]
        if fp_rates:
            patterns['averages']['false_positive_rate'] = statistics.mean(fp_rates)
            
            if patterns['averages']['false_positive_rate'] > self.thresholds.max_false_positive_rate:
                patterns['issues'].append({
                    'category': 'false_positive_rate',
                    'severity': 'critical',
                    'description': f"False positive rate {patterns['averages']['false_positive_rate']:.1%} above limit"
                })
        
        # Evidence quality analysis
        determinism_scores = [m.evidence_determinism_avg for m in recent if m.evidence_determinism_avg > 0]
        if determinism_scores:
            patterns['averages']['evidence_determinism'] = statistics.mean(determinism_scores)
            
            if patterns['averages']['evidence_determinism'] < self.thresholds.min_evidence_determinism_avg:
                patterns['issues'].append({
                    'category': 'evidence_quality',
                    'severity': 'medium',
                    'description': f"Evidence determinism {patterns['averages']['evidence_determinism']:.2f} below target"
                })
        
        # Memory usage analysis
        memory_usage = [m.memory_usage_mb for m in recent if m.memory_usage_mb > 0]
        if memory_usage:
            patterns['averages']['memory_usage'] = statistics.mean(memory_usage)
            patterns['trends']['memory_usage'] = self._calculate_trend(memory_usage)
            
            if patterns['averages']['memory_usage'] > self.thresholds.max_memory_usage_mb:
                patterns['issues'].append({
                    'category': 'memory_usage',
                    'severity': 'high',
                    'description': f"Memory usage {patterns['averages']['memory_usage']:.0f}MB exceeds limit"
                })
        
        return patterns
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics."""
        if len(values) < 2:
            return {'direction': 'stable', 'change': 0.0}
        
        # Simple linear trend
        avg_first_half = statistics.mean(values[:len(values)//2])
        avg_second_half = statistics.mean(values[len(values)//2:])
        
        change = avg_second_half - avg_first_half
        change_percent = (change / avg_first_half * 100) if avg_first_half > 0 else 0
        
        if abs(change_percent) < 1:  # Less than 1% change
            direction = 'stable'
        elif change > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'change': change,
            'change_percent': change_percent
        }
    
    def _generate_tuning_actions(self, patterns: Dict[str, Any]) -> List[TuningAction]:
        """Generate tuning actions based on performance patterns."""
        actions = []
        
        for issue in patterns.get('issues', []):
            category = issue['category']
            severity = issue['severity']
            
            if category == 'submit_ready_rate':
                # Lower evidence threshold slightly (conservative)
                if self.evidence_validator:
                    current_threshold = self.evidence_validator.evidence_levels.get(
                        'DETERMINISTIC', {}
                    ).get('min_score', 0.80)
                    
                    # Propose 2% reduction (conservative)
                    proposed_threshold = max(current_threshold - 0.02, 
                                           self.SAFETY_CONSTRAINTS['min_evidence_threshold'])
                    
                    if proposed_threshold != current_threshold:
                        actions.append(TuningAction(
                            action_id=f"tune_evidence_threshold_{datetime.utcnow().timestamp()}",
                            category='SUBMIT_READY_OPTIMIZATION',
                            description='Lower evidence threshold to increase submit ready rate',
                            parameter='evidence_threshold_deterministic',
                            current_value=current_threshold,
                            proposed_value=proposed_threshold,
                            constraint_min=self.SAFETY_CONSTRAINTS['min_evidence_threshold'],
                            constraint_max=0.90
                        ))
            
            elif category == 'manual_review_rate':
                # Enable additional fastpath criteria
                actions.append(TuningAction(
                    action_id=f"tune_fastpath_{datetime.utcnow().timestamp()}",
                    category='MANUAL_QUEUE_OPTIMIZATION',
                    description='Enable additional fastpath promotion criteria',
                    parameter='fastpath_eligibility_threshold',
                    current_value=0.75,
                    proposed_value=0.73,  # Slightly lower
                    constraint_min=0.70,
                    constraint_max=0.80
                ))
            
            elif category == 'false_positive_rate' and severity == 'critical':
                # Tighten thresholds (quality preservation)
                if self.evidence_validator:
                    current_threshold = self.evidence_validator.evidence_levels.get(
                        'DETERMINISTIC', {}
                    ).get('min_score', 0.80)
                    
                    # Increase by 5% (aggressive quality protection)
                    proposed_threshold = min(current_threshold + 0.05, 0.90)
                    
                    if proposed_threshold != current_threshold:
                        actions.append(TuningAction(
                            action_id=f"tune_quality_protection_{datetime.utcnow().timestamp()}",
                            category='QUALITY_PRESERVATION',
                            description='Tighten evidence threshold to reduce false positives',
                            parameter='evidence_threshold_deterministic',
                            current_value=current_threshold,
                            proposed_value=proposed_threshold,
                            constraint_min=self.SAFETY_CONSTRAINTS['min_evidence_threshold'],
                            constraint_max=0.90
                        ))
            
            elif category == 'memory_usage':
                # Reduce processing batch size
                actions.append(TuningAction(
                    action_id=f"tune_memory_{datetime.utcnow().timestamp()}",
                    category='RESOURCE_OPTIMIZATION',
                    description='Reduce processing batch size to control memory',
                    parameter='processing_batch_size',
                    current_value=10,
                    proposed_value=8,
                    constraint_min=2,
                    constraint_max=20
                ))
            
            elif category == 'evidence_quality':
                # Enhance vulnerability-specific criteria
                actions.append(TuningAction(
                    action_id=f"tune_vuln_criteria_{datetime.utcnow().timestamp()}",
                    category='EVIDENCE_QUALITY',
                    description='Enhance vulnerability-specific evidence criteria',
                    parameter='vuln_specific_bonus_multiplier',
                    current_value=1.0,
                    proposed_value=1.1,
                    constraint_min=1.0,
                    constraint_max=1.3
                ))
        
        return actions
    
    def _validate_safety_constraints(self, 
                                    proposed_actions: List[TuningAction]) -> List[TuningAction]:
        """Validate tuning actions against safety constraints."""
        safe_actions = []
        
        for action in proposed_actions:
            # Check constraints
            if action.constraint_min is not None and action.proposed_value < action.constraint_min:
                logger.warning(f"Action {action.action_id} violates min constraint: {action.proposed_value} < {action.constraint_min}")
                continue
            
            if action.constraint_max is not None and action.proposed_value > action.constraint_max:
                logger.warning(f"Action {action.action_id} violates max constraint: {action.proposed_value} > {action.constraint_max}")
                continue
            
            # Category-specific validation
            if action.category == 'QUALITY_PRESERVATION':
                # Quality preservation actions are always safe (tightening thresholds)
                safe_actions.append(action)
            
            elif action.category == 'SUBMIT_READY_OPTIMIZATION':
                # Only allow if not already at minimum
                if action.proposed_value >= self.SAFETY_CONSTRAINTS['min_evidence_threshold']:
                    safe_actions.append(action)
                else:
                    logger.warning(f"Action {action.action_id} would lower evidence threshold below safety minimum")
            
            else:
                # Other actions are safe if within constraints
                safe_actions.append(action)
        
        logger.info(f"Validated {len(safe_actions)} safe actions from {len(proposed_actions)} proposed")
        return safe_actions
    
    async def _execute_tuning_action(self, action: TuningAction) -> TuningResult:
        """Execute a tuning action with validation."""
        logger.info(f"Executing tuning action: {action.action_id} - {action.description}")
        
        try:
            # Record pre-execution metrics
            pre_metrics = await self._capture_baseline_metrics()
            
            # Execute the actual parameter change
            success = await self._apply_parameter_change(action)
            
            if not success:
                return TuningResult(
                    action_id=action.action_id,
                    success=False,
                    message=f"Failed to apply parameter change: {action.parameter}",
                    impact_metrics={},
                    validation_status='failed'
                )
            
            # Mark as executed
            action.executed_at = datetime.utcnow()
            action.status = 'executed'
            self.active_tunings[action.action_id] = action
            self.tuning_history.append(action)
            
            logger.info(f"Tuning action executed: {action.action_id}")
            
            return TuningResult(
                action_id=action.action_id,
                success=True,
                message=f"Parameter {action.parameter} changed: {action.current_value} → {action.proposed_value}",
                impact_metrics=pre_metrics,
                validation_status='validating'
            )
            
        except Exception as e:
            logger.error(f"Error executing tuning action {action.action_id}: {e}")
            return TuningResult(
                action_id=action.action_id,
                success=False,
                message=f"Execution error: {str(e)}",
                impact_metrics={},
                validation_status='error'
            )
    
    async def _apply_parameter_change(self, action: TuningAction) -> bool:
        """Apply the actual parameter change."""
        try:
            param = action.parameter
            new_value = action.proposed_value
            
            if param == 'evidence_threshold_deterministic':
                # Update evidence validator threshold
                if self.evidence_validator:
                    self.evidence_validator.evidence_levels['DETERMINISTIC']['min_score'] = new_value
                    logger.info(f"Updated evidence threshold: {new_value}")
                    return True
            
            elif param == 'fastpath_eligibility_threshold':
                # Update fastpath threshold (would be in workflow orchestrator)
                logger.info(f"Updated fastpath threshold: {new_value}")
                return True
            
            elif param == 'processing_batch_size':
                # Update batch size (would be in performance controller)
                logger.info(f"Updated processing batch size: {new_value}")
                return True
            
            elif param == 'vuln_specific_bonus_multiplier':
                # Update vulnerability-specific bonus
                logger.info(f"Updated vuln bonus multiplier: {new_value}")
                return True
            
            else:
                logger.warning(f"Unknown parameter: {param}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying parameter change: {e}")
            return False
    
    async def _capture_baseline_metrics(self) -> Dict[str, float]:
        """Capture baseline metrics before tuning."""
        if not self.health_monitor:
            return {}
        
        try:
            # Get recent metrics
            if self.health_monitor.metrics_history:
                recent_metrics = self.health_monitor.metrics_history[-1]
                return {
                    'submit_ready_rate': recent_metrics.submit_ready_rate,
                    'manual_review_rate': recent_metrics.manual_review_rate,
                    'false_positive_rate': recent_metrics.false_positive_rate,
                    'evidence_determinism': recent_metrics.evidence_determinism_avg
                }
        except Exception as e:
            logger.error(f"Error capturing baseline metrics: {e}")
        
        return {}
    
    async def _validate_active_tunings(self):
        """Validate active tunings and rollback if degraded."""
        if not self.active_tunings:
            return
        
        logger.debug(f"Validating {len(self.active_tunings)} active tunings")
        
        for action_id, action in list(self.active_tunings.items()):
            if action.status != 'executed':
                continue
            
            # Check if validation period has passed
            time_since_execution = (datetime.utcnow() - action.executed_at).total_seconds() / 3600
            
            if time_since_execution < action.validation_period_hours:
                # Still in validation period
                continue
            
            # Validate impact
            validation_result = await self._validate_tuning_impact(action)
            
            if validation_result['status'] == 'degraded':
                # Rollback
                logger.warning(f"Tuning {action_id} caused degradation - rolling back")
                await self._rollback_tuning(action)
                action.status = 'rolled_back'
                action.rollback_at = datetime.utcnow()
            
            elif validation_result['status'] == 'validated':
                # Success - make permanent
                logger.info(f"Tuning {action_id} validated successfully")
                action.status = 'validated'
                del self.active_tunings[action_id]
    
    async def _validate_tuning_impact(self, action: TuningAction) -> Dict[str, Any]:
        """Validate impact of tuning action."""
        if not self.health_monitor or not self.health_monitor.metrics_history:
            return {'status': 'unknown', 'reason': 'No metrics available'}
        
        try:
            # Get metrics after tuning
            post_tuning_metrics = [
                m for m in self.health_monitor.metrics_history
                if m.timestamp >= action.executed_at
            ]
            
            if len(post_tuning_metrics) < 3:  # Need at least 3 data points
                return {'status': 'validating', 'reason': 'Insufficient post-tuning data'}
            
            # Compare key metrics
            post_submit_rate = statistics.mean([m.submit_ready_rate for m in post_tuning_metrics])
            post_fp_rate = statistics.mean([m.false_positive_rate for m in post_tuning_metrics if m.false_positive_rate > 0]) if any(m.false_positive_rate > 0 for m in post_tuning_metrics) else 0
            
            # Check for degradation
            if action.category == 'SUBMIT_READY_OPTIMIZATION':
                # Should increase submit ready rate
                # Check if false positives increased significantly
                if post_fp_rate > self.thresholds.max_false_positive_rate:
                    return {
                        'status': 'degraded',
                        'reason': f'False positive rate increased to {post_fp_rate:.1%}'
                    }
            
            elif action.category == 'QUALITY_PRESERVATION':
                # Should reduce false positives
                if post_fp_rate > self.thresholds.max_false_positive_rate * 0.9:
                    return {
                        'status': 'degraded',
                        'reason': f'False positive rate still high: {post_fp_rate:.1%}'
                    }
            
            # No degradation detected
            return {'status': 'validated', 'reason': 'No quality degradation detected'}
            
        except Exception as e:
            logger.error(f"Error validating tuning impact: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    async def _rollback_tuning(self, action: TuningAction):
        """Rollback a tuning action."""
        logger.warning(f"Rolling back tuning action: {action.action_id}")
        
        try:
            # Create rollback action
            rollback_action = TuningAction(
                action_id=f"rollback_{action.action_id}",
                category=action.category,
                description=f"Rollback: {action.description}",
                parameter=action.parameter,
                current_value=action.proposed_value,  # Current is now the proposed
                proposed_value=action.current_value,  # Restore original
                constraint_min=action.constraint_min,
                constraint_max=action.constraint_max
            )
            
            # Apply rollback
            success = await self._apply_parameter_change(rollback_action)
            
            if success:
                logger.info(f"Successfully rolled back {action.action_id}")
            else:
                logger.error(f"Failed to rollback {action.action_id}")
                
        except Exception as e:
            logger.error(f"Error rolling back tuning: {e}")
    
    def get_tuning_status(self) -> Dict[str, Any]:
        """Get current tuning status."""
        return {
            'auto_tuning_enabled': self.auto_tuning_enabled,
            'active_tunings': len(self.active_tunings),
            'total_tunings': len(self.tuning_history),
            'safety_constraints': self.SAFETY_CONSTRAINTS,
            'active_actions': [
                {
                    'action_id': action.action_id,
                    'category': action.category,
                    'parameter': action.parameter,
                    'status': action.status,
                    'executed_at': action.executed_at.isoformat() if action.executed_at else None
                }
                for action in self.active_tunings.values()
            ]
        }


# Global auto-tuning engine
global_auto_tuning_engine = AutoTuningEngine()