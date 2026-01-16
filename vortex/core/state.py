"""
VORTEX State Management - V17.0 ULTIMATE
Global state management for runtime configuration

FEATURES:
- Runtime configuration
- Feature flags
- Performance metrics
- Resource tracking
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Global system state."""
    
    # Operational state
    initialized: bool = False
    running: bool = False
    shutdown_requested: bool = False
    
    # Component status
    database_connected: bool = False
    network_client_active: bool = False
    health_monitoring_active: bool = False
    
    # Performance metrics
    total_scans: int = 0
    active_scans: int = 0
    total_findings: int = 0
    submit_ready_findings: int = 0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Timestamps
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        'auto_tuning_enabled': True,
        'health_monitoring_enabled': True,
        'ai_advisory_enabled': True,
        'authority_enforcement_enabled': True,
        'fastpath_enabled': True
    })


class StateManager:
    """
    Thread-safe global state manager.
    
    Provides centralized access to system state and configuration.
    """
    
    def __init__(self):
        self._state = SystemState()
        self._lock = Lock()
    
    def get_state(self) -> SystemState:
        """Get current system state (copy)."""
        with self._lock:
            # Return a copy to prevent external modification
            return SystemState(
                initialized=self._state.initialized,
                running=self._state.running,
                shutdown_requested=self._state.shutdown_requested,
                database_connected=self._state.database_connected,
                network_client_active=self._state.network_client_active,
                health_monitoring_active=self._state.health_monitoring_active,
                total_scans=self._state.total_scans,
                active_scans=self._state.active_scans,
                total_findings=self._state.total_findings,
                submit_ready_findings=self._state.submit_ready_findings,
                memory_usage_mb=self._state.memory_usage_mb,
                cpu_usage_percent=self._state.cpu_usage_percent,
                started_at=self._state.started_at,
                last_health_check=self._state.last_health_check,
                config=self._state.config.copy(),
                features=self._state.features.copy()
            )
    
    def update_state(self, **kwargs):
        """Update state attributes."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
                else:
                    logger.warning(f"Unknown state attribute: {key}")
    
    def set_config(self, key: str, value: Any):
        """Set configuration value."""
        with self._lock:
            self._state.config[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        with self._lock:
            return self._state.config.get(key, default)
    
    def enable_feature(self, feature: str):
        """Enable feature flag."""
        with self._lock:
            if feature in self._state.features:
                self._state.features[feature] = True
                logger.info(f"Feature enabled: {feature}")
    
    def disable_feature(self, feature: str):
        """Disable feature flag."""
        with self._lock:
            if feature in self._state.features:
                self._state.features[feature] = False
                logger.info(f"Feature disabled: {feature}")
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if feature is enabled."""
        with self._lock:
            return self._state.features.get(feature, False)
    
    def increment_metric(self, metric: str, amount: int = 1):
        """Increment metric counter."""
        with self._lock:
            if hasattr(self._state, metric):
                current = getattr(self._state, metric)
                if isinstance(current, int):
                    setattr(self._state, metric, current + amount)
    
    def decrement_metric(self, metric: str, amount: int = 1):
        """Decrement metric counter."""
        with self._lock:
            if hasattr(self._state, metric):
                current = getattr(self._state, metric)
                if isinstance(current, int):
                    setattr(self._state, metric, max(0, current - amount))
    
    def reset_metrics(self):
        """Reset all metrics to zero."""
        with self._lock:
            self._state.total_scans = 0
            self._state.active_scans = 0
            self._state.total_findings = 0
            self._state.submit_ready_findings = 0
            logger.info("Metrics reset")


# Global state manager instance
global_state_manager = StateManager()


# Helper function for web interface
def get_global_state() -> SystemState:
    """Get current global state (web interface helper)."""
    return global_state_manager.get_state()