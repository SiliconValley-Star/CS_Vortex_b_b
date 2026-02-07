"""
VORTEX Monitoring Utilities - V17.0 ULTIMATE
System monitoring, metrics tracking, and performance measurement
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)

# Global metrics storage (in-memory, can be persisted to database)
_metrics_store = {}


def track_metric(metric_name: str, value: float, timestamp: Optional[datetime] = None):
    """
    Track a metric value.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        timestamp: Optional timestamp (default: now)
    """
    try:
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        if metric_name not in _metrics_store:
            _metrics_store[metric_name] = []
        
        _metrics_store[metric_name].append({
            'value': value,
            'timestamp': timestamp.isoformat()
        })
        
        # Keep only last 1000 entries per metric
        if len(_metrics_store[metric_name]) > 1000:
            _metrics_store[metric_name] = _metrics_store[metric_name][-1000:]
            
    except Exception as e:
        logger.error(f"Metric tracking error: {e}")


def log_event(event_name: str, details: Dict[str, Any]):
    """
    Log a system event.
    
    Args:
        event_name: Event name
        details: Event details
    """
    try:
        logger.info(f"EVENT: {event_name}", extra=details)
    except Exception as e:
        logger.error(f"Event logging error: {e}")


def measure_time(func):
    """
    Decorator to measure function execution time.
    
    Usage:
        @measure_time
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            metric_name = f"function_time.{func.__name__}"
            track_metric(metric_name, elapsed)
            
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper


def get_system_stats() -> Dict[str, Any]:
    """
    Get current system statistics.
    
    Returns:
        Dictionary of system stats
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        stats = {
            'cpu_percent': cpu_percent,
            'memory': {
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'used_mb': memory.used / (1024 * 1024),
                'percent': memory.percent
            },
            'disk': {
                'total_gb': disk.total / (1024 * 1024 * 1024),
                'used_gb': disk.used / (1024 * 1024 * 1024),
                'free_gb': disk.free / (1024 * 1024 * 1024),
                'percent': disk.percent
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return stats
    except Exception as e:
        logger.error(f"System stats error: {e}")
        return {}


def get_metric_summary(metric_name: str) -> Dict[str, Any]:
    """
    Get summary statistics for a metric.
    
    Args:
        metric_name: Name of metric
        
    Returns:
        Summary statistics
    """
    try:
        if metric_name not in _metrics_store or not _metrics_store[metric_name]:
            return {'error': 'Metric not found or empty'}
        
        values = [entry['value'] for entry in _metrics_store[metric_name]]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1],
            'first_timestamp': _metrics_store[metric_name][0]['timestamp'],
            'last_timestamp': _metrics_store[metric_name][-1]['timestamp']
        }
    except Exception as e:
        logger.error(f"Metric summary error: {e}")
        return {'error': str(e)}


def clear_metrics(metric_name: Optional[str] = None):
    """
    Clear metrics from memory.
    
    Args:
        metric_name: Specific metric to clear, or None for all
    """
    global _metrics_store
    try:
        if metric_name:
            if metric_name in _metrics_store:
                del _metrics_store[metric_name]
        else:
            _metrics_store = {}
    except Exception as e:
        logger.error(f"Metrics clear error: {e}")


class PerformanceTimer:
    """Context manager for measuring code block execution time."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        track_metric(f"block_time.{self.name}", self.elapsed)
        logger.debug(f"{self.name} took {self.elapsed:.3f}s")


class SystemMonitor:
    """
    System monitoring class for tracking system health and metrics.
    """
    
    def __init__(self, monitoring_config):
        self.config = monitoring_config
        self.monitoring_active = False
        self.start_time = None
        
        # Metrics
        self.metrics = {
            'uptime_seconds': 0,
            'total_requests': 0,
            'error_count': 0,
            'warning_count': 0
        }
    
    async def start(self):
        """Start system monitoring."""
        self.monitoring_active = True
        self.start_time = datetime.utcnow()
        logger.info("System monitoring started")
    
    async def stop(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        logger.info("System monitoring stopped")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        system_stats = get_system_stats()
        
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            self.metrics['uptime_seconds'] = uptime
        
        return {
            'monitoring': {
                'healthy': self.monitoring_active,
                'status': 'active' if self.monitoring_active else 'stopped',
                'details': f"Uptime: {self.metrics['uptime_seconds']:.0f}s"
            },
            'system': {
                'healthy': system_stats.get('cpu_percent', 0) < 80,
                'status': 'operational',
                'details': f"CPU: {system_stats.get('cpu_percent', 0):.1f}%"
            },
            'memory': {
                'healthy': system_stats.get('memory', {}).get('percent', 0) < 85,
                'status': 'operational',
                'details': f"Memory: {system_stats.get('memory', {}).get('percent', 0):.1f}%"
            }
        }
    
    def record_request(self):
        """Record a request."""
        self.metrics['total_requests'] += 1
    
    def record_error(self):
        """Record an error."""
        self.metrics['error_count'] += 1
    
    def record_warning(self):
        """Record a warning."""
        self.metrics['warning_count'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()


__all__ = [
    'SystemMonitor',
    'track_metric',
    'log_event',
    'measure_time',
    'get_system_stats',
    'get_metric_summary',
    'clear_metrics',
    'PerformanceTimer',
]