"""
VORTEX Performance Metrics System
Comprehensive metrics collection and monitoring

FEATURES:
- Request latency tracking
- Database query timing
- AI API call monitoring
- Memory allocation tracking
- Prometheus-compatible metrics
- Real-time statistics
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistical summary of metrics."""
    count: int
    total: float
    mean: float
    median: float
    min: float
    max: float
    p95: float
    p99: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'count': self.count,
            'total': self.total,
            'mean': self.mean,
            'median': self.median,
            'min': self.min,
            'max': self.max,
            'p95': self.p95,
            'p99': self.p99
        }


class MetricsCollector:
    """
    Central metrics collection system.
    
    Tracks various performance metrics across the system.
    """
    
    def __init__(self, retention_minutes: int = 60):
        self.retention_minutes = retention_minutes
        self.retention_seconds = retention_minutes * 60
        
        # Metric storage (metric_name -> list of points)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Counters (monotonically increasing)
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Gauges (current value)
        self.gauges: Dict[str, float] = {}
        
        # Histograms (distribution tracking)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics cache
        self._stats_cache: Dict[str, MetricStats] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self.cache_ttl_seconds = 60
        
        logger.info("Metrics collector initialized")
    
    def record_timing(self, metric_name: str, duration_seconds: float, 
                     labels: Optional[Dict[str, str]] = None):
        """
        Record a timing metric.
        
        Args:
            metric_name: Name of the metric
            duration_seconds: Duration in seconds
            labels: Optional labels for the metric
        """
        point = MetricPoint(
            value=duration_seconds,
            labels=labels or {}
        )
        
        self.metrics[metric_name].append(point)
        self.histograms[metric_name].append(duration_seconds)
        
        # Cleanup old metrics
        self._cleanup_old_metrics(metric_name)
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """
        Increment a counter.
        
        Args:
            counter_name: Name of the counter
            value: Amount to increment
        """
        self.counters[counter_name] += value
    
    def set_gauge(self, gauge_name: str, value: float):
        """
        Set a gauge value.
        
        Args:
            gauge_name: Name of the gauge
            value: Current value
        """
        self.gauges[gauge_name] = value
    
    def get_stats(self, metric_name: str, force_refresh: bool = False) -> Optional[MetricStats]:
        """
        Get statistical summary for a metric.
        
        Args:
            metric_name: Name of the metric
            force_refresh: Force refresh of cached stats
            
        Returns:
            Statistical summary or None if no data
        """
        # Check cache
        if not force_refresh and metric_name in self._stats_cache:
            expiry = self._cache_expiry.get(metric_name)
            if expiry and datetime.utcnow() < expiry:
                return self._stats_cache[metric_name]
        
        # Calculate stats
        values = [p.value for p in self.metrics[metric_name]]
        
        if not values:
            return None
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        stats = MetricStats(
            count=count,
            total=sum(sorted_values),
            mean=statistics.mean(sorted_values),
            median=statistics.median(sorted_values),
            min=min(sorted_values),
            max=max(sorted_values),
            p95=sorted_values[int(count * 0.95)] if count > 1 else sorted_values[0],
            p99=sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0]
        )
        
        # Cache results
        self._stats_cache[metric_name] = stats
        self._cache_expiry[metric_name] = datetime.utcnow() + timedelta(seconds=self.cache_ttl_seconds)
        
        return stats
    
    def get_counter(self, counter_name: str) -> int:
        """Get current counter value."""
        return self.counters.get(counter_name, 0)
    
    def get_gauge(self, gauge_name: str) -> Optional[float]:
        """Get current gauge value."""
        return self.gauges.get(gauge_name)
    
    def get_recent_values(self, metric_name: str, count: int = 100) -> List[float]:
        """Get recent metric values."""
        points = list(self.metrics[metric_name])[-count:]
        return [p.value for p in points]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics summary."""
        summary = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timings': {}
        }
        
        for metric_name in self.metrics.keys():
            stats = self.get_stats(metric_name)
            if stats:
                summary['timings'][metric_name] = stats.to_dict()
        
        return summary
    
    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than retention period."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.retention_seconds)
        
        metrics = self.metrics[metric_name]
        while metrics and metrics[0].timestamp < cutoff:
            metrics.popleft()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self._stats_cache.clear()
        self._cache_expiry.clear()
        logger.info("All metrics reset")


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, 
                 labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            self.duration = time.time() - self.start_time
            self.collector.record_timing(self.metric_name, self.duration, self.labels)
        return False


class PerformanceMetrics:
    """
    High-level performance metrics tracking.
    
    Specialized metrics for different subsystems:
    - Request latency
    - Database queries
    - AI API calls
    - Scanner operations
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
    
    # Request Metrics
    
    def record_request(self, url: str, duration: float, status_code: int):
        """Record HTTP request metrics."""
        self.collector.record_timing(
            'http_request_duration',
            duration,
            {'status_code': str(status_code)}
        )
        self.collector.increment_counter('http_requests_total')
        
        if status_code >= 400:
            self.collector.increment_counter('http_requests_errors')
    
    # Database Metrics
    
    def record_db_query(self, query_type: str, duration: float):
        """Record database query metrics."""
        self.collector.record_timing(
            'db_query_duration',
            duration,
            {'query_type': query_type}
        )
        self.collector.increment_counter('db_queries_total')
    
    def record_db_connection(self, pool_size: int, active: int):
        """Record database connection pool metrics."""
        self.collector.set_gauge('db_pool_size', float(pool_size))
        self.collector.set_gauge('db_pool_active', float(active))
    
    # AI Metrics
    
    def record_ai_call(self, provider: str, model: str, duration: float, 
                      tokens: Optional[int] = None):
        """Record AI API call metrics."""
        self.collector.record_timing(
            'ai_call_duration',
            duration,
            {'provider': provider, 'model': model}
        )
        self.collector.increment_counter('ai_calls_total')
        
        if tokens:
            self.collector.increment_counter('ai_tokens_total', tokens)
    
    def record_ai_error(self, provider: str, error_type: str):
        """Record AI API error."""
        self.collector.increment_counter(
            'ai_errors_total',
            1
        )
    
    # Scanner Metrics
    
    def record_scan(self, scanner_type: str, duration: float, findings_count: int):
        """Record scanner operation metrics."""
        self.collector.record_timing(
            'scan_duration',
            duration,
            {'scanner_type': scanner_type}
        )
        self.collector.increment_counter('scans_total')
        self.collector.increment_counter('findings_detected', findings_count)
    
    # Memory Metrics
    
    def record_memory_usage(self, memory_mb: float, zone: str):
        """Record memory usage metrics."""
        self.collector.set_gauge('memory_usage_mb', memory_mb)
        self.collector.set_gauge('memory_zone', float(ord(zone[0])))  # Encode zone as number
    
    # Performance Summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'requests': {
                'total': self.collector.get_counter('http_requests_total'),
                'errors': self.collector.get_counter('http_requests_errors'),
                'latency': self.collector.get_stats('http_request_duration')
            },
            'database': {
                'queries_total': self.collector.get_counter('db_queries_total'),
                'query_latency': self.collector.get_stats('db_query_duration'),
                'pool_size': self.collector.get_gauge('db_pool_size'),
                'pool_active': self.collector.get_gauge('db_pool_active')
            },
            'ai': {
                'calls_total': self.collector.get_counter('ai_calls_total'),
                'errors_total': self.collector.get_counter('ai_errors_total'),
                'tokens_total': self.collector.get_counter('ai_tokens_total'),
                'call_duration': self.collector.get_stats('ai_call_duration')
            },
            'scans': {
                'total': self.collector.get_counter('scans_total'),
                'findings_detected': self.collector.get_counter('findings_detected'),
                'scan_duration': self.collector.get_stats('scan_duration')
            },
            'memory': {
                'usage_mb': self.collector.get_gauge('memory_usage_mb'),
                'zone': self.collector.get_gauge('memory_zone')
            }
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, value in self.collector.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in self.collector.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Histograms (simplified)
        for metric_name in self.collector.metrics.keys():
            stats = self.collector.get_stats(metric_name)
            if stats:
                lines.append(f"# TYPE {metric_name} summary")
                lines.append(f"{metric_name}_count {stats.count}")
                lines.append(f"{metric_name}_sum {stats.total}")
                lines.append(f"{metric_name}{{quantile=\"0.5\"}} {stats.median}")
                lines.append(f"{metric_name}{{quantile=\"0.95\"}} {stats.p95}")
                lines.append(f"{metric_name}{{quantile=\"0.99\"}} {stats.p99}")
        
        return "\n".join(lines)


# Global metrics instance
global_metrics = PerformanceMetrics()