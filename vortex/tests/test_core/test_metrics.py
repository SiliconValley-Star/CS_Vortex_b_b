"""
Test suite for Performance Metrics System
Tests metrics collection, aggregation, and Prometheus export
"""

import pytest
from unittest.mock import Mock, patch
import time

from core.metrics import PerformanceMetrics, MetricsCollector, global_metrics


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return PerformanceMetrics()
    
    def test_metrics_initialization(self, metrics):
        """Test metrics initializes correctly."""
        assert metrics.request_count == 0
        assert metrics.total_duration == 0.0
        assert metrics.avg_duration == 0.0
        assert metrics.min_duration == float('inf')
        assert metrics.max_duration == 0.0
    
    def test_add_duration(self, metrics):
        """Test adding duration updates metrics."""
        metrics.add_duration(1.0)
        
        assert metrics.request_count == 1
        assert metrics.total_duration == 1.0
        assert metrics.avg_duration == 1.0
        assert metrics.min_duration == 1.0
        assert metrics.max_duration == 1.0
    
    def test_add_multiple_durations(self, metrics):
        """Test multiple duration additions."""
        metrics.add_duration(1.0)
        metrics.add_duration(2.0)
        metrics.add_duration(3.0)
        
        assert metrics.request_count == 3
        assert metrics.total_duration == 6.0
        assert metrics.avg_duration == 2.0
        assert metrics.min_duration == 1.0
        assert metrics.max_duration == 3.0
    
    def test_to_dict(self, metrics):
        """Test metrics conversion to dict."""
        metrics.add_duration(1.5)
        metrics.add_duration(2.5)
        
        data = metrics.to_dict()
        
        assert isinstance(data, dict)
        assert data['count'] == 2
        assert data['total'] == 4.0
        assert data['avg'] == 2.0
        assert data['min'] == 1.5
        assert data['max'] == 2.5


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    @pytest.fixture
    def collector(self):
        """Create metrics collector instance."""
        return MetricsCollector()
    
    def test_collector_initialization(self, collector):
        """Test collector initializes correctly."""
        assert isinstance(collector.requests, dict)
        assert isinstance(collector.db_queries, dict)
        assert isinstance(collector.ai_calls, dict)
        assert isinstance(collector.scans, dict)
    
    def test_record_request(self, collector):
        """Test HTTP request recording."""
        collector.record_request('https://example.com', 1.5, 200)
        
        assert '200' in collector.requests
        assert collector.requests['200'].request_count == 1
        assert collector.requests['200'].total_duration == 1.5
    
    def test_record_multiple_requests(self, collector):
        """Test multiple request recordings."""
        collector.record_request('https://example.com/1', 1.0, 200)
        collector.record_request('https://example.com/2', 2.0, 200)
        collector.record_request('https://example.com/3', 0.5, 404)
        
        assert collector.requests['200'].request_count == 2
        assert collector.requests['404'].request_count == 1
    
    def test_record_db_query(self, collector):
        """Test database query recording."""
        collector.record_db_query('SELECT', 0.5)
        collector.record_db_query('SELECT', 0.7)
        collector.record_db_query('INSERT', 1.2)
        
        assert collector.db_queries['SELECT'].request_count == 2
        assert collector.db_queries['INSERT'].request_count == 1
    
    def test_record_ai_call(self, collector):
        """Test AI call recording."""
        collector.record_ai_call('openrouter', 'claude-4', 2.5, 1000)
        
        key = 'openrouter:claude-4'
        assert key in collector.ai_calls
        assert collector.ai_calls[key].request_count == 1
        assert collector.ai_calls[key].total_duration == 2.5
    
    def test_record_scan(self, collector):
        """Test scan recording."""
        collector.record_scan('sqli', 5.0, 3)
        collector.record_scan('xss', 3.5, 1)
        
        assert collector.scans['sqli'].request_count == 1
        assert collector.scans['xss'].request_count == 1
    
    def test_get_summary(self, collector):
        """Test summary generation."""
        collector.record_request('https://example.com', 1.0, 200)
        collector.record_db_query('SELECT', 0.5)
        collector.record_ai_call('openrouter', 'claude-4', 2.0, 500)
        collector.record_scan('sqli', 3.0, 2)
        
        summary = collector.get_summary()
        
        assert isinstance(summary, dict)
        assert 'requests' in summary
        assert 'db_queries' in summary
        assert 'ai_calls' in summary
        assert 'scans' in summary
    
    def test_export_prometheus(self, collector):
        """Test Prometheus format export."""
        collector.record_request('https://example.com', 1.0, 200)
        collector.record_request('https://example.com', 1.5, 200)
        
        prometheus_output = collector.export_prometheus()
        
        assert isinstance(prometheus_output, str)
        assert 'http_requests_total' in prometheus_output
        assert 'http_request_duration_seconds' in prometheus_output
    
    def test_reset(self, collector):
        """Test metrics reset."""
        collector.record_request('https://example.com', 1.0, 200)
        collector.record_db_query('SELECT', 0.5)
        
        collector.reset()
        
        assert len(collector.requests) == 0
        assert len(collector.db_queries) == 0
        assert len(collector.ai_calls) == 0
        assert len(collector.scans) == 0
    
    def test_global_metrics(self):
        """Test global metrics instance."""
        assert global_metrics is not None
        assert isinstance(global_metrics, MetricsCollector)


class TestMetricsIntegration:
    """Integration tests for metrics system."""
    
    def test_concurrent_metric_recording(self):
        """Test concurrent metric recording."""
        collector = MetricsCollector()
        
        # Simulate concurrent operations
        for i in range(100):
            collector.record_request(f'https://example.com/{i}', 0.1 * i, 200)
        
        assert collector.requests['200'].request_count == 100
    
    def test_metrics_accuracy(self):
        """Test metrics calculation accuracy."""
        collector = MetricsCollector()
        
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        for duration in durations:
            collector.record_request('https://example.com', duration, 200)
        
        metrics = collector.requests['200']
        
        assert metrics.request_count == 5
        assert metrics.total_duration == 15.0
        assert metrics.avg_duration == 3.0
        assert metrics.min_duration == 1.0
        assert metrics.max_duration == 5.0
    
    def test_prometheus_export_format(self):
        """Test Prometheus export format validity."""
        collector = MetricsCollector()
        
        collector.record_request('https://example.com', 1.0, 200)
        collector.record_request('https://example.com', 2.0, 404)
        collector.record_db_query('SELECT', 0.5)
        
        output = collector.export_prometheus()
        
        # Check for required Prometheus format elements
        assert '# HELP' in output
        assert '# TYPE' in output
        assert '{' in output
        assert '}' in output