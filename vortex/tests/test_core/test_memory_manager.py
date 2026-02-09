"""
Test suite for Memory Manager
Tests memory tracking, cleanup, and leak detection integration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import gc

from core.memory_manager import MemoryManager, global_memory_manager


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create memory manager instance."""
        return MemoryManager(threshold_mb=100, check_interval=1.0)
    
    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.threshold_mb == 100
        assert manager.check_interval == 1.0
        assert manager.tracking_enabled is False
        assert isinstance(manager.snapshots, list)
    
    def test_start_tracking(self, manager):
        """Test memory tracking can be started."""
        manager.start_tracking()
        
        assert manager.tracking_enabled is True
    
    def test_stop_tracking(self, manager):
        """Test memory tracking can be stopped."""
        manager.start_tracking()
        manager.stop_tracking()
        
        assert manager.tracking_enabled is False
    
    def test_get_current_usage(self, manager):
        """Test current memory usage retrieval."""
        usage = manager.get_current_usage()
        
        assert isinstance(usage, dict)
        assert 'rss_mb' in usage
        assert 'vms_mb' in usage
        assert 'percent' in usage
        assert usage['rss_mb'] > 0
    
    def test_force_cleanup(self, manager):
        """Test forced garbage collection."""
        # Create some garbage
        _ = [i for i in range(10000)]
        
        initial_usage = manager.get_current_usage()
        manager.force_cleanup()
        final_usage = manager.get_current_usage()
        
        # Cleanup should have been triggered
        assert isinstance(final_usage, dict)
    
    def test_is_above_threshold(self, manager):
        """Test threshold checking."""
        # Set very high threshold
        manager.threshold_mb = 100000
        assert manager.is_above_threshold() is False
        
        # Set very low threshold
        manager.threshold_mb = 0.001
        assert manager.is_above_threshold() is True
    
    def test_take_snapshot(self, manager):
        """Test memory snapshot."""
        manager.start_tracking()
        
        snapshot = manager.take_snapshot()
        
        assert snapshot is not None
        assert 'timestamp' in snapshot
        assert 'usage' in snapshot
        assert snapshot['usage']['rss_mb'] > 0
    
    def test_get_leak_report(self, manager):
        """Test leak report generation."""
        manager.start_tracking()
        
        # Take a few snapshots
        manager.take_snapshot()
        manager.take_snapshot()
        
        report = manager.get_leak_report()
        
        assert isinstance(report, dict)
    
    def test_clear_snapshots(self, manager):
        """Test snapshot clearing."""
        manager.start_tracking()
        manager.take_snapshot()
        manager.take_snapshot()
        
        assert len(manager.snapshots) > 0
        
        manager.clear_snapshots()
        
        assert len(manager.snapshots) == 0
    
    def test_get_stats(self, manager):
        """Test statistics retrieval."""
        stats = manager.get_stats()
        
        assert isinstance(stats, dict)
        assert 'tracking_enabled' in stats
        assert 'current_usage' in stats
        assert 'threshold_mb' in stats
    
    def test_global_memory_manager(self):
        """Test global memory manager instance."""
        assert global_memory_manager is not None
        assert isinstance(global_memory_manager, MemoryManager)


class TestMemoryManagerIntegration:
    """Integration tests for Memory Manager."""
    
    @pytest.fixture
    def manager(self):
        """Create memory manager with leak detector."""
        mgr = MemoryManager(threshold_mb=100)
        return mgr
    
    def test_memory_tracking_workflow(self, manager):
        """Test complete memory tracking workflow."""
        # Start tracking
        manager.start_tracking()
        assert manager.tracking_enabled is True
        
        # Take initial snapshot
        snapshot1 = manager.take_snapshot()
        assert snapshot1 is not None
        
        # Allocate some memory
        data = [i for i in range(100000)]
        
        # Take second snapshot
        snapshot2 = manager.take_snapshot()
        assert snapshot2 is not None
        
        # Usage should have increased
        assert snapshot2['usage']['rss_mb'] >= snapshot1['usage']['rss_mb']
        
        # Cleanup
        del data
        manager.force_cleanup()
        
        # Stop tracking
        manager.stop_tracking()
        assert manager.tracking_enabled is False
    
    def test_leak_detection_integration(self, manager):
        """Test leak detector integration."""
        manager.start_tracking()
        
        # Simulate potential leak
        leak_data = []
        for i in range(5):
            leak_data.append([j for j in range(10000)])
            manager.take_snapshot()
        
        # Get leak report
        report = manager.get_leak_report()
        assert isinstance(report, dict)
        
        # Cleanup
        del leak_data
        manager.force_cleanup()
        manager.stop_tracking()