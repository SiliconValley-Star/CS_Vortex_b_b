"""
VORTEX Streaming Memory Manager - V17.0 ULTIMATE
Semantic memory management with streaming support

RESPONSIBILITIES:
- Memory usage monitoring
- Automatic cleanup when thresholds exceeded
- Semantic context management
- Memory-efficient data structures
"""

import asyncio
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """
    Memory manager with semantic context tracking and automatic cleanup.
    Ensures system operates within memory constraints.
    """
    
    def __init__(self, max_memory_mb: int = 6000):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = 0.85  # Cleanup at 85%
        self.emergency_threshold = 0.95  # Emergency cleanup at 95%
        
        # Memory tracking
        self.current_memory_mb = 0.0
        self.peak_memory_mb = 0.0
        self.cleanup_count = 0
        self.emergency_cleanup_count = 0
        
        # Context storage (limited size)
        self.context_store: deque = deque(maxlen=1000)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 30  # seconds
        
        # Statistics
        self.stats = {
            'total_cleanups': 0,
            'emergency_cleanups': 0,
            'peak_memory_mb': 0.0,
            'avg_memory_mb': 0.0,
            'memory_warnings': 0
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            self.current_memory_mb = memory_mb
            
            if memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = memory_mb
                self.stats['peak_memory_mb'] = memory_mb
            
            return memory_mb
        except Exception as e:
            logger.error(f"Memory usage check error: {e}")
            return 0.0
    
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of limit."""
        return (self.current_memory_mb / self.max_memory_mb) * 100
    
    def should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        memory_mb = self.get_memory_usage()
        threshold_mb = self.max_memory_mb * self.cleanup_threshold
        return memory_mb >= threshold_mb
    
    def should_emergency_cleanup(self) -> bool:
        """Check if emergency cleanup is needed."""
        memory_mb = self.get_memory_usage()
        threshold_mb = self.max_memory_mb * self.emergency_threshold
        return memory_mb >= threshold_mb
    
    async def perform_cleanup(self, force: bool = False):
        """
        Perform memory cleanup.
        
        Args:
            force: Force cleanup regardless of thresholds
        """
        if not force and not self.should_cleanup():
            return
        
        logger.info("Performing memory cleanup...")
        
        try:
            # Clear old context
            if len(self.context_store) > 500:
                # Keep only recent 500 entries
                while len(self.context_store) > 500:
                    self.context_store.popleft()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.cleanup_count += 1
            self.stats['total_cleanups'] += 1
            
            memory_after = self.get_memory_usage()
            logger.info(f"Cleanup complete. Memory: {memory_after:.1f} MB")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    async def perform_emergency_cleanup(self):
        """Perform aggressive emergency cleanup."""
        logger.warning("Performing EMERGENCY memory cleanup...")
        
        try:
            # Clear most of context store
            self.context_store.clear()
            
            # Force aggressive garbage collection
            import gc
            gc.collect()
            gc.collect()
            gc.collect()
            
            self.emergency_cleanup_count += 1
            self.stats['emergency_cleanups'] += 1
            
            memory_after = self.get_memory_usage()
            logger.warning(f"Emergency cleanup complete. Memory: {memory_after:.1f} MB")
            
        except Exception as e:
            logger.error(f"Emergency cleanup error: {e}")
    
    async def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Memory monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        try:
            while self.monitoring_active:
                # Check memory
                memory_mb = self.get_memory_usage()
                memory_percent = self.get_memory_percent()
                
                # Emergency cleanup if needed
                if self.should_emergency_cleanup():
                    logger.critical(f"Memory at {memory_percent:.1f}% - emergency cleanup!")
                    self.stats['memory_warnings'] += 1
                    await self.perform_emergency_cleanup()
                
                # Regular cleanup if needed
                elif self.should_cleanup():
                    logger.warning(f"Memory at {memory_percent:.1f}% - cleanup needed")
                    self.stats['memory_warnings'] += 1
                    await self.perform_cleanup()
                
                # Wait for next check
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    def store_context(self, context: Dict[str, Any]):
        """
        Store context entry.
        
        Args:
            context: Context data to store
        """
        self.context_store.append({
            'data': context,
            'timestamp': datetime.utcnow()
        })
    
    def get_recent_context(self, count: int = 10) -> list:
        """
        Get recent context entries.
        
        Args:
            count: Number of entries to retrieve
            
        Returns:
            List of recent context entries
        """
        return list(self.context_store)[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return {
            'current_memory_mb': self.current_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'memory_percent': self.get_memory_percent(),
            'peak_memory_mb': self.peak_memory_mb,
            'cleanup_count': self.cleanup_count,
            'emergency_cleanup_count': self.emergency_cleanup_count,
            'context_entries': len(self.context_store),
            'monitoring_active': self.monitoring_active,
            **self.stats
        }
    
    def get_memory_status(self) -> str:
        """Get memory status indicator."""
        memory_percent = self.get_memory_percent()
        
        if memory_percent >= self.emergency_threshold * 100:
            return "CRITICAL"
        elif memory_percent >= self.cleanup_threshold * 100:
            return "WARNING"
        elif memory_percent >= 70:
            return "ATTENTION"
        else:
            return "HEALTHY"