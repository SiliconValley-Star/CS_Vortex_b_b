"""
VORTEX Dynamic Memory Manager - V17.0 ULTIMATE
Production-grade memory management with semantic cleanup

Per .clinerules VORTEX_OPERATIONAL_HEALTH.md:
- Memory zones: GREEN (<60%), YELLOW (60-85%), RED (>85%)
- Emergency cleanup at 95%
- Semantic targeting of actual memory consumers

FEATURES:
- Dynamic backpressure control
- Semantic cleanup targeting large responses
- Memory zone monitoring
- Automatic emergency recovery
- Response truncation strategies
"""

import asyncio
import psutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# V22.0 - Memory Leak Detection Integration
try:
    from core.memory_leak_detector import global_leak_detector, MemoryLeakDetector
    LEAK_DETECTION_AVAILABLE = True
except ImportError:
    LEAK_DETECTION_AVAILABLE = False
    logging.warning("Memory leak detection not available")

logger = logging.getLogger(__name__)


class MemoryZone(str, Enum):
    """Memory usage zones with associated thresholds."""
    GREEN = "GREEN"      # <60% - Normal operation
    YELLOW = "YELLOW"    # 60-85% - Warning, start optimization
    RED = "RED"          # 85-95% - Critical, aggressive cleanup
    EMERGENCY = "EMERGENCY"  # >95% - Emergency procedures


@dataclass
class MemoryStats:
    """Current memory statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    zone: MemoryZone
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Process-specific
    process_mb: float = 0.0
    process_percent: float = 0.0
    
    # Trend analysis
    trending_up: bool = False
    rate_of_change: float = 0.0


@dataclass
class CleanupResult:
    """Memory cleanup operation result."""
    bytes_freed: int
    items_cleaned: int
    cleanup_duration: float
    zone_before: MemoryZone
    zone_after: MemoryZone
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DynamicMemoryManager:
    """
    Dynamic memory manager with semantic cleanup.
    
    RESPONSIBILITIES:
    - Monitor memory usage continuously
    - Apply backpressure when approaching limits
    - Perform semantic cleanup of large objects
    - Emergency recovery procedures
    - Trend analysis for proactive management
    
    Per .clinerules: Maximum memory usage <6GB
    """
    
    # Memory thresholds (V21.0 - Conservative defaults for production)
    MEMORY_LIMIT_MB = 4000.0  # 4GB (reduced from 6GB for stability)
    GREEN_THRESHOLD = 0.60    # 60%
    YELLOW_THRESHOLD = 0.70   # 70% (reduced from 85% for earlier cleanup)
    RED_THRESHOLD = 0.85      # 85% (reduced from 95% for earlier intervention)
    EMERGENCY_THRESHOLD = 0.95  # 95% - Emergency procedures
    
    # Cleanup strategies
    TRUNCATION_THRESHOLD = 50000  # 50KB - truncate larger responses
    AGGRESSIVE_TRUNCATION = 10000  # 10KB - aggressive mode
    
    def __init__(self, lightweight_mode: bool = False):
        """
        Initialize memory manager.
        
        Args:
            lightweight_mode: If True, use more aggressive limits for low-resource environments
        """
        self.process = psutil.Process()
        self.baseline_memory: Optional[float] = None
        self.lightweight_mode = lightweight_mode
        
        # V21.0 - Adjust thresholds for lightweight mode
        if lightweight_mode:
            self.MEMORY_LIMIT_MB = 1024.0  # 1GB for lightweight
            self.GREEN_THRESHOLD = 0.50    # 50%
            self.YELLOW_THRESHOLD = 0.65   # 65%
            self.RED_THRESHOLD = 0.80      # 80%
            self.EMERGENCY_THRESHOLD = 0.90  # 90%
            logger.info("Lightweight memory mode enabled: 1GB limit")
        
        # V22.0 - Leak detector integration
        self.leak_detector = global_leak_detector if LEAK_DETECTION_AVAILABLE else None
        
        # Response cache for semantic cleanup
        self.response_cache: Dict[str, bytes] = {}
        self.large_objects: List[tuple] = []  # (size, key, timestamp)
        
        # History for trend analysis
        self.memory_history: List[MemoryStats] = []
        self.max_history = 60  # Keep last 60 measurements
        
        # Current state
        self.current_zone = MemoryZone.GREEN
        self.cleanup_in_progress = False
        self.emergency_mode = False
        
        # Statistics
        self.total_cleanups = 0
        self.total_bytes_freed = 0
        self.emergency_cleanups = 0
        
        logger.info(f"Dynamic Memory Manager initialized (limit: {self.MEMORY_LIMIT_MB}MB)")
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.
        
        Returns:
            Current memory stats with zone classification
        """
        # System memory
        vm = psutil.virtual_memory()
        
        # Process memory
        process_info = self.process.memory_info()
        process_mb = process_info.rss / (1024 * 1024)
        
        # Calculate zone (V21.0 - Use configurable thresholds)
        percent_used = vm.percent / 100.0
        if percent_used >= self.EMERGENCY_THRESHOLD:
            zone = MemoryZone.EMERGENCY
        elif percent_used >= self.RED_THRESHOLD:
            zone = MemoryZone.RED
        elif percent_used >= self.YELLOW_THRESHOLD:
            zone = MemoryZone.YELLOW
        else:
            zone = MemoryZone.GREEN
        
        stats = MemoryStats(
            total_mb=vm.total / (1024 * 1024),
            available_mb=vm.available / (1024 * 1024),
            used_mb=vm.used / (1024 * 1024),
            percent_used=percent_used,
            zone=zone,
            process_mb=process_mb,
            process_percent=(process_mb / (vm.total / (1024 * 1024))) * 100
        )
        
        # Trend analysis
        if len(self.memory_history) >= 2:
            prev_stats = self.memory_history[-1]
            stats.rate_of_change = stats.percent_used - prev_stats.percent_used
            stats.trending_up = stats.rate_of_change > 0.01  # >1% increase
        
        # Update history
        self.memory_history.append(stats)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
        
        self.current_zone = zone
        return stats
    
    async def should_apply_backpressure(self) -> bool:
        """
        Check if backpressure should be applied.
        
        Returns:
            True if system should slow down processing
        """
        stats = self.get_memory_stats()
        
        # Apply backpressure in RED or EMERGENCY zones
        if stats.zone in [MemoryZone.RED, MemoryZone.EMERGENCY]:
            logger.warning(f"Applying backpressure - Memory zone: {stats.zone}")
            return True
        
        # Proactive backpressure if trending up rapidly
        if stats.trending_up and stats.rate_of_change > 0.05:  # >5% increase
            logger.info("Applying proactive backpressure - rapid memory increase")
            return True
        
        return False
    
    async def register_large_response(self, key: str, data: bytes) -> None:
        """
        Register large response for potential cleanup.
        
        Args:
            key: Unique identifier for the response
            data: Response data
        """
        size = len(data)
        
        # Only track large objects
        if size > self.TRUNCATION_THRESHOLD:
            self.response_cache[key] = data
            self.large_objects.append((size, key, datetime.utcnow()))
            
            # Keep sorted by size (largest first)
            self.large_objects.sort(reverse=True)
            
            logger.debug(f"Registered large response: {key} ({size} bytes)")
    
    async def semantic_cleanup(self, aggressive: bool = False) -> CleanupResult:
        """
        Perform semantic cleanup targeting large objects.
        
        Per .clinerules: Semantic cleanup targets actual memory consumers,
        not arbitrary object types.
        
        Args:
            aggressive: If True, use more aggressive truncation
            
        Returns:
            Cleanup result with stats
        """
        if self.cleanup_in_progress:
            logger.debug("Cleanup already in progress, skipping")
            return CleanupResult(0, 0, 0.0, self.current_zone, self.current_zone)
        
        self.cleanup_in_progress = True
        start_time = datetime.utcnow()
        zone_before = self.current_zone
        
        bytes_freed = 0
        items_cleaned = 0
        
        try:
            truncation_limit = self.AGGRESSIVE_TRUNCATION if aggressive else self.TRUNCATION_THRESHOLD
            
            # Clean largest objects first
            for size, key, timestamp in list(self.large_objects):
                if key in self.response_cache:
                    original_data = self.response_cache[key]
                    original_size = len(original_data)
                    
                    # Truncate response
                    if original_size > truncation_limit:
                        truncated = original_data[:truncation_limit]
                        self.response_cache[key] = truncated
                        
                        bytes_freed += (original_size - len(truncated))
                        items_cleaned += 1
                        
                        logger.debug(f"Truncated {key}: {original_size} -> {len(truncated)} bytes")
                
                # Stop if we've freed enough memory (in aggressive mode)
                if aggressive and bytes_freed > 100 * 1024 * 1024:  # 100MB
                    break
            
            # Remove old entries (>5 minutes old)
            current_time = datetime.utcnow()
            self.large_objects = [
                (size, key, ts) for size, key, ts in self.large_objects
                if (current_time - ts).total_seconds() < 300
            ]
            
            # Update statistics
            self.total_cleanups += 1
            self.total_bytes_freed += bytes_freed
            
            # Get new zone
            stats = self.get_memory_stats()
            zone_after = stats.zone
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Semantic cleanup: freed {bytes_freed} bytes, "
                       f"cleaned {items_cleaned} items in {duration:.2f}s")
            
            return CleanupResult(
                bytes_freed=bytes_freed,
                items_cleaned=items_cleaned,
                cleanup_duration=duration,
                zone_before=zone_before,
                zone_after=zone_after
            )
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)
            return CleanupResult(0, 0, 0.0, zone_before, self.current_zone)
        
        finally:
            self.cleanup_in_progress = False
    
    async def emergency_cleanup(self) -> CleanupResult:
        """
        Emergency cleanup when memory critical.
        
        Per .clinerules: Emergency cleanup at >95% memory usage
        """
        logger.critical("EMERGENCY MEMORY CLEANUP TRIGGERED")
        self.emergency_mode = True
        self.emergency_cleanups += 1
        
        try:
            # Aggressive semantic cleanup
            result = await self.semantic_cleanup(aggressive=True)
            
            # Clear all response cache if still critical
            stats = self.get_memory_stats()
            if stats.zone == MemoryZone.EMERGENCY:
                cache_size = len(self.response_cache)
                self.response_cache.clear()
                self.large_objects.clear()
                
                logger.critical(f"Cleared entire response cache ({cache_size} items)")
                result.items_cleaned += cache_size
            
            return result
            
        finally:
            self.emergency_mode = False
    
    async def auto_manage_memory(self) -> Optional[CleanupResult]:
        """
        Automatic memory management based on current zone.
        
        Returns:
            Cleanup result if cleanup was performed
        """
        stats = self.get_memory_stats()
        
        if stats.zone == MemoryZone.EMERGENCY:
            logger.critical(f"EMERGENCY zone: {stats.percent_used:.1%} memory used")
            return await self.emergency_cleanup()
        
        elif stats.zone == MemoryZone.RED:
            logger.error(f"RED zone: {stats.percent_used:.1%} memory used")
            return await self.semantic_cleanup(aggressive=True)
        
        elif stats.zone == MemoryZone.YELLOW:
            logger.warning(f"YELLOW zone: {stats.percent_used:.1%} memory used")
            return await self.semantic_cleanup(aggressive=False)
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory manager status."""
        stats = self.get_memory_stats()
        
        status = {
            'zone': stats.zone.value,
            'memory_usage_mb': stats.used_mb,
            'memory_percent': stats.percent_used * 100,
            'memory_limit_mb': self.MEMORY_LIMIT_MB,  # V21.0
            'process_mb': stats.process_mb,
            'available_mb': stats.available_mb,
            'total_cleanups': self.total_cleanups,
            'total_bytes_freed': self.total_bytes_freed,
            'emergency_cleanups': self.emergency_cleanups,
            'trending_up': stats.trending_up,
            'rate_of_change': stats.rate_of_change,
            'emergency_mode': self.emergency_mode,
            'lightweight_mode': self.lightweight_mode,  # V21.0
            'large_objects_tracked': len(self.large_objects),
            'thresholds': {  # V21.0 - Expose thresholds
                'green': self.GREEN_THRESHOLD,
                'yellow': self.YELLOW_THRESHOLD,
                'red': self.RED_THRESHOLD,
                'emergency': self.EMERGENCY_THRESHOLD,
            }
        }
        
        # V22.0 - Add leak detection status
        if self.leak_detector:
            status['leak_detection'] = self.leak_detector.get_status()
        
        return status
    
    def get_truncation_limit(self) -> int:
        """Get current response truncation limit based on memory zone."""
        if self.current_zone == MemoryZone.EMERGENCY:
            return self.AGGRESSIVE_TRUNCATION
        elif self.current_zone == MemoryZone.RED:
            return self.TRUNCATION_THRESHOLD // 2
        elif self.current_zone == MemoryZone.YELLOW:
            return self.TRUNCATION_THRESHOLD
        else:
            return self.TRUNCATION_THRESHOLD * 2  # More lenient in GREEN
    
    def start_leak_detection(self):
        """Start memory leak detection if available."""
        if self.leak_detector and LEAK_DETECTION_AVAILABLE:
            self.leak_detector.start_tracking()
            logger.info("Memory leak detection started")
        else:
            logger.warning("Memory leak detection not available")
    
    def stop_leak_detection(self):
        """Stop memory leak detection."""
        if self.leak_detector and LEAK_DETECTION_AVAILABLE:
            self.leak_detector.stop_tracking()
            logger.info("Memory leak detection stopped")
    
    def get_leak_report(self):
        """Generate memory leak report if available."""
        if self.leak_detector and LEAK_DETECTION_AVAILABLE:
            try:
                return self.leak_detector.generate_report()
            except Exception as e:
                logger.error(f"Failed to generate leak report: {e}")
                return None
        return None


# Global memory manager instance
global_memory_manager = DynamicMemoryManager()