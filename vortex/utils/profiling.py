"""
VORTEX Performance Profiling Utilities
Production-grade profiling infrastructure for performance analysis

FEATURES:
- Function-level profiling (sync & async)
- Memory profiling
- Context-based profiling
- JSON export for analysis
- Zero overhead when disabled
"""

import cProfile
import pstats
import io
import time
import asyncio
import functools
import logging
import json
from typing import Any, Callable, Optional, Dict
from datetime import datetime
from pathlib import Path
import tracemalloc

logger = logging.getLogger(__name__)


class ProfileResult:
    """Result of a profiling operation."""
    
    def __init__(self, 
                 function_name: str,
                 execution_time: float,
                 memory_peak: Optional[int] = None,
                 call_count: int = 1):
        self.function_name = function_name
        self.execution_time = execution_time
        self.memory_peak = memory_peak
        self.call_count = call_count
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'function_name': self.function_name,
            'execution_time_seconds': self.execution_time,
            'memory_peak_bytes': self.memory_peak,
            'call_count': self.call_count,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class Profiler:
    """
    Unified profiler for CPU and memory profiling.
    
    Usage:
        profiler = Profiler()
        with profiler.profile('my_operation'):
            expensive_operation()
        
        result = profiler.get_result('my_operation')
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.results: Dict[str, ProfileResult] = {}
        self._active_profiles: Dict[str, Dict[str, Any]] = {}
    
    def profile(self, name: str, track_memory: bool = True):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Identifier for this profile
            track_memory: Whether to track memory usage
        """
        return ProfileContext(self, name, track_memory)
    
    def start_profile(self, name: str, track_memory: bool = True):
        """Start profiling operation."""
        if not self.enabled:
            return
        
        profile_data = {
            'start_time': time.time(),
            'profiler': cProfile.Profile()
        }
        
        if track_memory:
            tracemalloc.start()
            profile_data['memory_start'] = tracemalloc.get_traced_memory()[0]
        
        profile_data['profiler'].enable()
        self._active_profiles[name] = profile_data
    
    def stop_profile(self, name: str) -> Optional[ProfileResult]:
        """Stop profiling operation and return result."""
        if not self.enabled or name not in self._active_profiles:
            return None
        
        profile_data = self._active_profiles.pop(name)
        profile_data['profiler'].disable()
        
        # Calculate execution time
        execution_time = time.time() - profile_data['start_time']
        
        # Get memory peak if tracking
        memory_peak = None
        if 'memory_start' in profile_data:
            current, peak = tracemalloc.get_traced_memory()
            memory_peak = peak - profile_data['memory_start']
            tracemalloc.stop()
        
        # Create result
        result = ProfileResult(
            function_name=name,
            execution_time=execution_time,
            memory_peak=memory_peak
        )
        
        self.results[name] = result
        return result
    
    def get_result(self, name: str) -> Optional[ProfileResult]:
        """Get profiling result by name."""
        return self.results.get(name)
    
    def get_all_results(self) -> Dict[str, ProfileResult]:
        """Get all profiling results."""
        return self.results.copy()
    
    def export_results(self, filepath: str):
        """Export all results to JSON file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'profile_results': [
                result.to_dict() for result in self.results.values()
            ],
            'export_time': datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Profiling results exported to {filepath}")
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()


class ProfileContext:
    """Context manager for profiling."""
    
    def __init__(self, profiler: Profiler, name: str, track_memory: bool):
        self.profiler = profiler
        self.name = name
        self.track_memory = track_memory
        self.result: Optional[ProfileResult] = None
    
    def __enter__(self):
        self.profiler.start_profile(self.name, self.track_memory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.result = self.profiler.stop_profile(self.name)
        return False


def profile(enabled: bool = True, track_memory: bool = False):
    """
    Decorator for profiling synchronous functions.
    
    Args:
        enabled: Whether profiling is enabled
        track_memory: Whether to track memory usage
        
    Usage:
        @profile(track_memory=True)
        def expensive_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        if not enabled:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler()
            
            with profiler.profile(func.__name__, track_memory):
                result = func(*args, **kwargs)
            
            profile_result = profiler.get_result(func.__name__)
            if profile_result:
                logger.debug(
                    f"Profile [{func.__name__}]: "
                    f"{profile_result.execution_time:.3f}s"
                )
            
            return result
        
        return wrapper
    return decorator


def profile_async(enabled: bool = True, track_memory: bool = False):
    """
    Decorator for profiling asynchronous functions.
    
    Args:
        enabled: Whether profiling is enabled
        track_memory: Whether to track memory usage
        
    Usage:
        @profile_async(track_memory=True)
        async def expensive_async_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        if not enabled:
            return func
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = Profiler()
            
            with profiler.profile(func.__name__, track_memory):
                result = await func(*args, **kwargs)
            
            profile_result = profiler.get_result(func.__name__)
            if profile_result:
                logger.debug(
                    f"Profile [{func.__name__}]: "
                    f"{profile_result.execution_time:.3f}s"
                )
            
            return result
        
        return wrapper
    return decorator


class MemoryProfiler:
    """Specialized memory profiler using tracemalloc."""
    
    def __init__(self):
        self.snapshots = []
        self.is_tracking = False
    
    def start(self):
        """Start memory tracking."""
        if not self.is_tracking:
            tracemalloc.start()
            self.is_tracking = True
            logger.debug("Memory profiling started")
    
    def stop(self):
        """Stop memory tracking."""
        if self.is_tracking:
            tracemalloc.stop()
            self.is_tracking = False
            logger.debug("Memory profiling stopped")
    
    def take_snapshot(self, name: str = ""):
        """Take a memory snapshot."""
        if not self.is_tracking:
            logger.warning("Memory tracking not active")
            return
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            'name': name or f"snapshot_{len(self.snapshots)}",
            'snapshot': snapshot,
            'timestamp': datetime.utcnow()
        })
    
    def get_top_allocations(self, limit: int = 10) -> list:
        """Get top memory allocations."""
        if not self.snapshots:
            return []
        
        latest = self.snapshots[-1]['snapshot']
        top_stats = latest.statistics('lineno')
        
        return [
            {
                'filename': stat.traceback.format()[0],
                'size_mb': stat.size / (1024 * 1024),
                'count': stat.count
            }
            for stat in top_stats[:limit]
        ]
    
    def compare_snapshots(self, snapshot1_idx: int = 0, snapshot2_idx: int = -1):
        """Compare two snapshots to find memory growth."""
        if len(self.snapshots) < 2:
            logger.warning("Need at least 2 snapshots to compare")
            return []
        
        snap1 = self.snapshots[snapshot1_idx]['snapshot']
        snap2 = self.snapshots[snapshot2_idx]['snapshot']
        
        top_stats = snap2.compare_to(snap1, 'lineno')
        
        return [
            {
                'filename': stat.traceback.format()[0],
                'size_diff_mb': stat.size_diff / (1024 * 1024),
                'count_diff': stat.count_diff
            }
            for stat in top_stats[:10]
        ]


# Global profiler instance
global_profiler = Profiler()
global_memory_profiler = MemoryProfiler()