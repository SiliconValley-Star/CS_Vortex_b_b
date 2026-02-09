"""
VORTEX Memory Leak Detector - V22.0
Advanced memory leak detection and analysis system

FEATURES:
- Object lifecycle tracking
- Memory snapshot comparison
- Leak pattern detection  
- Growth rate analysis
- Automatic leak alerting
- Detailed leak reports
"""

import asyncio
import logging
import tracemalloc
import gc
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import sys

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory snapshot at a specific point in time."""
    timestamp: datetime
    snapshot: tracemalloc.Snapshot
    total_size: int  # bytes
    total_blocks: int
    top_allocations: List[Dict[str, Any]]
    
    def get_size_mb(self) -> float:
        """Get total size in MB."""
        return self.total_size / (1024 * 1024)


@dataclass
class LeakPattern:
    """Detected memory leak pattern."""
    pattern_type: str  # 'gradual', 'sudden', 'cyclic', 'stable'
    source_file: str
    source_line: int
    size_growth_bytes: int
    growth_rate_mb_per_min: float
    confidence: float  # 0.0-1.0
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int = 1
    
    def get_severity(self) -> str:
        """Get leak severity level."""
        if self.growth_rate_mb_per_min > 10:
            return "CRITICAL"
        elif self.growth_rate_mb_per_min > 5:
            return "HIGH"
        elif self.growth_rate_mb_per_min > 1:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class LeakReport:
    """Comprehensive leak detection report."""
    report_id: str
    timestamp: datetime
    duration_minutes: float
    total_growth_mb: float
    average_growth_rate: float
    detected_leaks: List[LeakPattern]
    top_growing_files: List[Tuple[str, int]]  # (filename, bytes)
    recommendations: List[str]
    
    def has_critical_leaks(self) -> bool:
        """Check if report contains critical leaks."""
        return any(leak.get_severity() == "CRITICAL" for leak in self.detected_leaks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_id': self.report_id,
            'timestamp': self.timestamp.isoformat(),
            'duration_minutes': self.duration_minutes,
            'total_growth_mb': self.total_growth_mb,
            'average_growth_rate': self.average_growth_rate,
            'detected_leaks_count': len(self.detected_leaks),
            'critical_leaks': sum(1 for l in self.detected_leaks if l.get_severity() == "CRITICAL"),
            'top_growing_files': self.top_growing_files[:10],
            'recommendations': self.recommendations
        }


class MemoryLeakDetector:
    """
    Advanced memory leak detector with pattern recognition.
    
    CAPABILITIES:
    - Tracks memory allocation over time
    - Compares snapshots to identify growth
    - Detects leak patterns (gradual, sudden, cyclic)
    - Generates detailed leak reports
    - Provides remediation recommendations
    """
    
    def __init__(self, 
                 snapshot_interval_seconds: int = 300,  # 5 minutes
                 max_snapshots: int = 20):
        self.snapshot_interval = snapshot_interval_seconds
        self.max_snapshots = max_snapshots
        
        # Snapshot storage
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        
        # Leak tracking
        self.detected_leaks: Dict[str, LeakPattern] = {}
        self.leak_history: List[LeakPattern] = []
        
        # State
        self.is_tracking = False
        self.tracking_task: Optional[asyncio.Task] = None
        self.start_time: Optional[datetime] = None
        
        # Statistics
        self.total_snapshots = 0
        self.leaks_detected = 0
        self.false_positives = 0
        
        logger.info("Memory Leak Detector initialized")
    
    def start_tracking(self):
        """Start memory leak tracking."""
        if self.is_tracking:
            logger.warning("Leak tracking already active")
            return
        
        # Start tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Track 25 frames
        
        self.is_tracking = True
        self.start_time = datetime.utcnow()
        
        # Take baseline snapshot
        self.baseline_snapshot = self._take_snapshot()
        logger.info("Memory leak tracking started")
    
    def stop_tracking(self):
        """Stop memory leak tracking."""
        if not self.is_tracking:
            return
        
        self.is_tracking = False
        
        # Stop tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        logger.info("Memory leak tracking stopped")
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)
        
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')
        
        total_size = sum(stat.size for stat in stats)
        total_blocks = sum(stat.count for stat in stats)
        
        # Get top allocations
        top_allocations = []
        for stat in stats[:20]:
            top_allocations.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_bytes': stat.size,
                'size_mb': stat.size / (1024 * 1024),
                'count': stat.count
            })
        
        mem_snapshot = MemorySnapshot(
            timestamp=datetime.utcnow(),
            snapshot=snapshot,
            total_size=total_size,
            total_blocks=total_blocks,
            top_allocations=top_allocations
        )
        
        # Store snapshot
        self.snapshots.append(mem_snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        self.total_snapshots += 1
        
        return mem_snapshot
    
    def compare_snapshots(self, 
                         snapshot1: MemorySnapshot, 
                         snapshot2: MemorySnapshot) -> List[Dict[str, Any]]:
        """
        Compare two snapshots to find memory growth.
        
        Returns:
            List of file statistics with growth information
        """
        stats = snapshot2.snapshot.compare_to(snapshot1.snapshot, 'lineno')
        
        growth_stats = []
        for stat in stats[:50]:  # Top 50 growing allocations
            if stat.size_diff > 0:  # Only growing allocations
                growth_stats.append({
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'line': stat.traceback[0].lineno if stat.traceback else 0,
                    'size_diff_bytes': stat.size_diff,
                    'size_diff_mb': stat.size_diff / (1024 * 1024),
                    'count_diff': stat.count_diff,
                    'current_size': stat.size,
                    'current_count': stat.count
                })
        
        return growth_stats
    
    def analyze_leak_patterns(self) -> List[LeakPattern]:
        """
        Analyze snapshots to detect leak patterns.
        
        Returns:
            List of detected leak patterns
        """
        if len(self.snapshots) < 3:
            logger.debug("Not enough snapshots for leak analysis")
            return []
        
        detected = []
        
        # Compare consecutive snapshots
        for i in range(1, len(self.snapshots)):
            prev_snapshot = self.snapshots[i-1]
            current_snapshot = self.snapshots[i]
            
            growth_stats = self.compare_snapshots(prev_snapshot, current_snapshot)
            
            # Analyze each growing allocation
            for stat in growth_stats:
                # Calculate growth rate
                time_diff = (current_snapshot.timestamp - prev_snapshot.timestamp).total_seconds() / 60
                growth_rate = stat['size_diff_mb'] / time_diff if time_diff > 0 else 0
                
                # Detect leak if growth rate is significant
                if growth_rate > 0.1:  # >0.1 MB/min
                    # Determine pattern type
                    pattern_type = self._classify_leak_pattern(stat, growth_rate)
                    
                    # Calculate confidence
                    confidence = self._calculate_leak_confidence(stat, growth_rate)
                    
                    if confidence > 0.6:  # Confidence threshold
                        leak = LeakPattern(
                            pattern_type=pattern_type,
                            source_file=stat['file'],
                            source_line=stat['line'],
                            size_growth_bytes=stat['size_diff_bytes'],
                            growth_rate_mb_per_min=growth_rate,
                            confidence=confidence,
                            first_seen=prev_snapshot.timestamp,
                            last_seen=current_snapshot.timestamp
                        )
                        
                        # Check if this is a known leak
                        leak_key = f"{leak.source_file}:{leak.source_line}"
                        if leak_key in self.detected_leaks:
                            # Update existing leak
                            existing = self.detected_leaks[leak_key]
                            existing.occurrence_count += 1
                            existing.last_seen = leak.last_seen
                            existing.size_growth_bytes += leak.size_growth_bytes
                        else:
                            # New leak
                            self.detected_leaks[leak_key] = leak
                            detected.append(leak)
                            self.leaks_detected += 1
        
        return detected
    
    def _classify_leak_pattern(self, stat: Dict, growth_rate: float) -> str:
        """Classify the type of memory leak pattern."""
        if growth_rate > 5:
            return "sudden"
        elif growth_rate > 1:
            return "gradual"
        elif stat['count_diff'] > 100:
            return "cyclic"
        else:
            return "stable"
    
    def _calculate_leak_confidence(self, stat: Dict, growth_rate: float) -> float:
        """
        Calculate confidence score for leak detection.
        
        Factors:
        - Growth rate (higher = more confident)
        - Allocation count (more = more confident)
        - Size (larger = more confident)
        """
        confidence = 0.0
        
        # Growth rate factor (0-0.4)
        if growth_rate > 10:
            confidence += 0.4
        elif growth_rate > 5:
            confidence += 0.3
        elif growth_rate > 1:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Size factor (0-0.3)
        size_mb = stat['size_diff_mb']
        if size_mb > 50:
            confidence += 0.3
        elif size_mb > 10:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # Count factor (0-0.3)
        if stat['count_diff'] > 1000:
            confidence += 0.3
        elif stat['count_diff'] > 100:
            confidence += 0.2
        else:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def generate_report(self) -> LeakReport:
        """
        Generate comprehensive leak detection report.
        
        Returns:
            Detailed leak report with recommendations
        """
        if not self.snapshots:
            raise ValueError("No snapshots available for report generation")
        
        # Calculate overall statistics
        duration = (datetime.utcnow() - self.start_time).total_seconds() / 60 if self.start_time else 0
        
        if self.baseline_snapshot and self.snapshots:
            latest_snapshot = self.snapshots[-1]
            total_growth = latest_snapshot.get_size_mb() - self.baseline_snapshot.get_size_mb()
            avg_growth_rate = total_growth / duration if duration > 0 else 0
        else:
            total_growth = 0
            avg_growth_rate = 0
        
        # Get top growing files
        if self.baseline_snapshot and self.snapshots:
            growth_stats = self.compare_snapshots(self.baseline_snapshot, self.snapshots[-1])
            top_files = [(s['file'], s['size_diff_bytes']) for s in growth_stats[:10]]
        else:
            top_files = []
        
        # Generate recommendations
        recommendations = self._generate_recommendations(list(self.detected_leaks.values()))
        
        report = LeakReport(
            report_id=f"leak_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
            duration_minutes=duration,
            total_growth_mb=total_growth,
            average_growth_rate=avg_growth_rate,
            detected_leaks=list(self.detected_leaks.values()),
            top_growing_files=top_files,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self, leaks: List[LeakPattern]) -> List[str]:
        """Generate remediation recommendations based on detected leaks."""
        recommendations = []
        
        if not leaks:
            recommendations.append("✓ No memory leaks detected")
            return recommendations
        
        # Critical leaks
        critical = [l for l in leaks if l.get_severity() == "CRITICAL"]
        if critical:
            recommendations.append(
                f"⚠️ CRITICAL: {len(critical)} critical leak(s) detected - immediate action required"
            )
            for leak in critical[:3]:
                recommendations.append(
                    f"  → {leak.source_file}:{leak.source_line} "
                    f"(+{leak.growth_rate_mb_per_min:.2f} MB/min)"
                )
        
        # High severity leaks
        high = [l for l in leaks if l.get_severity() == "HIGH"]
        if high:
            recommendations.append(
                f"⚠️ HIGH: {len(high)} high-severity leak(s) - action needed soon"
            )
        
        # Pattern-specific recommendations
        gradual_leaks = [l for l in leaks if l.pattern_type == "gradual"]
        if gradual_leaks:
            recommendations.append(
                "💡 Gradual leaks detected - check for unbounded collections or caches"
            )
        
        sudden_leaks = [l for l in leaks if l.pattern_type == "sudden"]
        if sudden_leaks:
            recommendations.append(
                "💡 Sudden leaks detected - check for large object allocations"
            )
        
        cyclic_leaks = [l for l in leaks if l.pattern_type == "cyclic"]
        if cyclic_leaks:
            recommendations.append(
                "💡 Cyclic leaks detected - check for circular references or event handlers"
            )
        
        # General recommendations
        recommendations.append("💡 Run garbage collection: gc.collect()")
        recommendations.append("💡 Review object lifecycle management")
        recommendations.append("💡 Check for proper cleanup in async operations")
        
        return recommendations
    
    async def continuous_monitoring(self):
        """Continuous memory leak monitoring loop."""
        logger.info("Starting continuous leak monitoring")
        
        try:
            while self.is_tracking:
                # Take snapshot
                snapshot = self._take_snapshot()
                logger.debug(f"Snapshot taken: {snapshot.get_size_mb():.2f} MB")
                
                # Analyze for leaks
                new_leaks = self.analyze_leak_patterns()
                if new_leaks:
                    for leak in new_leaks:
                        logger.warning(
                            f"Memory leak detected: {leak.source_file}:{leak.source_line} "
                            f"({leak.get_severity()}) +{leak.growth_rate_mb_per_min:.2f} MB/min"
                        )
                
                # Wait for next interval
                await asyncio.sleep(self.snapshot_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring error: {e}", exc_info=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        return {
            'is_tracking': self.is_tracking,
            'total_snapshots': self.total_snapshots,
            'active_snapshots': len(self.snapshots),
            'detected_leaks': len(self.detected_leaks),
            'leaks_by_severity': {
                'critical': sum(1 for l in self.detected_leaks.values() if l.get_severity() == "CRITICAL"),
                'high': sum(1 for l in self.detected_leaks.values() if l.get_severity() == "HIGH"),
                'medium': sum(1 for l in self.detected_leaks.values() if l.get_severity() == "MEDIUM"),
                'low': sum(1 for l in self.detected_leaks.values() if l.get_severity() == "LOW")
            },
            'tracking_duration_minutes': (
                (datetime.utcnow() - self.start_time).total_seconds() / 60 
                if self.start_time else 0
            )
        }


# Global leak detector instance
global_leak_detector = MemoryLeakDetector()