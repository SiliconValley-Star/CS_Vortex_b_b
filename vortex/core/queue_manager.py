"""
VORTEX Intelligent Queue Manager - V17.0 ULTIMATE
Priority-based queue management with overflow prevention

Per .clinerules:
- Priority-based processing (HIGH > MEDIUM > LOW)
- Intelligent dropping strategies
- Queue health monitoring
- Backpressure coordination with memory manager

FEATURES:
- Multiple priority queues
- Automatic overflow prevention
- Dead letter queue for failed items
- Queue metrics and monitoring
- Integration with memory management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

from domain.enums import VerificationStatus
from core.exceptions import QueueError, QueueOverflowError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class QueuePriority(str, Enum):
    """Queue priority levels."""
    CRITICAL = "CRITICAL"  # Immediate processing
    HIGH = "HIGH"          # High priority
    MEDIUM = "MEDIUM"      # Normal priority
    LOW = "LOW"            # Lower priority
    BACKLOG = "BACKLOG"    # Background processing


class QueueStatus(str, Enum):
    """Queue operational status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OVERFLOWING = "OVERFLOWING"
    BLOCKED = "BLOCKED"


@dataclass
class QueueItem(Generic[T]):
    """Queue item with metadata."""
    data: T
    priority: QueuePriority
    added_at: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    last_error: Optional[str] = None
    
    # Metadata
    item_id: Optional[str] = None
    source: str = "unknown"
    
    def age_seconds(self) -> float:
        """Get item age in seconds."""
        return (datetime.utcnow() - self.added_at).total_seconds()


@dataclass
class QueueMetrics:
    """Queue metrics for monitoring."""
    queue_name: str
    total_size: int
    priority_distribution: Dict[str, int]
    oldest_item_age: float
    processing_rate: float  # Items per second
    
    # Health indicators
    overflow_events: int
    dropped_items: int
    failed_items: int
    
    # Status
    status: QueueStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PriorityQueue(Generic[T]):
    """
    Priority-based queue with overflow protection.
    
    Items are processed based on priority: CRITICAL > HIGH > MEDIUM > LOW > BACKLOG
    """
    
    def __init__(self, 
                 name: str,
                 max_size: int = 1000,
                 overflow_strategy: str = "drop_low_priority"):
        self.name = name
        self.max_size = max_size
        self.overflow_strategy = overflow_strategy
        
        # Priority queues
        self.queues: Dict[QueuePriority, deque] = {
            QueuePriority.CRITICAL: deque(),
            QueuePriority.HIGH: deque(),
            QueuePriority.MEDIUM: deque(),
            QueuePriority.LOW: deque(),
            QueuePriority.BACKLOG: deque(),
        }
        
        # Dead letter queue for failed items
        self.dead_letter_queue: deque = deque(maxlen=100)
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        
        # Statistics
        self.total_added = 0
        self.total_processed = 0
        self.total_dropped = 0
        self.total_failed = 0
        self.overflow_events = 0
        
        # Processing tracking
        self.processing_start_time: Optional[datetime] = None
        self.last_processing_time: Optional[datetime] = None
    
    async def put(self, item: T, priority: QueuePriority = QueuePriority.MEDIUM) -> bool:
        """
        Add item to queue with priority.
        
        Args:
            item: Item to add
            priority: Item priority level
            
        Returns:
            True if added successfully, False if dropped due to overflow
        """
        async with self._lock:
            current_size = self.size()
            
            # Check overflow
            if current_size >= self.max_size:
                self.overflow_events += 1
                
                if self.overflow_strategy == "drop_low_priority":
                    # Try to drop lowest priority items
                    if await self._drop_lowest_priority():
                        logger.warning(f"Queue {self.name} overflow: dropped low priority item")
                    else:
                        logger.error(f"Queue {self.name} overflow: rejecting {priority} item")
                        self.total_dropped += 1
                        return False
                
                elif self.overflow_strategy == "reject":
                    logger.error(f"Queue {self.name} full: rejecting item")
                    self.total_dropped += 1
                    return False
            
            # Create queue item
            queue_item = QueueItem(
                data=item,
                priority=priority,
                source=self.name
            )
            
            # Add to appropriate queue
            self.queues[priority].append(queue_item)
            self.total_added += 1
            
            # Notify waiting consumers
            self._not_empty.notify()
            
            logger.debug(f"Added {priority} item to {self.name} (size: {self.size()})")
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[QueueItem[T]]:
        """
        Get highest priority item from queue.
        
        Args:
            timeout: Maximum seconds to wait, None for infinite
            
        Returns:
            Queue item or None if timeout
        """
        async with self._not_empty:
            # Wait for items
            while self.is_empty():
                try:
                    if timeout is not None:
                        await asyncio.wait_for(self._not_empty.wait(), timeout)
                    else:
                        await self._not_empty.wait()
                except asyncio.TimeoutError:
                    return None
            
            # Get highest priority item
            for priority in [QueuePriority.CRITICAL, QueuePriority.HIGH, 
                            QueuePriority.MEDIUM, QueuePriority.LOW, 
                            QueuePriority.BACKLOG]:
                if self.queues[priority]:
                    item = self.queues[priority].popleft()
                    self.total_processed += 1
                    
                    # Track processing time
                    if self.processing_start_time is None:
                        self.processing_start_time = datetime.utcnow()
                    self.last_processing_time = datetime.utcnow()
                    
                    return item
            
            return None
    
    async def _drop_lowest_priority(self) -> bool:
        """Drop lowest priority item to make space."""
        # Try to drop in reverse priority order
        for priority in [QueuePriority.BACKLOG, QueuePriority.LOW, 
                        QueuePriority.MEDIUM]:
            if self.queues[priority]:
                dropped = self.queues[priority].popleft()
                self.total_dropped += 1
                logger.debug(f"Dropped {priority} item from {self.name}")
                return True
        
        return False
    
    def size(self) -> int:
        """Get total queue size."""
        return sum(len(q) for q in self.queues.values())
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of items by priority."""
        return {
            priority.value: len(self.queues[priority])
            for priority in QueuePriority
        }
    
    def get_oldest_item_age(self) -> float:
        """Get age of oldest item in seconds."""
        oldest_age = 0.0
        
        for priority_queue in self.queues.values():
            if priority_queue:
                # Check first item (oldest)
                age = priority_queue[0].age_seconds()
                oldest_age = max(oldest_age, age)
        
        return oldest_age
    
    def get_processing_rate(self) -> float:
        """Calculate items processed per second."""
        if not self.processing_start_time or not self.last_processing_time:
            return 0.0
        
        duration = (self.last_processing_time - self.processing_start_time).total_seconds()
        if duration <= 0:
            return 0.0
        
        return self.total_processed / duration
    
    def get_status(self) -> QueueStatus:
        """Determine queue operational status."""
        size = self.size()
        utilization = size / self.max_size
        
        if utilization >= 0.95:
            return QueueStatus.OVERFLOWING
        elif utilization >= 0.80:
            return QueueStatus.DEGRADED
        else:
            return QueueStatus.HEALTHY
    
    def get_metrics(self) -> QueueMetrics:
        """Get comprehensive queue metrics."""
        return QueueMetrics(
            queue_name=self.name,
            total_size=self.size(),
            priority_distribution=self.get_priority_distribution(),
            oldest_item_age=self.get_oldest_item_age(),
            processing_rate=self.get_processing_rate(),
            overflow_events=self.overflow_events,
            dropped_items=self.total_dropped,
            failed_items=self.total_failed,
            status=self.get_status()
        )
    
    async def move_to_dead_letter(self, item: QueueItem[T], reason: str):
        """Move failed item to dead letter queue."""
        item.last_error = reason
        self.dead_letter_queue.append(item)
        self.total_failed += 1
        
        logger.warning(f"Moved item to dead letter queue: {reason}")


class QueueManager:
    """
    Intelligent queue manager coordinating multiple priority queues.
    
    RESPONSIBILITIES:
    - Manage multiple named queues
    - Coordinate with memory manager for backpressure
    - Monitor queue health
    - Provide queue metrics
    - Handle overflow and failures
    
    Per .clinerules: Intelligent queue system with priority-based processing
    """
    
    def __init__(self):
        self.queues: Dict[str, PriorityQueue] = {}
        
        # Create standard queues
        self._initialize_standard_queues()
        
        # Monitoring
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info("Queue Manager initialized")
    
    async def initialize(self):
        """
        Async initialization for queue manager.
        Currently queues are sync, but this method exists for consistency.
        """
        logger.info("Queue Manager async initialization complete")
        return True
    
    def _initialize_standard_queues(self):
        """Initialize standard system queues."""
        # Findings processing queue
        self.queues['findings'] = PriorityQueue(
            name='findings',
            max_size=1000,
            overflow_strategy='drop_low_priority'
        )
        
        # Verification queue
        self.queues['verification'] = PriorityQueue(
            name='verification',
            max_size=500,
            overflow_strategy='drop_low_priority'
        )
        
        # Submission ready queue
        self.queues['submission_ready'] = PriorityQueue(
            name='submission_ready',
            max_size=200,
            overflow_strategy='reject'  # Don't drop submission-ready items
        )
        
        # Manual review queue
        self.queues['manual_review'] = PriorityQueue(
            name='manual_review',
            max_size=1000,
            overflow_strategy='drop_low_priority'
        )
        
        # Scan tasks queue (for CLI scan operations)
        self.queues['scan_tasks'] = PriorityQueue(
            name='scan_tasks',
            max_size=100,
            overflow_strategy='reject'
        )
        
        logger.info(f"Initialized {len(self.queues)} standard queues")
    
    def get_queue(self, name: str) -> Optional[PriorityQueue]:
        """Get queue by name."""
        return self.queues.get(name)
    
    def create_queue(self, name: str, max_size: int = 1000,
                    overflow_strategy: str = "drop_low_priority") -> PriorityQueue:
        """Create new queue."""
        if name in self.queues:
            logger.warning(f"Queue {name} already exists")
            return self.queues[name]
        
        queue = PriorityQueue(
            name=name,
            max_size=max_size,
            overflow_strategy=overflow_strategy
        )
        self.queues[name] = queue
        
        logger.info(f"Created queue: {name}")
        return queue
    
    async def enqueue_finding(self, queue_name: str, item: Any, priority: int = 2) -> bool:
        """
        Enqueue item to specified queue with priority mapping.
        
        Args:
            queue_name: Name of the queue
            item: Item to enqueue
            priority: Numeric priority (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW, 4=BACKLOG)
            
        Returns:
            True if enqueued successfully, False otherwise
        """
        queue = self.get_queue(queue_name)
        if not queue:
            # Create queue if doesn't exist
            queue = self.create_queue(queue_name)
        
        # Map numeric priority to QueuePriority enum
        priority_map = {
            0: QueuePriority.CRITICAL,
            1: QueuePriority.HIGH,
            2: QueuePriority.MEDIUM,
            3: QueuePriority.LOW,
            4: QueuePriority.BACKLOG
        }
        queue_priority = priority_map.get(priority, QueuePriority.MEDIUM)
        
        return await queue.put(item, queue_priority)
    
    async def dequeue(self, queue_name: str, timeout: Optional[float] = 1.0) -> Optional[Any]:
        """
        Dequeue item from specified queue.
        
        Args:
            queue_name: Name of the queue
            timeout: Maximum seconds to wait for an item
            
        Returns:
            Queue item or None if timeout/queue not found
        """
        queue = self.get_queue(queue_name)
        if not queue:
            return None
        
        queue_item = await queue.get(timeout=timeout)
        return queue_item if queue_item else None
    
    def get_all_metrics(self) -> Dict[str, QueueMetrics]:
        """Get metrics for all queues."""
        return {
            name: queue.get_metrics()
            for name, queue in self.queues.items()
        }
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall queue manager status."""
        metrics = self.get_all_metrics()
        
        total_size = sum(m.total_size for m in metrics.values())
        total_dropped = sum(m.dropped_items for m in metrics.values())
        total_failed = sum(m.failed_items for m in metrics.values())
        
        # Determine overall status
        statuses = [m.status for m in metrics.values()]
        if QueueStatus.OVERFLOWING in statuses:
            overall_status = QueueStatus.OVERFLOWING
        elif QueueStatus.DEGRADED in statuses:
            overall_status = QueueStatus.DEGRADED
        else:
            overall_status = QueueStatus.HEALTHY
        
        return {
            'total_queues': len(self.queues),
            'total_items': total_size,
            'total_dropped': total_dropped,
            'total_failed': total_failed,
            'overall_status': overall_status.value,
            'queue_metrics': {
                name: {
                    'size': m.total_size,
                    'status': m.status.value,
                    'processing_rate': m.processing_rate,
                    'oldest_age': m.oldest_item_age
                }
                for name, m in metrics.items()
            }
        }
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start queue health monitoring."""
        if self.monitoring_active:
            logger.warning("Queue monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        
        logger.info(f"Started queue monitoring (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop queue health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped queue monitoring")
    
    async def _monitoring_loop(self, interval: int):
        """Queue monitoring loop."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(interval)
                
                # Check each queue
                for name, queue in self.queues.items():
                    metrics = queue.get_metrics()
                    
                    if metrics.status == QueueStatus.OVERFLOWING:
                        logger.error(f"Queue {name} OVERFLOWING: {metrics.total_size} items")
                    
                    elif metrics.status == QueueStatus.DEGRADED:
                        logger.warning(f"Queue {name} DEGRADED: {metrics.total_size} items")
                    
                    # Alert on old items
                    if metrics.oldest_item_age > 3600:  # >1 hour
                        logger.warning(f"Queue {name} has items older than 1 hour")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue monitoring error: {e}", exc_info=True)
    
    async def apply_backpressure_if_needed(self) -> bool:
        """
        Check if backpressure should be applied based on queue health.
        
        Coordinates with memory manager for comprehensive backpressure.
        
        Returns:
            True if backpressure should be applied
        """
        metrics = self.get_all_metrics()
        
        # Apply backpressure if any queue is overflowing
        for metric in metrics.values():
            if metric.status == QueueStatus.OVERFLOWING:
                logger.warning(f"Applying backpressure due to queue overflow: {metric.queue_name}")
                return True
        
        # Apply backpressure if too many queues are degraded
        degraded_count = sum(1 for m in metrics.values() if m.status == QueueStatus.DEGRADED)
        if degraded_count >= len(self.queues) / 2:
            logger.warning(f"Applying backpressure: {degraded_count} queues degraded")
            return True
        
        return False
    
    async def shutdown(self):
        """
        Shutdown queue manager and cleanup resources.
        
        This method should be called during system shutdown to:
        - Stop monitoring
        - Log remaining items
        - Clean up resources
        """
        logger.info("Shutting down Queue Manager...")
        
        # Stop monitoring if active
        await self.stop_monitoring()
        
        # Log remaining items in queues
        for name, queue in self.queues.items():
            size = queue.size()
            if size > 0:
                logger.warning(f"Queue '{name}' has {size} unprocessed items at shutdown")
        
        logger.info("Queue Manager shutdown complete")


# Global queue manager instance
global_queue_manager = QueueManager()