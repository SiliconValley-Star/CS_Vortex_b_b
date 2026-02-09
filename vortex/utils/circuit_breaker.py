"""
VORTEX Circuit Breaker - V23.0
Fault tolerance pattern for service resilience

FEATURES:
- Three states: CLOSED, OPEN, HALF_OPEN
- Configurable failure thresholds
- Automatic recovery attempts
- Detailed health metrics
- Event callbacks
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: float = 60.0  # Time before attempting recovery
    half_open_max_calls: int = 3  # Max calls in half-open state
    
    # Optional callbacks
    on_open: Optional[Callable] = None
    on_close: Optional[Callable] = None
    on_half_open: Optional[Callable] = None


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    time_in_open: float = 0.0  # seconds
    time_in_half_open: float = 0.0
    time_in_closed: float = 0.0
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Implements the circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        
        # Timing
        self.state_changed_at = datetime.utcnow()
        self.last_failure_time: Optional[datetime] = None
        
        # Statistics
        self.stats = CircuitBreakerStats()
        
        logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        if new_state == self.state:
            return
        
        old_state = self.state
        self.state = new_state
        self.state_changed_at = datetime.utcnow()
        self.stats.state_changes += 1
        
        logger.info(f"Circuit breaker '{self.name}': {old_state} → {new_state}")
        
        # Call state change callbacks
        if new_state == CircuitState.OPEN and self.config.on_open:
            try:
                self.config.on_open(self)
            except Exception as e:
                logger.error(f"Error in on_open callback: {e}")
        
        elif new_state == CircuitState.CLOSED and self.config.on_close:
            try:
                self.config.on_close(self)
            except Exception as e:
                logger.error(f"Error in on_close callback: {e}")
        
        elif new_state == CircuitState.HALF_OPEN and self.config.on_half_open:
            try:
                self.config.on_half_open(self)
            except Exception as e:
                logger.error(f"Error in on_half_open callback: {e}")
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_calls = 0
            self.success_count = 0
    
    def _check_open_timeout(self):
        """Check if should transition from OPEN to HALF_OPEN."""
        if self.state != CircuitState.OPEN:
            return
        
        elapsed = (datetime.utcnow() - self.state_changed_at).total_seconds()
        if elapsed >= self.config.timeout_seconds:
            self._transition_to(CircuitState.HALF_OPEN)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker (sync).
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        # Check if should transition from OPEN to HALF_OPEN
        self._check_open_timeout()
        
        # OPEN state - reject calls
        if self.state == CircuitState.OPEN:
            self.stats.rejected_calls += 1
            retry_after = self.config.timeout_seconds - (
                datetime.utcnow() - self.state_changed_at
            ).total_seconds()
            raise CircuitBreakerOpen(self.name, max(0, retry_after))
        
        # HALF_OPEN state - limit calls
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.stats.rejected_calls += 1
                raise CircuitBreakerOpen(self.name, 0)
            self.half_open_calls += 1
        
        # Execute call
        self.stats.total_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function through circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        # Check if should transition from OPEN to HALF_OPEN
        self._check_open_timeout()
        
        # OPEN state - reject calls
        if self.state == CircuitState.OPEN:
            self.stats.rejected_calls += 1
            retry_after = self.config.timeout_seconds - (
                datetime.utcnow() - self.state_changed_at
            ).total_seconds()
            raise CircuitBreakerOpen(self.name, max(0, retry_after))
        
        # HALF_OPEN state - limit calls
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.stats.rejected_calls += 1
                raise CircuitBreakerOpen(self.name, 0)
            self.half_open_calls += 1
        
        # Execute call
        self.stats.total_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.stats.successful_calls += 1
        self.stats.last_success_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            # Enough successes to close
            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.stats.failed_calls += 1
        self.stats.last_failure_time = datetime.utcnow()
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open returns to open
            self._transition_to(CircuitState.OPEN)
        
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            
            # Too many failures, open circuit
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        logger.info(f"Manually resetting circuit breaker '{self.name}'")
        self._transition_to(CircuitState.CLOSED)
        self.failure_count = 0
        self.success_count = 0
    
    def get_state(self) -> CircuitState:
        """Get current state."""
        self._check_open_timeout()
        return self.state
    
    def is_available(self) -> bool:
        """Check if circuit breaker allows calls."""
        self._check_open_timeout()
        
        if self.state == CircuitState.OPEN:
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': self.stats.total_calls,
            'successful_calls': self.stats.successful_calls,
            'failed_calls': self.stats.failed_calls,
            'rejected_calls': self.stats.rejected_calls,
            'failure_rate': self.stats.get_failure_rate(),
            'success_rate': self.stats.get_success_rate(),
            'state_changes': self.stats.state_changes,
            'last_failure': self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            'last_success': self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
            'state_changed_at': self.state_changed_at.isoformat()
        }
    
    def __call__(self, func: Callable) -> Callable:
        """
        Use as decorator for sync functions.
        
        Usage:
            breaker = CircuitBreaker("my_service")
            
            @breaker
            def my_function():
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def __call_async__(self, func: Callable) -> Callable:
        """
        Use as decorator for async functions.
        
        Usage:
            breaker = CircuitBreaker("my_service")
            
            @breaker.__call_async__
            async def my_async_function():
                pass
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        return wrapper


class CircuitBreakerManager:
    """Manage multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        logger.info("Circuit breaker manager initialized")
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create circuit breaker.
        
        Args:
            name: Breaker name
            config: Configuration (used only if creating new breaker)
            
        Returns:
            Circuit breaker instance
        """
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        total_breakers = len(self.breakers)
        open_breakers = sum(1 for b in self.breakers.values() if b.get_state() == CircuitState.OPEN)
        half_open_breakers = sum(1 for b in self.breakers.values() if b.get_state() == CircuitState.HALF_OPEN)
        
        return {
            'total_breakers': total_breakers,
            'open': open_breakers,
            'half_open': half_open_breakers,
            'closed': total_breakers - open_breakers - half_open_breakers,
            'healthy': open_breakers == 0
        }


# Global circuit breaker manager
global_breaker_manager = CircuitBreakerManager()