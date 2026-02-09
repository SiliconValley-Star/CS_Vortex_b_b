"""
VORTEX Resilience Coordinator - V23.0
Unified resilience patterns for production stability

FEATURES:
- Combines retry + circuit breaker
- Timeout management
- Fallback strategies
- Health monitoring
- Graceful degradation
"""

import asyncio
import logging
from typing import Optional, Callable, Any, Dict
from datetime import datetime
from dataclasses import dataclass

from utils.retry import RetryConfig, retry_async, retry_sync, RetryExhausted
from utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen,
    CircuitState, global_breaker_manager
)

logger = logging.getLogger(__name__)


@dataclass
class ResilienceConfig:
    """Combined resilience configuration."""
    # Retry settings
    retry_config: Optional[RetryConfig] = None
    
    # Circuit breaker settings
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    circuit_breaker_name: str = "default"
    
    # Fallback
    fallback: Optional[Callable] = None
    
    # Timeout
    timeout_seconds: Optional[float] = None


class ResilienceError(Exception):
    """Base exception for resilience errors."""
    pass


class OperationFailedError(ResilienceError):
    """Raised when operation fails after all resilience attempts."""
    
    def __init__(self, operation: str, original_error: Exception):
        self.operation = operation
        self.original_error = original_error
        super().__init__(
            f"Operation '{operation}' failed: {type(original_error).__name__}: {original_error}"
        )


class ResilienceCoordinator:
    """
    Coordinates resilience patterns for robust operations.
    
    Combines:
    - Retry logic with exponential backoff
    - Circuit breaker for cascading failure prevention
    - Timeout management
    - Fallback strategies
    """
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        
        # Get or create circuit breaker
        if self.config.circuit_breaker_config:
            self.circuit_breaker = global_breaker_manager.get_breaker(
                self.config.circuit_breaker_name,
                self.config.circuit_breaker_config
            )
        else:
            self.circuit_breaker = None
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_breaker_rejections': 0,
            'fallback_executions': 0,
            'timeouts': 0
        }
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with full resilience.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result or fallback result
            
        Raises:
            OperationFailedError: If all resilience mechanisms fail
        """
        self.stats['total_calls'] += 1
        operation_name = func.__name__
        
        try:
            # Wrap in timeout if configured
            if self.config.timeout_seconds:
                result = await asyncio.wait_for(
                    self._execute_with_resilience_async(func, *args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
            else:
                result = await self._execute_with_resilience_async(func, *args, **kwargs)
            
            self.stats['successful_calls'] += 1
            return result
            
        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.error(f"Operation '{operation_name}' timed out after {self.config.timeout_seconds}s")
            
            # Try fallback
            if self.config.fallback:
                return await self._execute_fallback_async(operation_name, *args, **kwargs)
            raise
            
        except (CircuitBreakerOpen, RetryExhausted) as e:
            self.stats['failed_calls'] += 1
            
            # Try fallback
            if self.config.fallback:
                return await self._execute_fallback_async(operation_name, *args, **kwargs)
            
            raise OperationFailedError(operation_name, e)
            
        except Exception as e:
            self.stats['failed_calls'] += 1
            logger.error(f"Operation '{operation_name}' failed: {e}")
            
            # Try fallback
            if self.config.fallback:
                return await self._execute_fallback_async(operation_name, *args, **kwargs)
            
            raise OperationFailedError(operation_name, e)
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute sync function with full resilience.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result or fallback result
            
        Raises:
            OperationFailedError: If all resilience mechanisms fail
        """
        self.stats['total_calls'] += 1
        operation_name = func.__name__
        
        try:
            result = self._execute_with_resilience_sync(func, *args, **kwargs)
            self.stats['successful_calls'] += 1
            return result
            
        except (CircuitBreakerOpen, RetryExhausted) as e:
            self.stats['failed_calls'] += 1
            
            # Try fallback
            if self.config.fallback:
                return self._execute_fallback_sync(operation_name, *args, **kwargs)
            
            raise OperationFailedError(operation_name, e)
            
        except Exception as e:
            self.stats['failed_calls'] += 1
            logger.error(f"Operation '{operation_name}' failed: {e}")
            
            # Try fallback
            if self.config.fallback:
                return self._execute_fallback_sync(operation_name, *args, **kwargs)
            
            raise OperationFailedError(operation_name, e)
    
    async def _execute_with_resilience_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with retry and circuit breaker."""
        
        async def wrapped_execution():
            # Circuit breaker first
            if self.circuit_breaker:
                return await self.circuit_breaker.call_async(func, *args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        # Retry wrapper
        if self.config.retry_config:
            return await retry_async(wrapped_execution, config=self.config.retry_config)
        else:
            return await wrapped_execution()
    
    def _execute_with_resilience_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with retry and circuit breaker (sync)."""
        
        def wrapped_execution():
            # Circuit breaker first
            if self.circuit_breaker:
                return self.circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        # Retry wrapper
        if self.config.retry_config:
            return retry_sync(wrapped_execution, config=self.config.retry_config)
        else:
            return wrapped_execution()
    
    async def _execute_fallback_async(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute fallback function."""
        self.stats['fallback_executions'] += 1
        logger.warning(f"Executing fallback for '{operation_name}'")
        
        try:
            if asyncio.iscoroutinefunction(self.config.fallback):
                return await self.config.fallback(*args, **kwargs)
            else:
                return self.config.fallback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            raise
    
    def _execute_fallback_sync(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute fallback function (sync)."""
        self.stats['fallback_executions'] += 1
        logger.warning(f"Executing fallback for '{operation_name}'")
        
        try:
            return self.config.fallback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        stats = self.stats.copy()
        
        # Add circuit breaker stats if available
        if self.circuit_breaker:
            stats['circuit_breaker'] = self.circuit_breaker.get_stats()
        
        # Calculate success rate
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_breaker_rejections': 0,
            'fallback_executions': 0,
            'timeouts': 0
        }
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        # Check circuit breaker state
        if self.circuit_breaker:
            if self.circuit_breaker.get_state() == CircuitState.OPEN:
                return False
        
        # Check success rate
        if self.stats['total_calls'] >= 10:  # Minimum calls for meaningful rate
            success_rate = self.stats['successful_calls'] / self.stats['total_calls']
            if success_rate < 0.5:  # Below 50% success rate
                return False
        
        return True


# Convenience functions for common patterns

async def with_resilience_async(
    func: Callable,
    *args,
    retry_attempts: int = 3,
    circuit_breaker_name: Optional[str] = None,
    fallback: Optional[Callable] = None,
    timeout: Optional[float] = None,
    **kwargs
) -> Any:
    """
    Execute async function with resilience (convenience function).
    
    Args:
        func: Async function to execute
        retry_attempts: Number of retry attempts
        circuit_breaker_name: Circuit breaker name (optional)
        fallback: Fallback function (optional)
        timeout: Timeout in seconds (optional)
        
    Returns:
        Function result
    """
    config = ResilienceConfig(
        retry_config=RetryConfig(max_attempts=retry_attempts),
        circuit_breaker_name=circuit_breaker_name or func.__name__,
        fallback=fallback,
        timeout_seconds=timeout
    )
    
    coordinator = ResilienceCoordinator(config)
    return await coordinator.execute_async(func, *args, **kwargs)


def with_resilience_sync(
    func: Callable,
    *args,
    retry_attempts: int = 3,
    circuit_breaker_name: Optional[str] = None,
    fallback: Optional[Callable] = None,
    **kwargs
) -> Any:
    """
    Execute sync function with resilience (convenience function).
    
    Args:
        func: Function to execute
        retry_attempts: Number of retry attempts
        circuit_breaker_name: Circuit breaker name (optional)
        fallback: Fallback function (optional)
        
    Returns:
        Function result
    """
    config = ResilienceConfig(
        retry_config=RetryConfig(max_attempts=retry_attempts),
        circuit_breaker_name=circuit_breaker_name or func.__name__,
        fallback=fallback
    )
    
    coordinator = ResilienceCoordinator(config)
    return coordinator.execute_sync(func, *args, **kwargs)


# Global resilience coordinator for shared use
global_resilience = ResilienceCoordinator()