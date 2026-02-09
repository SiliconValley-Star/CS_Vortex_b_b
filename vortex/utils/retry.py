"""
VORTEX Retry Mechanism - V23.0
Intelligent retry logic with exponential backoff

FEATURES:
- Configurable retry strategies
- Exponential backoff with jitter
- Retry condition customization
- Timeout handling
- Detailed retry statistics
"""

import asyncio
import logging
import time
import random
from typing import Optional, Callable, Any, Type, Tuple, List
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    timeout: Optional[float] = None  # seconds
    
    # Retryable exceptions
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    
    # Non-retryable exceptions (takes precedence)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (
        KeyboardInterrupt,
        SystemExit,
        MemoryError,
    )


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    exception: Optional[Exception]
    delay: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = False


@dataclass
class RetryStats:
    """Statistics about retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay_seconds: float = 0.0
    attempts_history: List[RetryAttempt] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    def get_average_delay(self) -> float:
        """Calculate average delay between retries."""
        if not self.attempts_history:
            return 0.0
        return self.total_delay_seconds / len(self.attempts_history)


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )


class RetryContext:
    """Context for managing retry state."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.stats = RetryStats()
        self.current_attempt = 0
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if operation should be retried.
        
        Args:
            exception: Exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        # Check max attempts
        if self.current_attempt >= self.config.max_attempts:
            return False
        
        # Check non-retryable exceptions first
        if isinstance(exception, self.config.non_retryable_exceptions):
            logger.debug(f"Non-retryable exception: {type(exception).__name__}")
            return False
        
        # Check retryable exceptions
        if isinstance(exception, self.config.retryable_exceptions):
            return True
        
        return False
    
    def calculate_delay(self) -> float:
        """
        Calculate delay before next retry.
        
        Uses exponential backoff with optional jitter.
        """
        # Exponential backoff
        delay = self.config.base_delay * (
            self.config.exponential_base ** (self.current_attempt - 1)
        )
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled (±25%)
        if self.config.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Ensure positive
        
        return delay
    
    def record_attempt(self, exception: Optional[Exception], delay: float, success: bool):
        """Record retry attempt."""
        attempt = RetryAttempt(
            attempt_number=self.current_attempt,
            exception=exception,
            delay=delay,
            success=success
        )
        
        self.stats.attempts_history.append(attempt)
        self.stats.total_attempts += 1
        self.stats.total_delay_seconds += delay
        
        if success:
            self.stats.successful_attempts += 1
        else:
            self.stats.failed_attempts += 1


async def retry_async(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Execute async function with retry logic.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result
        
    Raises:
        RetryExhausted: If all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    context = RetryContext(config)
    last_exception = None
    
    start_time = time.time()
    
    for attempt in range(1, config.max_attempts + 1):
        context.current_attempt = attempt
        
        try:
            # Check timeout
            if config.timeout:
                elapsed = time.time() - start_time
                if elapsed >= config.timeout:
                    raise TimeoutError(f"Retry timeout after {elapsed:.1f}s")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Success
            context.record_attempt(None, 0.0, True)
            
            if attempt > 1:
                logger.info(f"Retry succeeded on attempt {attempt}/{config.max_attempts}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if should retry
            if not context.should_retry(e):
                logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
                raise
            
            # Calculate delay
            if attempt < config.max_attempts:
                delay = context.calculate_delay()
                context.record_attempt(e, delay, False)
                
                logger.warning(
                    f"Attempt {attempt}/{config.max_attempts} failed: {type(e).__name__}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
            else:
                # Final attempt failed
                context.record_attempt(e, 0.0, False)
    
    # All retries exhausted
    raise RetryExhausted(config.max_attempts, last_exception)


def retry_sync(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Execute sync function with retry logic.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result
        
    Raises:
        RetryExhausted: If all retries fail
    """
    if config is None:
        config = RetryConfig()
    
    context = RetryContext(config)
    last_exception = None
    
    start_time = time.time()
    
    for attempt in range(1, config.max_attempts + 1):
        context.current_attempt = attempt
        
        try:
            # Check timeout
            if config.timeout:
                elapsed = time.time() - start_time
                if elapsed >= config.timeout:
                    raise TimeoutError(f"Retry timeout after {elapsed:.1f}s")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Success
            context.record_attempt(None, 0.0, True)
            
            if attempt > 1:
                logger.info(f"Retry succeeded on attempt {attempt}/{config.max_attempts}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if non-retryable (takes precedence)
            if isinstance(e, config.non_retryable_exceptions):
                logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
                context.record_attempt(e, 0.0, False)
                raise
            
            # Check if not in retryable exceptions
            if not isinstance(e, config.retryable_exceptions):
                logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
                context.record_attempt(e, 0.0, False)
                raise
            
            # This is retryable - check if we have more attempts
            if attempt < config.max_attempts:
                delay = context.calculate_delay()
                context.record_attempt(e, delay, False)
                
                logger.warning(
                    f"Attempt {attempt}/{config.max_attempts} failed: {type(e).__name__}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                time.sleep(delay)
            else:
                # Final attempt failed
                context.record_attempt(e, 0.0, False)
    
    # All retries exhausted
    raise RetryExhausted(config.max_attempts, last_exception)


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to sync functions.
    
    Args:
        config: Retry configuration
        
    Usage:
        @with_retry(RetryConfig(max_attempts=5))
        def my_function():
            pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_sync(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


def with_retry_async(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to async functions.
    
    Args:
        config: Retry configuration
        
    Usage:
        @with_retry_async(RetryConfig(max_attempts=5))
        async def my_async_function():
            pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


# Common retry configurations
NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=10.0,
    exponential_base=1.5,
    jitter=True
)

AI_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)