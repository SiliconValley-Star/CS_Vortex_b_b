"""
Test suite for Retry Mechanism
Tests retry logic, backoff strategies, and decorators
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import time

from utils.retry import (
    RetryConfig, retry_async, retry_sync, 
    with_retry, with_retry_async,
    NETWORK_RETRY_CONFIG, DATABASE_RETRY_CONFIG, AI_RETRY_CONFIG
)


class TestRetryConfig:
    """Test RetryConfig dataclass."""
    
    def test_config_creation(self):
        """Test RetryConfig initialization."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_predefined_configs(self):
        """Test predefined retry configurations."""
        assert NETWORK_RETRY_CONFIG.max_attempts == 3
        assert DATABASE_RETRY_CONFIG.max_attempts == 3
        assert AI_RETRY_CONFIG.max_attempts == 5
        
        assert isinstance(NETWORK_RETRY_CONFIG, RetryConfig)
        assert isinstance(DATABASE_RETRY_CONFIG, RetryConfig)
        assert isinstance(AI_RETRY_CONFIG, RetryConfig)


class TestRetryAsync:
    """Test async retry functionality."""
    
    @pytest.mark.asyncio
    async def test_retry_async_success_first_attempt(self):
        """Test successful operation on first attempt."""
        mock_func = AsyncMock(return_value="success")
        config = RetryConfig(max_attempts=3)
        
        result = await retry_async(mock_func, config)
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_async_success_after_failures(self):
        """Test successful operation after retries."""
        mock_func = AsyncMock(side_effect=[
            Exception("Error 1"),
            Exception("Error 2"),
            "success"
        ])
        config = RetryConfig(max_attempts=5, base_delay=0.01)
        
        result = await retry_async(mock_func, config)
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_exhausted(self):
        """Test retry exhaustion."""
        mock_func = AsyncMock(side_effect=Exception("Persistent error"))
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        
        with pytest.raises(Exception, match="Persistent error"):
            await retry_async(mock_func, config)
        
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []
        
        async def failing_func():
            call_times.append(time.time())
            raise Exception("Test error")
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False
        )
        
        with pytest.raises(Exception):
            await retry_async(failing_func, config)
        
        assert len(call_times) == 3
        # Check delays are increasing
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1
    
    @pytest.mark.asyncio
    async def test_retry_async_with_args(self):
        """Test retry with function arguments."""
        mock_func = AsyncMock(return_value="result")
        config = RetryConfig(max_attempts=3)
        
        result = await retry_async(
            mock_func,
            config,
            "arg1", "arg2",
            kwarg1="value1"
        )
        
        assert result == "result"
        mock_func.assert_called_with("arg1", "arg2", kwarg1="value1")


class TestRetrySync:
    """Test sync retry functionality."""
    
    def test_retry_sync_success(self):
        """Test successful sync operation."""
        mock_func = Mock(return_value="success")
        config = RetryConfig(max_attempts=3)
        
        result = retry_sync(mock_func, config)
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_sync_with_failures(self):
        """Test sync retry with failures."""
        mock_func = Mock(side_effect=[
            Exception("Error 1"),
            "success"
        ])
        config = RetryConfig(max_attempts=5, base_delay=0.01)
        
        result = retry_sync(mock_func, config)
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_sync_exhausted(self):
        """Test sync retry exhaustion."""
        mock_func = Mock(side_effect=Exception("Persistent error"))
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        
        with pytest.raises(Exception, match="Persistent error"):
            retry_sync(mock_func, config)
        
        assert mock_func.call_count == 3


class TestRetryDecorators:
    """Test retry decorators."""
    
    @pytest.mark.asyncio
    async def test_with_retry_async_decorator(self):
        """Test async retry decorator."""
        call_count = 0
        
        @with_retry_async(RetryConfig(max_attempts=3, base_delay=0.01))
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return "success"
        
        result = await flaky_func()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_with_retry_async_with_args(self):
        """Test async decorator with arguments."""
        @with_retry_async(RetryConfig(max_attempts=3))
        async def func_with_args(x, y, z=None):
            return f"{x}-{y}-{z}"
        
        result = await func_with_args("a", "b", z="c")
        
        assert result == "a-b-c"
    
    def test_with_retry_sync_decorator(self):
        """Test sync retry decorator."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.01))
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert call_count == 2


class TestRetryIntegration:
    """Integration tests for retry mechanism."""
    
    @pytest.mark.asyncio
    async def test_network_retry_config(self):
        """Test network retry configuration."""
        attempt_count = 0
        
        async def simulate_network_request():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("Network error")
            return "Network response"
        
        result = await retry_async(simulate_network_request, NETWORK_RETRY_CONFIG)
        
        assert result == "Network response"
        assert attempt_count == 2
    
    @pytest.mark.asyncio
    async def test_database_retry_config(self):
        """Test database retry configuration."""
        attempt_count = 0
        
        async def simulate_db_query():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise Exception("DB connection lost")
            return "Query result"
        
        result = await retry_async(simulate_db_query, DATABASE_RETRY_CONFIG)
        
        assert result == "Query result"
        assert attempt_count == 2
    
    @pytest.mark.asyncio
    async def test_ai_retry_config(self):
        """Test AI call retry configuration."""
        attempt_count = 0
        
        async def simulate_ai_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("AI service timeout")
            return "AI response"
        
        result = await retry_async(simulate_ai_call, AI_RETRY_CONFIG)
        
        assert result == "AI response"
        assert attempt_count == 3