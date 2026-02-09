"""
Test suite for Resilience Coordinator
Tests unified resilience patterns (retry + circuit breaker + fallback)
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from core.resilience import (
    ResilienceCoordinator, ResilienceConfig,
    with_resilience_async, with_resilience_sync
)
from utils.retry import RetryConfig
from utils.circuit_breaker import CircuitBreakerConfig, CircuitBreakerState


class TestResilienceConfig:
    """Test ResilienceConfig dataclass."""
    
    def test_config_creation(self):
        """Test config initialization."""
        config = ResilienceConfig(
            retry_config=RetryConfig(max_attempts=3),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
            fallback_enabled=True,
            timeout=30.0
        )
        
        assert config.retry_config.max_attempts == 3
        assert config.circuit_breaker_config.failure_threshold == 5
        assert config.fallback_enabled is True
        assert config.timeout == 30.0


class TestResilienceCoordinator:
    """Test ResilienceCoordinator functionality."""
    
    @pytest.fixture
    def coordinator(self):
        """Create resilience coordinator."""
        return ResilienceCoordinator("test_service")
    
    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes correctly."""
        assert coordinator.service_name == "test_service"
        assert coordinator.retry_enabled is True
        assert coordinator.circuit_breaker_enabled is True
        assert coordinator.fallback is None
    
    @pytest.mark.asyncio
    async def test_execute_async_success(self, coordinator):
        """Test successful async execution."""
        mock_func = AsyncMock(return_value="success")
        
        result = await coordinator.execute_async(mock_func)
        
        assert result == "success"
        mock_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_async_with_retry(self, coordinator):
        """Test execution with retry on failure."""
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return "success"
        
        result = await coordinator.execute_async(flaky_func)
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_async_with_fallback(self, coordinator):
        """Test fallback execution on failure."""
        mock_func = AsyncMock(side_effect=Exception("Primary failed"))
        mock_fallback = AsyncMock(return_value="fallback_result")
        
        coordinator.set_fallback(mock_fallback)
        
        result = await coordinator.execute_async(mock_func)
        
        assert result == "fallback_result"
        mock_fallback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_async_with_timeout(self, coordinator):
        """Test timeout handling."""
        async def slow_func():
            await asyncio.sleep(10)
            return "too_late"
        
        coordinator.timeout = 0.1
        
        with pytest.raises(asyncio.TimeoutError):
            await coordinator.execute_async(slow_func)
    
    @pytest.mark.asyncio
    async def test_execute_async_circuit_breaker(self, coordinator):
        """Test circuit breaker integration."""
        mock_func = AsyncMock(side_effect=Exception("Service error"))
        
        # Trigger multiple failures to open circuit
        for _ in range(5):
            try:
                await coordinator.execute_async(mock_func)
            except Exception:
                pass
        
        # Circuit should be open
        breaker = coordinator.breaker_manager.get_breaker(coordinator.service_name)
        assert breaker.state == CircuitBreakerState.OPEN
    
    def test_set_fallback(self, coordinator):
        """Test setting fallback function."""
        def fallback_func():
            return "fallback"
        
        coordinator.set_fallback(fallback_func)
        
        assert coordinator.fallback is fallback_func
        assert coordinator.fallback_enabled is True
    
    def test_disable_retry(self, coordinator):
        """Test disabling retry."""
        coordinator.disable_retry()
        
        assert coordinator.retry_enabled is False
    
    def test_disable_circuit_breaker(self, coordinator):
        """Test disabling circuit breaker."""
        coordinator.disable_circuit_breaker()
        
        assert coordinator.circuit_breaker_enabled is False
    
    def test_get_stats(self, coordinator):
        """Test statistics retrieval."""
        stats = coordinator.get_stats()
        
        assert isinstance(stats, dict)
        assert 'service_name' in stats
        assert 'retry_enabled' in stats
        assert 'circuit_breaker_enabled' in stats


class TestResilienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_with_resilience_async_basic(self):
        """Test basic async resilience function."""
        mock_func = AsyncMock(return_value="success")
        
        result = await with_resilience_async(
            mock_func,
            service_name="test"
        )
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_with_resilience_async_with_retry(self):
        """Test async function with retry."""
        call_count = 0
        
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return "success"
        
        result = await with_resilience_async(
            flaky_func,
            service_name="test",
            retry_attempts=3
        )
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_with_resilience_async_with_fallback(self):
        """Test async function with fallback."""
        mock_func = AsyncMock(side_effect=Exception("Primary failed"))
        fallback_func = AsyncMock(return_value="fallback")
        
        result = await with_resilience_async(
            mock_func,
            service_name="test",
            fallback=fallback_func,
            retry_attempts=1
        )
        
        assert result == "fallback"
    
    @pytest.mark.asyncio
    async def test_with_resilience_async_with_timeout(self):
        """Test async function with timeout."""
        async def slow_func():
            await asyncio.sleep(10)
            return "too_late"
        
        with pytest.raises(asyncio.TimeoutError):
            await with_resilience_async(
                slow_func,
                service_name="test",
                timeout=0.1
            )
    
    def test_with_resilience_sync_basic(self):
        """Test basic sync resilience function."""
        mock_func = Mock(return_value="success")
        
        result = with_resilience_sync(
            mock_func,
            service_name="test"
        )
        
        assert result == "success"
    
    def test_with_resilience_sync_with_fallback(self):
        """Test sync function with fallback."""
        mock_func = Mock(side_effect=Exception("Primary failed"))
        fallback_func = Mock(return_value="fallback")
        
        result = with_resilience_sync(
            mock_func,
            service_name="test",
            fallback=fallback_func,
            retry_attempts=1
        )
        
        assert result == "fallback"


class TestResilienceIntegration:
    """Integration tests for resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_full_resilience_workflow(self):
        """Test complete resilience workflow."""
        call_count = 0
        fallback_called = False
        
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return "primary_result"
        
        async def fallback_service():
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"
        
        # Should succeed with retry
        result = await with_resilience_async(
            flaky_service,
            service_name="test",
            retry_attempts=3,
            fallback=fallback_service
        )
        
        assert result == "primary_result"
        assert call_count == 2
        assert fallback_called is False
    
    @pytest.mark.asyncio
    async def test_resilience_with_circuit_breaker_open(self):
        """Test behavior when circuit breaker is open."""
        coordinator = ResilienceCoordinator("test")
        
        # Open circuit by triggering failures
        mock_fail = AsyncMock(side_effect=Exception("Service down"))
        for _ in range(5):
            try:
                await coordinator.execute_async(mock_fail)
            except:
                pass
        
        # Circuit should be open
        breaker = coordinator.breaker_manager.get_breaker("test")
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Next call should use fallback immediately
        mock_fallback = AsyncMock(return_value="fallback")
        coordinator.set_fallback(mock_fallback)
        
        result = await coordinator.execute_async(mock_fail)
        assert result == "fallback"