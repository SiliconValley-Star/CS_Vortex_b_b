"""
Test suite for Circuit Breaker Pattern
Tests circuit breaker states, failure detection, and recovery
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import time

from utils.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
    CircuitBreakerManager, global_breaker_manager
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""
    
    def test_config_creation(self):
        """Test config initialization."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout=60.0,
            expected_exception=Exception
        )
        
        assert config.failure_threshold == 5
        assert config.timeout == 60.0
        assert config.expected_exception == Exception


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    @pytest.fixture
    def breaker(self):
        """Create circuit breaker instance."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout=1.0
        )
        return CircuitBreaker("test_service", config)
    
    def test_breaker_initialization(self, breaker):
        """Test breaker initializes in CLOSED state."""
        assert breaker.name == "test_service"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
    
    @pytest.mark.asyncio
    async def test_call_async_success(self, breaker):
        """Test successful async call."""
        mock_func = AsyncMock(return_value="success")
        
        result = await breaker.call_async(mock_func)
        
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.success_count == 1
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_call_async_failure(self, breaker):
        """Test failed async call."""
        mock_func = AsyncMock(side_effect=Exception("Test error"))
        
        with pytest.raises(Exception, match="Test error"):
            await breaker.call_async(mock_func)
        
        assert breaker.failure_count == 1
        assert breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, breaker):
        """Test circuit opens after failure threshold."""
        mock_func = AsyncMock(side_effect=Exception("Test error"))
        
        # Trigger failures up to threshold
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(mock_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.failure_count == 3
    
    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, breaker):
        """Test open circuit rejects calls immediately."""
        mock_func = AsyncMock(side_effect=Exception("Test error"))
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(mock_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Next call should be rejected without executing
        mock_success = AsyncMock(return_value="success")
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await breaker.call_async(mock_success)
        
        mock_success.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_circuit_moves_to_half_open(self, breaker):
        """Test circuit moves to HALF_OPEN after timeout."""
        mock_func = AsyncMock(side_effect=Exception("Test error"))
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(mock_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(1.1)
        
        # Next call should trigger HALF_OPEN state
        mock_test = AsyncMock(return_value="test")
        result = await breaker.call_async(mock_test)
        
        assert result == "test"
        assert breaker.state == CircuitBreakerState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, breaker):
        """Test successful call in HALF_OPEN closes circuit."""
        # Open the circuit
        mock_fail = AsyncMock(side_effect=Exception("Test error"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(mock_fail)
        
        # Wait and move to HALF_OPEN
        await asyncio.sleep(1.1)
        
        # Successful call should close circuit
        mock_success = AsyncMock(return_value="success")
        result = await breaker.call_async(mock_success)
        
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, breaker):
        """Test failure in HALF_OPEN reopens circuit."""
        # Open the circuit
        mock_fail = AsyncMock(side_effect=Exception("Test error"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(mock_fail)
        
        # Wait and move to HALF_OPEN
        await asyncio.sleep(1.1)
        
        # Failed call should reopen circuit
        with pytest.raises(Exception):
            await breaker.call_async(mock_fail)
        
        assert breaker.state == CircuitBreakerState.OPEN
    
    def test_reset(self, breaker):
        """Test circuit breaker reset."""
        breaker.failure_count = 5
        breaker.success_count = 3
        breaker.state = CircuitBreakerState.OPEN
        
        breaker.reset()
        
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
    
    def test_get_stats(self, breaker):
        """Test statistics retrieval."""
        breaker.failure_count = 2
        breaker.success_count = 5
        
        stats = breaker.get_stats()
        
        assert isinstance(stats, dict)
        assert stats['state'] == 'CLOSED'
        assert stats['failure_count'] == 2
        assert stats['success_count'] == 5


class TestCircuitBreakerManager:
    """Test CircuitBreakerManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create circuit breaker manager."""
        return CircuitBreakerManager()
    
    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert isinstance(manager.breakers, dict)
    
    def test_get_breaker_creates_new(self, manager):
        """Test getting breaker creates it if not exists."""
        breaker = manager.get_breaker("new_service")
        
        assert breaker is not None
        assert breaker.name == "new_service"
        assert "new_service" in manager.breakers
    
    def test_get_breaker_returns_existing(self, manager):
        """Test getting breaker returns existing instance."""
        breaker1 = manager.get_breaker("service")
        breaker2 = manager.get_breaker("service")
        
        assert breaker1 is breaker2
    
    def test_get_breaker_with_custom_config(self, manager):
        """Test getting breaker with custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = manager.get_breaker("service", config)
        
        assert breaker.config.failure_threshold == 10
    
    def test_reset_breaker(self, manager):
        """Test resetting specific breaker."""
        breaker = manager.get_breaker("service")
        breaker.failure_count = 5
        breaker.state = CircuitBreakerState.OPEN
        
        manager.reset_breaker("service")
        
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
    
    def test_reset_all_breakers(self, manager):
        """Test resetting all breakers."""
        breaker1 = manager.get_breaker("service1")
        breaker2 = manager.get_breaker("service2")
        
        breaker1.failure_count = 5
        breaker2.failure_count = 3
        
        manager.reset_all()
        
        assert breaker1.failure_count == 0
        assert breaker2.failure_count == 0
    
    def test_get_all_stats(self, manager):
        """Test getting stats for all breakers."""
        manager.get_breaker("service1")
        manager.get_breaker("service2")
        
        stats = manager.get_all_stats()
        
        assert isinstance(stats, dict)
        assert "service1" in stats
        assert "service2" in stats
    
    def test_global_breaker_manager(self):
        """Test global manager instance."""
        assert global_breaker_manager is not None
        assert isinstance(global_breaker_manager, CircuitBreakerManager)


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_full_circuit_lifecycle(self):
        """Test complete circuit breaker lifecycle."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.5
        )
        breaker = CircuitBreaker("test", config)
        
        # 1. CLOSED state - successful calls
        mock_success = AsyncMock(return_value="OK")
        result = await breaker.call_async(mock_success)
        assert result == "OK"
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # 2. Trigger failures to OPEN
        mock_fail = AsyncMock(side_effect=Exception("Error"))
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call_async(mock_fail)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # 3. Wait for timeout to HALF_OPEN
        await asyncio.sleep(0.6)
        
        # 4. Successful call closes circuit
        result = await breaker.call_async(mock_success)
        assert result == "OK"
        assert breaker.state == CircuitBreakerState.CLOSED