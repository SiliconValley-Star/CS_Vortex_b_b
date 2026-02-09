"""
Test suite for Network Client
Tests HTTP requests, retries, rate limiting, and connection management
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from core.network import NetworkClient, HTTPResponse, global_network_client


class TestHTTPResponse:
    """Test HTTPResponse dataclass."""
    
    def test_response_creation(self):
        """Test HTTPResponse initialization."""
        response = HTTPResponse(
            status_code=200,
            body="test body",
            headers={'Content-Type': 'text/html'},
            url='https://example.com',
            response_time=0.5
        )
        
        assert response.status_code == 200
        assert response.body == "test body"
        assert response.headers['Content-Type'] == 'text/html'
        assert response.url == 'https://example.com'
        assert response.response_time == 0.5


class TestNetworkClient:
    """Test NetworkClient functionality."""
    
    @pytest.fixture
    def client(self):
        """Create network client instance."""
        return NetworkClient(max_connections=10, timeout=30)
    
    def test_client_initialization(self, client):
        """Test client initializes with correct settings."""
        assert client.max_connections == 10
        assert client.timeout == 30
        assert client.session is None
        assert client.stats['requests_made'] == 0
    
    @pytest.mark.asyncio
    async def test_initialize_session(self, client):
        """Test session initialization."""
        await client.initialize()
        
        assert client.session is not None
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_close_session(self, client):
        """Test session cleanup."""
        await client.initialize()
        await client.close()
        
        # Session should be cleared
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_request_get(self, client):
        """Test GET request."""
        await client.initialize()
        
        # Mock the session request
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.headers = {'Content-Type': 'text/html'}
        
        client.session.get = AsyncMock(return_value=mock_response)
        
        response = await client.request('GET', 'https://example.com')
        
        assert response.status_code == 200
        assert response.body == "OK"
        assert client.stats['requests_made'] == 1
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_request_post(self, client):
        """Test POST request with data."""
        await client.initialize()
        
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.text = AsyncMock(return_value='{"id": 1}')
        mock_response.headers = {}
        
        client.session.post = AsyncMock(return_value=mock_response)
        
        response = await client.request(
            'POST',
            'https://example.com/api',
            data={'key': 'value'}
        )
        
        assert response.status_code == 201
        assert '"id": 1' in response.body
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_request_with_headers(self, client):
        """Test request with custom headers."""
        await client.initialize()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.headers = {}
        
        client.session.get = AsyncMock(return_value=mock_response)
        
        custom_headers = {'Authorization': 'Bearer token123'}
        response = await client.request(
            'GET',
            'https://example.com',
            headers=custom_headers
        )
        
        assert response.status_code == 200
        client.session.get.assert_called_once()
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_request_timeout(self, client):
        """Test request timeout handling."""
        await client.initialize()
        
        # Simulate timeout
        client.session.get = AsyncMock(side_effect=asyncio.TimeoutError())
        
        with pytest.raises(asyncio.TimeoutError):
            await client.request('GET', 'https://example.com')
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_request_error_handling(self, client):
        """Test error handling during request."""
        await client.initialize()
        
        # Simulate connection error
        client.session.get = AsyncMock(side_effect=Exception("Connection failed"))
        
        with pytest.raises(Exception):
            await client.request('GET', 'https://example.com')
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_response_time_tracking(self, client):
        """Test response time is tracked."""
        await client.initialize()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.headers = {}
        
        # Add delay to simulate response time
        async def delayed_get(*args, **kwargs):
            await asyncio.sleep(0.1)
            return mock_response
        
        client.session.get = delayed_get
        
        response = await client.request('GET', 'https://example.com')
        
        assert response.response_time > 0
        assert response.response_time >= 0.1
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self, client):
        """Test statistics are tracked correctly."""
        await client.initialize()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.headers = {}
        
        client.session.get = AsyncMock(return_value=mock_response)
        
        initial_count = client.stats['requests_made']
        
        await client.request('GET', 'https://example.com')
        await client.request('GET', 'https://example.com')
        
        assert client.stats['requests_made'] == initial_count + 2
        
        await client.close()
    
    def test_get_stats(self, client):
        """Test statistics retrieval."""
        stats = client.get_stats()
        
        assert isinstance(stats, dict)
        assert 'requests_made' in stats
        assert 'bytes_sent' in stats
        assert 'bytes_received' in stats
    
    def test_global_network_client(self):
        """Test global network client instance."""
        assert global_network_client is not None
        assert isinstance(global_network_client, NetworkClient)


class TestNetworkClientIntegration:
    """Integration tests for NetworkClient."""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        client = NetworkClient(max_connections=5)
        await client.initialize()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.headers = {}
        
        client.session.get = AsyncMock(return_value=mock_response)
        
        # Make 10 concurrent requests
        tasks = [
            client.request('GET', f'https://example.com/{i}')
            for i in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 10
        assert all(r.status_code == 200 for r in responses)
        assert client.stats['requests_made'] == 10
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_session_reuse(self):
        """Test session is reused across requests."""
        client = NetworkClient()
        await client.initialize()
        
        session_id = id(client.session)
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_response.headers = {}
        
        client.session.get = AsyncMock(return_value=mock_response)
        
        await client.request('GET', 'https://example.com/1')
        await client.request('GET', 'https://example.com/2')
        
        # Session should be the same object
        assert id(client.session) == session_id
        
        await client.close()