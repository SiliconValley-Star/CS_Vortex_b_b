"""
Test Suite for PHASE 3.1: Proxy Integration
Tests FREE proxy methods (Tor, proxy list, SOCKS5/HTTP)
"""

import pytest
import asyncio
from pathlib import Path

# Test proxy integration
def test_proxy_manager_import():
    """Test that ProxyManager can be imported."""
    from core.stealth.evasion import ProxyManager
    assert ProxyManager is not None

def test_proxy_manager_creation():
    """Test ProxyManager instantiation."""
    from core.stealth.evasion import ProxyManager
    
    pm = ProxyManager()
    assert pm is not None
    assert pm.proxies == []
    assert pm.current_index == 0

def test_add_tor_proxy():
    """Test Tor proxy addition (FREE)."""
    from core.stealth.evasion import ProxyManager
    
    pm = ProxyManager()
    pm.add_tor_proxy()
    
    assert len(pm.proxies) == 1
    assert pm.proxies[0].protocol == "socks5"
    assert pm.proxies[0].host == "127.0.0.1"
    assert pm.proxies[0].port == 9050

def test_add_single_proxy():
    """Test single proxy addition (FREE)."""
    from core.stealth.evasion import ProxyManager
    
    pm = ProxyManager()
    pm.add_proxy("http", "127.0.0.1", 8080)
    
    assert len(pm.proxies) == 1
    assert pm.proxies[0].protocol == "http"
    assert pm.proxies[0].url == "http://127.0.0.1:8080"

def test_add_proxy_with_auth():
    """Test proxy with authentication (FREE)."""
    from core.stealth.evasion import ProxyManager
    
    pm = ProxyManager()
    pm.add_proxy("http", "proxy.example.com", 3128, "user", "pass")
    
    assert len(pm.proxies) == 1
    assert pm.proxies[0].username == "user"
    assert pm.proxies[0].password == "pass"
    assert pm.proxies[0].url == "http://user:pass@proxy.example.com:3128"

def test_load_proxy_list():
    """Test loading proxy list from file (FREE)."""
    from core.stealth.evasion import ProxyManager
    import tempfile
    
    # Create temp proxy list
    proxy_content = """# Test proxies
1.2.3.4:8080
5.6.7.8:3128:user:pass
socks5://9.10.11.12:1080
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(proxy_content)
        temp_path = f.name
    
    try:
        pm = ProxyManager()
        pm.load_proxy_list(temp_path, protocol="http")
        
        # Should load 3 proxies (1 comment line ignored)
        assert len(pm.proxies) == 3
        
        # Check first proxy
        assert pm.proxies[0].host == "1.2.3.4"
        assert pm.proxies[0].port == 8080
        
        # Check authenticated proxy
        assert pm.proxies[1].username == "user"
        assert pm.proxies[1].password == "pass"
        
        # Check SOCKS5 proxy (protocol auto-detected)
        assert pm.proxies[2].protocol == "socks5"
        
    finally:
        import os
        os.unlink(temp_path)

def test_proxy_rotation():
    """Test proxy rotation mechanism (FREE)."""
    from core.stealth.evasion import ProxyManager
    
    pm = ProxyManager()
    pm.add_proxy("http", "proxy1.com", 8080)
    pm.add_proxy("http", "proxy2.com", 8080)
    pm.add_proxy("http", "proxy3.com", 8080)
    
    # Get proxies in rotation
    p1 = pm.get_next_proxy()
    p2 = pm.get_next_proxy()
    p3 = pm.get_next_proxy()
    p4 = pm.get_next_proxy()  # Should wrap around
    
    assert p1.host == "proxy1.com"
    assert p2.host == "proxy2.com"
    assert p3.host == "proxy3.com"
    assert p4.host == "proxy1.com"  # Wrapped

def test_proxy_ban_mechanism():
    """Test temporary proxy banning (FREE)."""
    from core.stealth.evasion import ProxyManager
    
    pm = ProxyManager()
    pm.add_proxy("http", "bad-proxy.com", 8080)
    pm.add_proxy("http", "good-proxy.com", 8080)
    
    proxy = pm.proxies[0]
    
    # Mark as failed 3 times
    pm.mark_failure(proxy)
    pm.mark_failure(proxy)
    pm.mark_failure(proxy)
    
    # Proxy should be banned
    assert proxy.url in pm.banned_proxies
    
    # Next proxy should be different
    next_proxy = pm.get_next_proxy()
    assert next_proxy.host == "good-proxy.com"

def test_proxy_stats():
    """Test proxy statistics (FREE)."""
    from core.stealth.evasion import ProxyManager
    
    pm = ProxyManager()
    pm.add_proxy("http", "proxy1.com", 8080)
    pm.add_proxy("http", "proxy2.com", 8080)
    
    stats = pm.get_stats()
    
    assert stats['total_proxies'] == 2
    assert stats['available_proxies'] == 2
    assert stats['banned_proxies'] == 0
    assert stats['total_requests'] == 0

@pytest.mark.asyncio
async def test_cli_proxy_integration():
    """Test CLI proxy flags integration."""
    # This would require running actual CLI, so just test imports
    from core.stealth.evasion import ProxyManager
    from main import cli
    
    assert ProxyManager is not None
    assert cli is not None
    
    # TODO: Add actual CLI integration test when running full scan


if __name__ == "__main__":
    pytest.main([__file__, "-v"])