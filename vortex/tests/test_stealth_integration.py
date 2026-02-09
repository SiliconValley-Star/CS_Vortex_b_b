
import asyncio
import logging
import sys
from unittest.mock import MagicMock, AsyncMock, patch
import types

# 1. Mock 'structlog' and 'aiohttp' globally
sys.modules['structlog'] = MagicMock()
mock_aiohttp = MagicMock()
class MockClientError(Exception): pass
mock_aiohttp.ClientError = MockClientError
sys.modules['aiohttp'] = mock_aiohttp
sys.modules['aiohttp_socks'] = MagicMock()

# 2. Mock all 'core' submodules that we don't need, to avoid core/__init__.py exploding
# We need to ensure that when core/__init__.py runs, it finds these modules
modules_to_mock = [
    'core.authority',
    'core.authority.hierarchy',
    'core.authority.validator',
    'core.authority.compliance',
    'core.evidence',
    'core.evidence.standards',
    'core.evidence.behavioral',
    'core.evidence.determinism',
    'core.ai',
    'core.ai.openrouter',
    'core.ai.advisory',
    'core.ai.fallbacks',
    'core.workflow',
    'core.workflow.state_machine',
    'core.workflow.orchestrator',
    'core.health',
    'core.health.monitor',
    'core.health.auto_tune',
    'core.database',
    'core.verification',
    'core.engine',
    'core.state',
    'core.stealth',
    'core.stealth.evasion'
]

for mod_name in modules_to_mock:
    sys.modules[mod_name] = MagicMock()

# 3. Now import the modules we strictly need TO TEST
# core.exceptions and core.network are real
from core.network import NetworkClient, HTTPResponse
# We mocked core.stealth.evasion, so we import the mocks from there to check against
from core.stealth.evasion import UserAgentRotator, ProxyManager, WAFDetector, ProxyConfig, WAFProfile

async def test_stealth_integration():
    """Verify NetworkClient uses Stealth components."""
    print("Testing Stealth Integration...")
    
    # Setup Mocks
    mock_ua_rotator = MagicMock()
    mock_ua_rotator.get_for_domain.return_value = "Mozilla/5.0 (Stealth Mode)"
    
    mock_proxy_manager = MagicMock()
    mock_proxy_config = MagicMock()
    mock_proxy_config.url = "http://1.2.3.4:8080"
    mock_proxy_manager.get_next_proxy.return_value = mock_proxy_config
    
    mock_waf_detector = MagicMock()
    mock_waf_detector.analyze_response.return_value = None  # No WAF detected
    
    # Initialize Client with mocks
    # Note: NetworkClient.__init__ attempts to instantiate these if STEALTH_AVAILABLE is True.
    # We need to make sure the mocked classes return our mock instances when instantiated.
    
    # Patch the classes in the mocked module
    with patch('core.stealth.evasion.UserAgentRotator', return_value=mock_ua_rotator), \
         patch('core.stealth.evasion.ProxyManager', return_value=mock_proxy_manager), \
         patch('core.stealth.evasion.WAFDetector', return_value=mock_waf_detector):
        
        client = NetworkClient()
        # Ensure stealth mode enabled (depends on STEALTH_AVAILABLE check in network.py)
        # Since we mocked core.stealth.evasion, the import in network.py should succeed.
        
        # Manually force inject just in case, though __init__ should have done it
        client.ua_rotator = mock_ua_rotator
        client.proxy_manager = mock_proxy_manager
        client.waf_detector = mock_waf_detector
        client.detect_waf = True
        client.rotate_ua = True
        client.use_proxies = True
        
        # Mock Session
        # Since we mocked aiohttp, client.session might need help if it uses aiohttp.ClientSession
        # But we mocked the module, so aiohttp.ClientSession is a Mock class.
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.url = "http://example.com"
        mock_response.headers = {}
        mock_response.cookies = {}
        mock_response.content.read.return_value = b"OK"
        mock_response.text.return_value = "OK"
        
        # Async Context Manager Mock for session.request
        mock_request_ctx = MagicMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        
        client.session = MagicMock()
        client.session.request = MagicMock(return_value=mock_request_ctx)
        
        # Execute Request
        print("Making request...")
        await client.request("GET", "http://example.com")
        
        # VERIFICATION
        
        # 1. Verify User-Agent was rotated
        call_args = client.session.request.call_args
        headers = call_args.kwargs['headers']
        print(f"Headers sent: {headers}")
        if headers.get('User-Agent') == "Mozilla/5.0 (Stealth Mode)":
            print("[PASS] User-Agent injected correctly.")
        else:
            print("[FAIL] User-Agent NOT injected. Got:", headers.get('User-Agent'))
            
        # 2. Verify Proxy was used
        proxy_arg = call_args.kwargs.get('proxy')
        print(f"Proxy used: {proxy_arg}")
        if proxy_arg == "http://1.2.3.4:8080":
            print("[PASS] Proxy injected correctly.")
        else:
            print("[FAIL] Proxy NOT injected.")
            
        # 3. Verify WAF Detector was called
        mock_waf_detector.analyze_response.assert_called_once()
        print("[PASS] WAF Detector called.")
        
        print("\nStealth Integration Test Complete.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_stealth_integration())
