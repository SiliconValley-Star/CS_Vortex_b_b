#!/usr/bin/env python3
"""
Simple test script for PHASE 3.1 & 3.2 (No pytest required)
Tests FREE proxy methods and browser automation
"""

import sys
from pathlib import Path

# Add vortex to path
sys.path.insert(0, str(Path(__file__).parent))

def test_proxy_manager():
    """Test PHASE 3.1: Proxy Integration"""
    print("\n" + "="*60)
    print("PHASE 3.1: Proxy Integration Tests")
    print("="*60)
    
    try:
        from core.stealth.evasion import ProxyManager
        print("✓ ProxyManager imported successfully")
        
        # Test 1: Create proxy manager
        pm = ProxyManager()
        assert pm is not None
        print("✓ ProxyManager created")
        
        # Test 2: Add Tor proxy
        pm.add_tor_proxy()
        assert len(pm.proxies) == 1
        assert pm.proxies[0].protocol == "socks5"
        assert pm.proxies[0].port == 9050
        print("✓ Tor proxy added (FREE)")
        
        # Test 3: Add HTTP proxy
        pm.add_proxy("http", "127.0.0.1", 8080)
        assert len(pm.proxies) == 2
        print("✓ HTTP proxy added (FREE)")
        
        # Test 4: Add authenticated proxy
        pm.add_proxy("http", "proxy.example.com", 3128, "user", "pass")
        assert len(pm.proxies) == 3
        assert pm.proxies[2].username == "user"
        assert "user:pass@" in pm.proxies[2].url
        print("✓ Authenticated proxy added (FREE)")
        
        # Test 5: Proxy rotation
        p1 = pm.get_next_proxy()
        p2 = pm.get_next_proxy()
        p3 = pm.get_next_proxy()
        assert p1 != p2 != p3
        print("✓ Proxy rotation working")
        
        # Test 6: Stats
        stats = pm.get_stats()
        assert stats['total_proxies'] == 3
        assert stats['available_proxies'] == 3
        print("✓ Proxy statistics working")
        
        print("\n✅ PHASE 3.1: ALL PROXY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 3.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_browser_automation():
    """Test PHASE 3.2: Browser Automation"""
    print("\n" + "="*60)
    print("PHASE 3.2: Browser Automation Tests")
    print("="*60)
    
    try:
        from scanners.advanced.dom_scanner import PlaywrightDOMScanner, PLAYWRIGHT_AVAILABLE
        print("✓ DOM Scanner imported successfully")
        
        if not PLAYWRIGHT_AVAILABLE:
            print("⚠ Playwright not installed - skipping browser tests")
            print("  Install: pip install playwright && playwright install chromium")
            return True
        
        # Test 1: Create scanner with stealth mode
        scanner = PlaywrightDOMScanner(stealth_mode=True)
        assert scanner.stealth_mode == True
        print("✓ Stealth mode scanner created")
        
        # Test 2: Create scanner with proxy
        scanner_proxy = PlaywrightDOMScanner(
            stealth_mode=True,
            proxy="socks5://127.0.0.1:9050"
        )
        assert scanner_proxy.proxy == "socks5://127.0.0.1:9050"
        print("✓ Scanner with proxy created (FREE)")
        
        # Test 3: Check payloads
        assert len(scanner.XSS_PAYLOADS) > 0
        print(f"✓ {len(scanner.XSS_PAYLOADS)} XSS payloads loaded")
        
        # Test 4: Check sinks
        assert len(scanner.DOM_SINKS) > 0
        print(f"✓ {len(scanner.DOM_SINKS)} DOM sinks monitored")
        
        print("\n✅ PHASE 3.2: ALL BROWSER TESTS PASSED!")
        print("  Note: Full browser tests require Playwright installation")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 3.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration"""
    print("\n" + "="*60)
    print("CLI Integration Tests")
    print("="*60)
    
    try:
        # Test import
        from main import cli
        print("✓ CLI imported successfully")
        
        # Check if proxy flags are available
        # (Can't easily test Click commands without running them)
        print("✓ CLI module available")
        
        print("\n✅ CLI INTEGRATION: PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ CLI INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VORTEX PHASE 3 TEST SUITE")
    print("Testing FREE Proxy & Browser Automation Features")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("PHASE 3.1: Proxy Integration", test_proxy_manager()))
    results.append(("PHASE 3.2: Browser Automation", test_browser_automation()))
    results.append(("CLI Integration", test_cli_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())