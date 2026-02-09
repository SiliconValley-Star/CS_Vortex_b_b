#!/usr/bin/env python3
"""
Simple test for PHASE 3.4: TLS Fingerprint Customization
"""

import sys
import ssl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_tls_profiles():
    """Test TLS profile management"""
    print("\n" + "="*60)
    print("PHASE 3.4: TLS Profile Management Test")
    print("="*60)
    
    try:
        from core.stealth.tls_profiles import tls_manager, get_ssl_context_for_browser
        print("✓ TLS manager imported")
        
        # Test 1: Get available profiles
        profiles = tls_manager.get_profile_names()
        print(f"✓ Available profiles: {len(profiles)}")
        for profile in profiles:
            print(f"  - {profile}")
        
        # Test 2: Get profile info
        for profile_name in ['modern_browser', 'chrome_like', 'legacy_browser']:
            info = tls_manager.get_profile_info(profile_name)
            print(f"\n✓ Profile: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  TLS: {info['min_version']} to {info['max_version']}")
            print(f"  Custom ciphers: {info['has_custom_ciphers']}")
        
        # Test 3: Create SSL contexts
        print("\n✓ Creating SSL contexts:")
        
        for profile in ['modern_browser', 'chrome_like', 'firefox_like']:
            ctx = tls_manager.create_ssl_context(profile, verify_ssl=True)
            print(f"  - {profile}: {type(ctx).__name__}")
            assert isinstance(ctx, ssl.SSLContext)
        
        # Test 4: Browser-specific contexts
        print("\n✓ Browser-specific contexts:")
        for browser in ['chrome', 'firefox', 'curl']:
            ctx = get_ssl_context_for_browser(browser, verify_ssl=True)
            print(f"  - {browser}: {type(ctx).__name__}")
            assert isinstance(ctx, ssl.SSLContext)
        
        # Test 5: Profile switching
        print("\n✓ Testing profile switching:")
        tls_manager.set_profile('chrome_like')
        print(f"  Current profile: {tls_manager.current_profile}")
        assert tls_manager.current_profile == 'chrome_like'
        
        # Test 6: Limitations
        print("\n⚠ Limitations:")
        limitations = tls_manager.get_limitations()
        for i, limitation in enumerate(limitations, 1):
            print(f"  {i}. {limitation}")
        
        print("\n✅ PHASE 3.4: ALL TLS TESTS PASSED!")
        print("\nNOTE: This is NOT full JA3/JA4 spoofing!")
        print("For production use, consider curl-impersonate or tls-client")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 3.4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ssl_context_properties():
    """Test SSL context properties"""
    print("\n" + "="*60)
    print("SSL Context Properties Test")
    print("="*60)
    
    try:
        from core.stealth.tls_profiles import tls_manager
        
        # Create context
        ctx = tls_manager.create_ssl_context('modern_browser', verify_ssl=True)
        
        print("✓ SSL Context Properties:")
        print(f"  - Protocol: {ctx.protocol}")
        print(f"  - Verify mode: {ctx.verify_mode}")
        print(f"  - Check hostname: {ctx.check_hostname}")
        print(f"  - Min TLS version: {ctx.minimum_version}")
        print(f"  - Max TLS version: {ctx.maximum_version}")
        
        # Test with verify disabled
        ctx_no_verify = tls_manager.create_ssl_context('modern_browser', verify_ssl=False)
        print("\n✓ No-verify context:")
        print(f"  - Verify mode: {ctx_no_verify.verify_mode}")
        print(f"  - Check hostname: {ctx_no_verify.check_hostname}")
        
        print("\n✅ CONTEXT PROPERTIES TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ CONTEXT PROPERTIES TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VORTEX PHASE 3.4 TEST SUITE")
    print("Testing TLS Profile Management (LIMITED)")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("TLS Profiles", test_tls_profiles()))
    results.append(("SSL Context Properties", test_ssl_context_properties()))
    
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
        print("\n⚠️  IMPORTANT:")
        print("This is LIMITED TLS customization, NOT full JA3/JA4 spoofing!")
        print("Python's ssl module cannot control all TLS fingerprint aspects.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())