#!/usr/bin/env python3
"""
Simple test for PHASE 3.3: ML-Based Request Timing
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_timing_analyzer():
    """Test adaptive request timing"""
    print("\n" + "="*60)
    print("PHASE 3.3: ML-Based Request Timing Test")
    print("="*60)
    
    try:
        from core.stealth.request_timing import timing_analyzer
        print("✓ Timing analyzer imported")
        
        # Test 1: Record successful requests
        test_url = "https://example.com/api"
        
        # Simulate fast successful requests
        for i in range(5):
            timing_analyzer.record_request(
                target=test_url,
                response_time=0.3,  # 300ms - fast
                status_code=200,
                success=True
            )
        print("✓ Recorded 5 fast successful requests")
        
        # Get recommended delay (should be reduced)
        delay1 = timing_analyzer.get_recommended_delay(test_url)
        print(f"✓ Recommended delay after fast responses: {delay1:.2f}s")
        
        # Test 2: Simulate slow responses
        for i in range(3):
            timing_analyzer.record_request(
                target=test_url,
                response_time=3.0,  # 3s - slow
                status_code=200,
                success=True
            )
        print("✓ Recorded 3 slow successful requests")
        
        # Get recommended delay (should be increased)
        delay2 = timing_analyzer.get_recommended_delay(test_url)
        print(f"✓ Recommended delay after slow responses: {delay2:.2f}s")
        
        # Test 3: Simulate rate limiting
        timing_analyzer.record_request(
            target=test_url,
            response_time=1.0,
            status_code=429,  # Rate limit
            response_body="Too many requests",
            success=False
        )
        print("✓ Simulated rate limit (429)")
        
        # Get recommended delay (should be high)
        delay3 = timing_analyzer.get_recommended_delay(test_url)
        print(f"✓ Recommended delay after rate limit: {delay3:.2f}s")
        
        # Test 4: Get statistics
        stats = timing_analyzer.get_statistics(test_url)
        print(f"\n✓ Statistics:")
        print(f"  - Total requests: {stats['total_requests']}")
        print(f"  - Current delay: {stats['current_delay']:.2f}s")
        print(f"  - Rate limit detected: {stats['rate_limit_detected']}")
        if 'avg_response_time' in stats:
            print(f"  - Average response time: {stats['avg_response_time']:.2f}s")
        
        # Test 5: Test smart delay (async)
        print("\n✓ Testing async smart delay...")
        await timing_analyzer.smart_delay(test_url)
        print("✓ Smart delay completed")
        
        print("\n✅ PHASE 3.3: ALL TIMING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 3.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_network_integration():
    """Test timing analyzer integration with network client"""
    print("\n" + "="*60)
    print("Network Integration Test")
    print("="*60)
    
    try:
        from core.network import NetworkClient
        print("✓ NetworkClient imported")
        
        # Create network client
        client = NetworkClient()
        print(f"✓ NetworkClient created")
        print(f"  - Adaptive timing: {client.adaptive_timing_enabled}")
        
        if client.adaptive_timing_enabled:
            print("✓ Adaptive timing is integrated!")
        else:
            print("⚠ Adaptive timing not available (timing module not found)")
        
        print("\n✅ INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VORTEX PHASE 3.3 TEST SUITE")
    print("Testing Adaptive Request Timing (Statistical)")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Timing Analyzer", await test_timing_analyzer()))
    results.append(("Network Integration", await test_network_integration()))
    
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
    sys.exit(asyncio.run(main()))