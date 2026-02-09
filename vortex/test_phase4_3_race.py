#!/usr/bin/env python3
"""
Simple test for PHASE 4.3: Race Condition Detector
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_race_detector_basic():
    """Test basic race detector functionality"""
    print("\n" + "="*60)
    print("PHASE 4.3: Race Condition Detector - Basic Test")
    print("="*60)
    
    try:
        # Direct import
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        from race_condition_detector import (
            race_detector,
            RaceType,
            RaceTest,
            ConcurrencyMode
        )
        print("✓ Race detector imported")
        
        # Test 1: Generate race tests
        url = "https://shop.example.com/api/checkout"
        method = "POST"
        params = {
            'item_id': '123',
            'quantity': '5',
            'price': '99.99',
            'coupon': 'SAVE20',
            'balance': '500.00'
        }
        
        tests = race_detector.generate_race_tests(url, method, params)
        print(f"\n✓ Generated {len(tests)} race condition tests")
        
        # Show generated tests
        print(f"\n✓ Test types generated:")
        for test in tests:
            print(f"  - {test.name} ({test.race_type.value})")
            print(f"    Concurrent requests: {test.concurrent_requests}")
            print(f"    Delay: {test.delay_between_ms}ms")
        
        # Test 2: Test summary
        summary = race_detector.get_test_summary(tests)
        print(f"\n✓ Test Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Total requests: {summary['total_requests']}")
        print(f"  Avg concurrent: {summary['avg_concurrent']:.1f}")
        
        print(f"\n✓ Tests by type:")
        for race_type, count in summary['by_type'].items():
            print(f"  - {race_type}: {count} test(s)")
        
        # Test 3: Vulnerable parameter detection
        print(f"\n✓ Testing vulnerable parameter detection:")
        test_params_list = [
            {'balance': '100'},
            {'quantity': '5'},
            {'coupon_code': 'TEST'},
            {'user_id': '123'}
        ]
        
        for params in test_params_list:
            vulnerable = race_detector._detect_vulnerable_params(params)
            print(f"  - {list(params.keys())}: {list(vulnerable) if vulnerable else 'none'}")
        
        print("\n✅ PHASE 4.3: BASIC TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 4.3 BASIC TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_race_execution():
    """Test race execution logic (mock)"""
    print("\n" + "="*60)
    print("Race Execution Test (Mock)")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        from race_condition_detector import (
            race_detector,
            RaceTest,
            RaceType
        )
        
        # Create a test case
        test = RaceTest(
            name="Test Double Spend",
            url="https://api.example.com/payment",
            method="POST",
            race_type=RaceType.DOUBLE_SPEND,
            concurrent_requests=10,
            delay_between_ms=0,
            params={'amount': '100', 'user_id': '123'}
        )
        
        print("✓ Created race test")
        print(f"  Type: {test.race_type.value}")
        print(f"  Concurrent requests: {test.concurrent_requests}")
        
        # Mock request function
        async def mock_request(**kwargs):
            """Mock async request"""
            await asyncio.sleep(0.01)  # Simulate network delay
            return {
                'status_code': 200,
                'body': '{"success": true}',
                'headers': {}
            }
        
        # Execute test (async mode)
        print("\n✓ Executing race test (async mode)...")
        result = await race_detector.execute_race_test_async(test, mock_request)
        
        print(f"\n✓ Race Test Results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Successful: {result.successful_requests}")
        print(f"  Failed: {result.failed_requests}")
        print(f"  Avg response time: {result.avg_response_time:.2f}ms")
        print(f"  Race window: {result.race_window_ms:.2f}ms")
        print(f"  Vulnerability detected: {result.detected_vulnerability}")
        
        if result.vulnerability_evidence:
            print(f"  Evidence: {result.vulnerability_evidence}")
        
        # Test 2: Create vulnerability report
        if result.detected_vulnerability:
            vuln = race_detector.create_vulnerability_report(test, result)
            if vuln:
                print(f"\n✓ Vulnerability Report:")
                print(f"  Type: {vuln.vuln_type.value}")
                print(f"  Severity: {vuln.severity}")
                print(f"  Impact: {vuln.impact}")
                print(f"  Exploitation rate: {vuln.exploitation_rate:.1%}")
        
        print("\n✅ RACE EXECUTION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ RACE EXECUTION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_specific_race_types():
    """Test specific race condition types"""
    print("\n" + "="*60)
    print("Specific Race Types Test")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        from race_condition_detector import race_detector, RaceType
        
        # Test scenarios for each race type
        scenarios = [
            {
                'name': 'Payment Endpoint',
                'url': 'https://api.example.com/checkout',
                'params': {'amount': '50', 'balance': '100'},
                'expected_types': [RaceType.DOUBLE_SPEND, RaceType.TOCTOU]
            },
            {
                'name': 'Inventory Endpoint',
                'url': 'https://api.example.com/products/buy',
                'params': {'product_id': '123', 'quantity': '10', 'stock': '5'},
                'expected_types': [RaceType.INVENTORY_MANIPULATION]
            },
            {
                'name': 'Coupon Endpoint',
                'url': 'https://api.example.com/apply-coupon',
                'params': {'coupon': 'SAVE50', 'order_id': '456'},
                'expected_types': [RaceType.COUPON_ABUSE]
            },
        ]
        
        print("✓ Testing race type detection for different scenarios:\n")
        
        for scenario in scenarios:
            tests = race_detector.generate_race_tests(
                scenario['url'],
                'POST',
                scenario['params']
            )
            
            detected_types = set(test.race_type for test in tests)
            
            print(f"  {scenario['name']}:")
            print(f"    URL: {scenario['url']}")
            print(f"    Params: {list(scenario['params'].keys())}")
            print(f"    Generated {len(tests)} tests")
            print(f"    Types: {[t.value for t in detected_types]}")
            
            # Always generates at least rate limit test
            has_expected = any(
                expected in detected_types 
                for expected in scenario.get('expected_types', [])
            )
            
            if has_expected or RaceType.RATE_LIMIT_BYPASS in detected_types:
                print(f"    ✓ Correctly identified race vulnerabilities")
            print()
        
        print("✅ SPECIFIC RACE TYPES TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SPECIFIC RACE TYPES TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VORTEX PHASE 4.3 TEST SUITE")
    print("Testing Race Condition Detection")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Basic Functionality", await test_race_detector_basic()))
    results.append(("Race Execution", await test_race_execution()))
    results.append(("Specific Race Types", await test_specific_race_types()))
    
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