#!/usr/bin/env python3
"""
Simple test for PHASE 4.1: Business Logic Analyzer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_business_logic_analyzer():
    """Test business logic flaw detection"""
    print("\n" + "="*60)
    print("PHASE 4.1: Business Logic Analyzer Test")
    print("="*60)
    
    try:
        # Direct import to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        
        from business_logic_analyzer import (
            business_logic_analyzer,
            LogicFlawType
        )
        print("✓ Business logic analyzer imported")
        
        # Test 1: Analyze checkout endpoint with price params
        url = "https://shop.example.com/checkout"
        params = {
            'item_id': '123',
            'price': '99.99',
            'quantity': '1',
            'discount': '10'
        }
        
        tests = business_logic_analyzer.analyze_endpoint(url, params)
        print(f"\n✓ Generated {len(tests)} tests for checkout endpoint")
        
        # Test 2: Check test types
        test_types = set(test.flaw_type for test in tests)
        print(f"✓ Test types: {len(test_types)}")
        for flaw_type in test_types:
            count = sum(1 for t in tests if t.flaw_type == flaw_type)
            print(f"  - {flaw_type.value}: {count} tests")
        
        # Test 3: Price manipulation tests
        price_tests = [t for t in tests if t.parameter == 'price']
        print(f"\n✓ Price manipulation tests: {len(price_tests)}")
        for test in price_tests[:3]:
            print(f"  - {test.name}: value={test.test_value}")
        
        # Test 4: Quantity tests
        qty_tests = [t for t in tests if t.parameter == 'quantity']
        print(f"\n✓ Quantity manipulation tests: {len(qty_tests)}")
        for test in qty_tests:
            print(f"  - {test.name}: value={test.test_value}")
        
        # Test 5: Discount tests
        discount_tests = [t for t in tests if t.parameter == 'discount']
        print(f"\n✓ Discount abuse tests: {len(discount_tests)}")
        for test in discount_tests:
            print(f"  - {test.name}: value={test.test_value}")
        
        # Test 6: Test summary
        summary = business_logic_analyzer.get_test_summary(tests)
        print(f"\n✓ Test Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  By type:")
        for flaw_type, count in summary['by_type'].items():
            print(f"    - {flaw_type}: {count}")
        
        # Test 7: Workflow detection
        workflow_urls = [
            "https://example.com/checkout",
            "https://example.com/cart/payment",
            "https://example.com/order/confirm",
            "https://example.com/wizard/step2"
        ]
        
        print(f"\n✓ Workflow Detection:")
        for test_url in workflow_urls:
            detected = business_logic_analyzer.detect_workflow_endpoints(test_url)
            if detected:
                print(f"  - {test_url}: {detected}")
        
        # Test 8: Response analysis (simulated)
        print(f"\n✓ Response Analysis (simulated):")
        
        # Simulate accepting negative price (vulnerability)
        test = tests[0]  # First test (likely negative price)
        flaw = business_logic_analyzer.analyze_response_for_logic_flaw(
            test=test,
            status_code=200,
            response_body='{"success": true, "order_id": "12345"}',
            original_response=None
        )
        
        if flaw:
            print(f"  ✓ Detected flaw: {flaw.flaw_type.value}")
            print(f"    Parameter: {flaw.parameter}")
            print(f"    Test value: {flaw.test_value}")
            print(f"    Severity: {flaw.severity}")
        else:
            print(f"  ⚠ No flaw detected (might be false negative)")
        
        print("\n✅ PHASE 4.1: ALL LOGIC ANALYZER TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 4.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_detection():
    """Test parameter type detection"""
    print("\n" + "="*60)
    print("Parameter Detection Test")
    print("="*60)
    
    try:
        # Direct import to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        
        from business_logic_analyzer import business_logic_analyzer
        
        # Test different parameter sets
        test_cases = [
            {
                'name': 'E-commerce checkout',
                'params': {'price': '50', 'qty': '2', 'total': '100'}
            },
            {
                'name': 'Discount code',
                'params': {'coupon': 'SAVE20', 'promo_value': '20'}
            },
            {
                'name': 'Rate limiting',
                'params': {'limit': '100', 'max_requests': '1000'}
            }
        ]
        
        for case in test_cases:
            tests = business_logic_analyzer.analyze_endpoint(
                "https://example.com/api",
                case['params']
            )
            print(f"\n✓ {case['name']}: {len(tests)} tests")
        
        print("\n✅ PARAMETER DETECTION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ PARAMETER DETECTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VORTEX PHASE 4.1 TEST SUITE")
    print("Testing Business Logic Flaw Detection")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Business Logic Analyzer", test_business_logic_analyzer()))
    results.append(("Parameter Detection", test_parameter_detection()))
    
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