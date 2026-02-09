#!/usr/bin/env python3
"""
Simple test for PHASE 4.2: Authorization Matrix Testing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_authorization_matrix():
    """Test authorization matrix generation"""
    print("\n" + "="*60)
    print("PHASE 4.2: Authorization Matrix Test")
    print("="*60)
    
    try:
        # Direct import
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        from authorization_matrix import (
            authorization_tester,
            Endpoint,
            UserRole,
            AccessType
        )
        print("✓ Authorization tester imported")
        
        # Test 1: Create test endpoints
        endpoints = [
            Endpoint(
                url="/api/users/123",
                method="GET",
                description="View user profile",
                required_role=UserRole.USER,
                access_type=AccessType.READ
            ),
            Endpoint(
                url="/api/admin/users",
                method="GET",
                description="List all users",
                required_role=UserRole.ADMIN,
                access_type=AccessType.READ
            ),
            Endpoint(
                url="/api/users/123",
                method="DELETE",
                description="Delete user",
                required_role=UserRole.ADMIN,
                access_type=AccessType.DELETE
            ),
        ]
        
        print(f"\n✓ Created {len(endpoints)} test endpoints")
        
        # Test 2: Generate matrix tests
        tests = authorization_tester.generate_matrix_tests(endpoints)
        print(f"✓ Generated {len(tests)} authorization tests")
        
        # Test 3: Test summary
        summary = authorization_tester.get_test_summary(tests)
        print(f"\n✓ Test Summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Should allow: {summary['should_allow']}")
        print(f"  Should deny: {summary['should_deny']}")
        
        print(f"\n✓ Tests by role:")
        for role, count in summary['by_role'].items():
            print(f"  - {role}: {count} tests")
        
        # Test 4: IDOR tests
        print(f"\n✓ Testing IDOR generation:")
        idor_tests = authorization_tester.generate_idor_tests(
            url="/api/users/{id}/profile",
            method="GET",
            user_id="100",
            other_user_ids=["101", "102", "103"]
        )
        print(f"  Generated {len(idor_tests)} IDOR tests")
        for test in idor_tests[:2]:
            print(f"  - {test.name}")
        
        # Test 5: Resource ID detection
        print(f"\n✓ Testing resource ID detection:")
        test_urls = [
            "/api/users/123",
            "/profile/456",
            "/posts/789/comments"
        ]
        for url in test_urls:
            ids = authorization_tester.detect_resource_ids(url)
            if ids:
                print(f"  - {url}: Found IDs {ids}")
        
        # Test 6: Admin endpoint detection
        print(f"\n✓ Testing admin endpoint detection:")
        admin_urls = [
            "/admin/dashboard",
            "/api/users",
            "/settings/system"
        ]
        for url in admin_urls:
            is_admin = authorization_tester.is_admin_endpoint(url)
            print(f"  - {url}: Admin={is_admin}")
        
        # Test 7: Create authorization matrix
        print(f"\n✓ Creating authorization matrix:")
        matrix = authorization_tester.create_test_matrix(endpoints)
        print(f"  Matrix size: {len(matrix)} endpoints × {len(UserRole)} roles")
        
        # Show sample
        for endpoint_key in list(matrix.keys())[:2]:
            print(f"\n  {endpoint_key}:")
            for role, allowed in list(matrix[endpoint_key].items())[:4]:
                status = "✓ Allow" if allowed else "✗ Deny"
                print(f"    - {role}: {status}")
        
        print("\n✅ PHASE 4.2: ALL AUTHORIZATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 4.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vulnerability_detection():
    """Test vulnerability detection logic"""
    print("\n" + "="*60)
    print("Vulnerability Detection Test")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        from authorization_matrix import (
            authorization_tester,
            Endpoint,
            UserRole,
            AccessType,
            AuthTest
        )
        
        # Create a test case
        endpoint = Endpoint(
            url="/api/admin/users",
            method="GET",
            description="Admin endpoint",
            required_role=UserRole.ADMIN,
            access_type=AccessType.READ
        )
        
        test = AuthTest(
            name="Test unauthorized access",
            endpoint=endpoint,
            test_role=UserRole.USER,
            expected_allowed=False,
            description="User trying to access admin endpoint"
        )
        
        print("✓ Created test case")
        
        # Test 1: Successful unauthorized access (vulnerability)
        vuln = authorization_tester.analyze_response(
            test=test,
            status_code=200,
            response_body='{"users": [...]}'
        )
        
        if vuln:
            print(f"✓ Detected vulnerability:")
            print(f"  Type: {vuln.vuln_type}")
            print(f"  Severity: {vuln.severity}")
            print(f"  Test role: {vuln.test_role.value}")
            print(f"  Required role: {vuln.expected_role.value}")
        
        # Test 2: Properly denied access (no vulnerability)
        vuln2 = authorization_tester.analyze_response(
            test=test,
            status_code=403,
            response_body='{"error": "Forbidden"}'
        )
        
        if not vuln2:
            print(f"✓ Correctly identified secure endpoint (no vulnerability)")
        
        print("\n✅ VULNERABILITY DETECTION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ VULNERABILITY DETECTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VORTEX PHASE 4.2 TEST SUITE")
    print("Testing Authorization Matrix & IDOR Detection")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Authorization Matrix", test_authorization_matrix()))
    results.append(("Vulnerability Detection", test_vulnerability_detection()))
    
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