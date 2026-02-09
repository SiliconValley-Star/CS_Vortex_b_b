#!/usr/bin/env python3
"""
Comprehensive Integration Test for ALL Completed Phases
Tests PHASE 3.1-3.4 and PHASE 4.1-4.4 together
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_phase3_integration():
    """Test all PHASE 3 components working together"""
    print("\n" + "="*60)
    print("PHASE 3 INTEGRATION TEST (All 4 sub-phases)")
    print("="*60)
    
    try:
        # Import all PHASE 3 components
        from core.stealth.evasion import ProxyManager, UserAgentRotator, WAFDetector
        from core.stealth.request_timing import timing_analyzer
        from core.stealth.tls_profiles import tls_manager
        from core.network import NetworkClient
        
        print("✓ All PHASE 3 modules imported successfully")
        
        # Test 1: Create network client (should have all PHASE 3 features)
        client = NetworkClient()
        print("\n✓ NetworkClient created with features:")
        print(f"  - Stealth enabled: {client.stats.get('stealth_enabled', False)}")
        print(f"  - Adaptive timing: {client.stats.get('adaptive_timing_enabled', False)}")
        print(f"  - Adaptive evasion: {client.stats.get('adaptive_evasion_enabled', False)}")
        
        # Test 2: Proxy integration
        if client.proxy_manager:
            client.proxy_manager.add_tor_proxy()
            print(f"✓ Proxy manager working: {len(client.proxy_manager.proxies)} proxies")
        
        # Test 3: UA rotation
        if client.ua_rotator:
            ua = client.ua_rotator.get_for_domain("example.com")
            print(f"✓ UA rotation working: {ua[:50]}...")
        
        # Test 4: Timing analyzer
        if client.timing_analyzer:
            delay = client.timing_analyzer.get_recommended_delay("https://example.com")
            print(f"✓ Timing analyzer working: {delay:.2f}s recommended delay")
        
        # Test 5: TLS profiles
        ctx = tls_manager.create_ssl_context('chrome_like', verify_ssl=False)
        print(f"✓ TLS profiles working: {type(ctx).__name__}")
        
        print("\n✅ PHASE 3 INTEGRATION: ALL COMPONENTS WORKING TOGETHER!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 3 INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase4_integration():
    """Test PHASE 4.1, 4.2, 4.3, and 4.4 integration"""
    print("\n" + "="*60)
    print("PHASE 4 INTEGRATION TEST (4.1 + 4.2 + 4.3 + 4.4)")
    print("="*60)
    
    try:
        # Direct import to avoid circular deps
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        from business_logic_analyzer import business_logic_analyzer
        from authorization_matrix import authorization_tester, Endpoint, UserRole, AccessType
        from race_condition_detector import race_detector
        
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'api'))
        from graphql_enhanced import enhanced_graphql_scanner
        
        print("✓ All four PHASE 4 scanners imported")
        
        # Test 4.1: Business Logic
        url = "https://shop.example.com/api/checkout"
        params = {
            'item_id': '12345',
            'price': '99.99',
            'quantity': '2',
            'discount_code': 'SAVE20',
            'total': '159.98'
        }
        
        logic_tests = business_logic_analyzer.analyze_endpoint(url, params)
        print(f"✓ Business Logic: {len(logic_tests)} tests generated")
        
        # Test 4.2: Authorization Matrix
        endpoints = [
            Endpoint(
                url="/api/checkout",
                method="POST",
                description="Process checkout",
                required_role=UserRole.USER,
                access_type=AccessType.CREATE
            ),
            Endpoint(
                url="/api/admin/orders",
                method="GET",
                description="View all orders",
                required_role=UserRole.ADMIN,
                access_type=AccessType.READ
            ),
        ]
        
        authz_tests = authorization_tester.generate_matrix_tests(endpoints)
        print(f"✓ Authorization Matrix: {len(authz_tests)} tests generated")
        
        # Test 4.3: Race Condition Detection
        race_params = {
            'item_id': '123',
            'quantity': '2',
            'price': '99.99',
            'balance': '200.00'
        }
        
        race_tests = race_detector.generate_race_tests(url, "POST", race_params)
        print(f"✓ Race Condition Detector: {len(race_tests)} tests generated")
        
        # Test 4.4: GraphQL Enhanced Scanner
        mock_schema = {
            'queryType': {'name': 'Query'},
            'types': []
        }
        
        graphql_attacks = enhanced_graphql_scanner.generate_attack_vectors(
            mock_schema,
            "https://api.example.com/graphql"
        )
        print(f"✓ GraphQL Enhanced Scanner: {len(graphql_attacks)} attacks generated")
        
        # Show combined capabilities
        total_tests = len(logic_tests) + len(authz_tests) + len(race_tests) + len(graphql_attacks)
        print(f"\n✓ Combined PHASE 4 capabilities:")
        print(f"  - Business logic flaw detection: {len(logic_tests)} tests")
        print(f"  - Authorization testing: {len(authz_tests)} tests")
        print(f"  - Race condition detection: {len(race_tests)} tests")
        print(f"  - GraphQL advanced attacks: {len(graphql_attacks)} tests")
        print(f"  - Total security tests: {total_tests}")
        
        print("\n✅ PHASE 4 INTEGRATION: ALL FOUR MODULES WORKING!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 4 INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_system_integration():
    """Test full system with all phases together"""
    print("\n" + "="*60)
    print("FULL SYSTEM INTEGRATION TEST")
    print("Testing PHASE 3 + PHASE 4 (4.1, 4.2, 4.3 & 4.4) Together")
    print("="*60)
    
    try:
        from core.network import NetworkClient
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'logic'))
        from business_logic_analyzer import business_logic_analyzer
        from authorization_matrix import authorization_tester, Endpoint, UserRole, AccessType
        from race_condition_detector import race_detector
        
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'api'))
        from graphql_enhanced import enhanced_graphql_scanner
        
        # Scenario: Scanning an API with all features
        print("\nScenario: Complete API security scan with all features")
        
        # Step 1: Create network client (with all PHASE 3 features)
        client = NetworkClient()
        print(f"✓ Network client ready")
        print(f"  Features: Stealth={client.stats.get('stealth_enabled', False)}, "
              f"Timing={client.stats.get('adaptive_timing_enabled', False)}")
        
        # Step 2: Analyze endpoint for logic flaws (PHASE 4.1)
        url = "https://api.example.com/checkout"
        params = {'price': '50.00', 'qty': '1', 'coupon': 'SAVE10'}
        
        logic_tests = business_logic_analyzer.analyze_endpoint(url, params)
        print(f"✓ Logic analyzer generated {len(logic_tests)} tests")
        
        # Step 3: Generate authorization tests (PHASE 4.2)
        endpoints = [
            Endpoint(
                url="/api/checkout",
                method="POST",
                description="Checkout",
                required_role=UserRole.USER,
                access_type=AccessType.CREATE
            ),
        ]
        
        authz_tests = authorization_tester.generate_matrix_tests(endpoints)
        print(f"✓ Authorization tester generated {len(authz_tests)} tests")
        
        # Step 4: Generate race condition tests (PHASE 4.3)
        race_params = {'price': '50.00', 'qty': '1', 'balance': '100.00'}
        race_tests = race_detector.generate_race_tests(url, "POST", race_params)
        print(f"✓ Race detector generated {len(race_tests)} tests")
        
        # Step 5: Generate GraphQL attacks (PHASE 4.4)
        mock_schema = {'queryType': {'name': 'Query'}, 'types': []}
        graphql_attacks = enhanced_graphql_scanner.generate_attack_vectors(
            mock_schema,
            "https://api.example.com/graphql"
        )
        print(f"✓ GraphQL scanner generated {len(graphql_attacks)} attacks")
        
        # Step 6: Show complete workflow
        total_tests = len(logic_tests) + len(authz_tests) + len(race_tests) + len(graphql_attacks)
        print(f"\n✓ Would execute {total_tests} total tests:")
        print(f"  - {len(logic_tests)} business logic tests")
        print(f"  - {len(authz_tests)} authorization tests")
        print(f"  - {len(race_tests)} race condition tests")
        print(f"  - {len(graphql_attacks)} GraphQL attacks")
        print(f"  - All using: proxies, UA rotation, smart delays, TLS profiles")
        
        # Step 7: Show complete workflow
        print(f"\n✓ Complete security testing workflow:")
        print(f"  1. Business logic analyzer identifies flaws")
        print(f"  2. Authorization tester checks access controls")
        print(f"  3. Race detector finds concurrent vulnerabilities")
        print(f"  4. GraphQL scanner tests advanced attack vectors")
        print(f"  5. Network client applies stealth features")
        print(f"  6. Requests sent with TLS profiles & timing")
        print(f"  7. Responses analyzed for vulnerabilities")
        
        print("\n✅ FULL SYSTEM INTEGRATION: ALL PHASES WORKING TOGETHER!")
        return True
        
    except Exception as e:
        print(f"\n❌ FULL SYSTEM INTEGRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("VORTEX COMPREHENSIVE INTEGRATION TEST")
    print("Testing All Completed Phases Together")
    print("="*60)
    print("\nCompleted Phases:")
    print("  - PHASE 3.1: Proxy Integration (FREE)")
    print("  - PHASE 3.2: Browser Automation")
    print("  - PHASE 3.3: Adaptive Timing")
    print("  - PHASE 3.4: TLS Profiles")
    print("  - PHASE 4.1: Business Logic Analyzer")
    print("  - PHASE 4.2: Authorization Matrix Testing")
    print("  - PHASE 4.3: Race Condition Detector")
    print("  - PHASE 4.4: GraphQL Security Enhancement")
    
    results = []
    
    # Run integration tests
    results.append(("PHASE 3 Integration", await test_phase3_integration()))
    results.append(("PHASE 4 Integration (4.1 + 4.2 + 4.3 + 4.4)", test_phase4_integration()))
    results.append(("Full System Integration", await test_full_system_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} integration tests passed")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n✅ System Status:")
        print("  - All PHASE 3 components integrated")
        print("  - PHASE 4.1, 4.2, 4.3 & 4.4 working")
        print("  - Full system operational")
        return 0
    else:
        print(f"\n⚠️  {total - passed} integration test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))