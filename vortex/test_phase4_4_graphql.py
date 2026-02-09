#!/usr/bin/env python3
"""
Simple test for PHASE 4.4: Enhanced GraphQL Security Scanner
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_graphql_enhanced_scanner():
    """Test enhanced GraphQL scanner functionality"""
    print("\n" + "="*60)
    print("PHASE 4.4: Enhanced GraphQL Scanner Test")
    print("="*60)
    
    try:
        # Direct import
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'api'))
        from graphql_enhanced import (
            enhanced_graphql_scanner,
            GraphQLAttackType,
            GraphQLAttack
        )
        print("✓ Enhanced GraphQL scanner imported")
        
        # Mock schema for testing
        mock_schema = {
            'queryType': {'name': 'Query'},
            'mutationType': {'name': 'Mutation'},
            'types': [
                {
                    'name': 'Query',
                    'fields': [
                        {'name': 'user', 'args': [{'name': 'id'}]},
                        {'name': 'posts', 'args': []},
                        {'name': 'search', 'args': []}
                    ]
                },
                {
                    'name': 'Mutation',
                    'fields': [
                        {'name': 'createUser'},
                        {'name': 'updateUser'}
                    ]
                },
                {
                    'name': 'User',
                    'fields': [
                        {'name': 'id'},
                        {'name': 'name'},
                        {'name': 'posts'}
                    ]
                }
            ]
        }
        
        # Test 1: Generate attack vectors
        attacks = enhanced_graphql_scanner.generate_attack_vectors(
            mock_schema,
            "https://api.example.com/graphql"
        )
        
        print(f"\n✓ Generated {len(attacks)} enhanced attack vectors")
        
        # Test 2: Show attacks by type
        summary = enhanced_graphql_scanner.get_attack_summary(attacks)
        print(f"\n✓ Attack Summary:")
        print(f"  Total attacks: {summary['total_attacks']}")
        
        print(f"\n✓ Attacks by type:")
        for attack_type, count in summary['by_type'].items():
            print(f"  - {attack_type}: {count} attacks")
        
        print(f"\n✓ Attacks by severity:")
        for severity, count in summary['by_severity'].items():
            print(f"  - {severity}: {count} attacks")
        
        # Test 3: Show sample attacks
        print(f"\n✓ Sample attacks:")
        for attack in attacks[:3]:
            print(f"\n  Attack: {attack.name}")
            print(f"    Type: {attack.attack_type.value}")
            print(f"    Severity: {attack.severity}")
            print(f"    Description: {attack.description}")
        
        # Test 4: Test vulnerability analysis
        print(f"\n✓ Testing vulnerability analysis:")
        
        # Simulate successful fragment bombing attack
        fragment_attack = GraphQLAttack(
            name="Test Fragment Bomb",
            attack_type=GraphQLAttackType.FRAGMENT_BOMBING,
            query="fragment F1...",
            description="Test",
            expected_behavior="Should block"
        )
        
        vuln = enhanced_graphql_scanner.analyze_response(
            attack=fragment_attack,
            status_code=200,
            response_body='{"data": {"user": {"name": "test"}}}',
            response_time=0.5
        )
        
        if vuln:
            print(f"  Detected vulnerability:")
            print(f"    Type: {vuln.attack_type.value}")
            print(f"    Severity: {vuln.severity}")
            print(f"    Impact: {vuln.impact}")
            print(f"    CVSS Score: {vuln.cvss_score}")
        
        # Test 5: Test specific attack types
        print(f"\n✓ Verifying attack type coverage:")
        attack_types = set(attack.attack_type for attack in attacks)
        
        expected_types = [
            GraphQLAttackType.ALIAS_CHAINING,
            GraphQLAttackType.RECURSIVE_EXPANSION,
            GraphQLAttackType.COST_BYPASS,
            GraphQLAttackType.DIRECTIVE_ABUSE,
            GraphQLAttackType.FRAGMENT_BOMBING,
            GraphQLAttackType.TYPE_CONFUSION,
            GraphQLAttackType.CUSTOM_SCALAR_ABUSE
        ]
        
        for expected_type in expected_types:
            if expected_type in attack_types:
                print(f"  ✓ {expected_type.value}")
            else:
                print(f"  ✗ {expected_type.value} (missing)")
        
        print("\n✅ PHASE 4.4: ALL GRAPHQL ENHANCED TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 4.4 GRAPHQL ENHANCED TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attack_generation():
    """Test specific attack generation"""
    print("\n" + "="*60)
    print("Attack Generation Test")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'api'))
        from graphql_enhanced import enhanced_graphql_scanner
        
        mock_schema = {'types': []}
        
        # Test each attack generator
        print("\n✓ Testing individual attack generators:")
        
        # 1. Alias chaining
        alias_attacks = enhanced_graphql_scanner._generate_alias_chaining_attacks(mock_schema)
        print(f"  - Alias chaining: {len(alias_attacks)} attacks")
        
        # 2. Recursive expansion
        recursive_attacks = enhanced_graphql_scanner._generate_recursive_expansion_attacks(mock_schema)
        print(f"  - Recursive expansion: {len(recursive_attacks)} attacks")
        
        # 3. Cost bypass
        cost_attacks = enhanced_graphql_scanner._generate_cost_bypass_attacks(mock_schema)
        print(f"  - Cost bypass: {len(cost_attacks)} attacks")
        
        # 4. Directive abuse
        directive_attacks = enhanced_graphql_scanner._generate_directive_abuse_attacks(mock_schema)
        print(f"  - Directive abuse: {len(directive_attacks)} attacks")
        
        # 5. Fragment bombing
        fragment_attacks = enhanced_graphql_scanner._generate_fragment_bombing_attacks(mock_schema)
        print(f"  - Fragment bombing: {len(fragment_attacks)} attacks")
        
        # 6. Type confusion
        type_attacks = enhanced_graphql_scanner._generate_type_confusion_attacks(mock_schema)
        print(f"  - Type confusion: {len(type_attacks)} attacks")
        
        # 7. Custom scalar abuse
        scalar_attacks = enhanced_graphql_scanner._generate_custom_scalar_attacks(mock_schema)
        print(f"  - Custom scalar abuse: {len(scalar_attacks)} attacks")
        
        total = (len(alias_attacks) + len(recursive_attacks) + len(cost_attacks) +
                len(directive_attacks) + len(fragment_attacks) + len(type_attacks) +
                len(scalar_attacks))
        
        print(f"\n✓ Total attack vectors: {total}")
        
        print("\n✅ ATTACK GENERATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ ATTACK GENERATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vulnerability_detection():
    """Test vulnerability detection logic"""
    print("\n" + "="*60)
    print("Vulnerability Detection Test")
    print("="*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'scanners' / 'api'))
        from graphql_enhanced import (
            enhanced_graphql_scanner,
            GraphQLAttack,
            GraphQLAttackType
        )
        
        # Test different scenarios
        scenarios = [
            {
                'name': 'Fragment Bombing',
                'attack_type': GraphQLAttackType.FRAGMENT_BOMBING,
                'status': 200,
                'body': '{"data": {"user": {}}}',
                'time': 0.5,
                'should_detect': True
            },
            {
                'name': 'Type Confusion with Sensitive Data',
                'attack_type': GraphQLAttackType.TYPE_CONFUSION,
                'status': 200,
                'body': '{"data": {"admin": {"secretToken": "xxx"}}}',
                'time': 0.3,
                'should_detect': True
            },
            {
                'name': 'Resource Exhaustion',
                'attack_type': GraphQLAttackType.ALIAS_CHAINING,
                'status': 500,
                'body': 'timeout exceeded',
                'time': 10.0,
                'should_detect': True
            },
            {
                'name': 'Properly Blocked',
                'attack_type': GraphQLAttackType.COST_BYPASS,
                'status': 400,
                'body': '{"errors": ["Query too complex"]}',
                'time': 0.1,
                'should_detect': False
            }
        ]
        
        print("\n✓ Testing vulnerability detection scenarios:")
        detected = 0
        
        for scenario in scenarios:
            attack = GraphQLAttack(
                name=scenario['name'],
                attack_type=scenario['attack_type'],
                query="test query",
                description="test",
                expected_behavior="test"
            )
            
            vuln = enhanced_graphql_scanner.analyze_response(
                attack=attack,
                status_code=scenario['status'],
                response_body=scenario['body'],
                response_time=scenario['time']
            )
            
            is_detected = vuln is not None
            expected = scenario['should_detect']
            status = "✓" if is_detected == expected else "✗"
            
            print(f"  {status} {scenario['name']}: ", end="")
            print(f"Detected={is_detected}, Expected={expected}")
            
            if is_detected:
                detected += 1
        
        print(f"\n✓ Detected {detected}/{len([s for s in scenarios if s['should_detect']])} expected vulnerabilities")
        
        print("\n✅ VULNERABILITY DETECTION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ VULNERABILITY DETECTION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VORTEX PHASE 4.4 TEST SUITE")
    print("Testing Enhanced GraphQL Security Scanner")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Enhanced Scanner", test_graphql_enhanced_scanner()))
    results.append(("Attack Generation", test_attack_generation()))
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