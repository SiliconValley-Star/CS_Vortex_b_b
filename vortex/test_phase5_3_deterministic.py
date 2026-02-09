#!/usr/bin/env python3
"""
PHASE 5.3 Test: Deterministic First Strategy
Tests deterministik analiz sistemini (AI'dan önce)
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from core.analysis.deterministic_analyzer import (
    DeterministicAnalyzer,
    DeterministicConfidence
)
from domain.models import AssessmentResult
from domain.enums import FindingType


def test_deterministic_analyzer():
    """Test deterministic analyzer"""
    print("\n" + "="*60)
    print("PHASE 5.3: DETERMINISTIC FIRST STRATEGY TEST")
    print("="*60)
    
    analyzer = DeterministicAnalyzer()
    
    # Test 1: SQL Injection - High Confidence (AI gerektirmez)
    print("\n1. Testing SQL Injection (High Confidence)")
    sql_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.SQLI_ERROR,
        url="https://example.com/search?q=test",
        heuristic_score=0.75,
        evidence="MySQL error: You have an error in your SQL syntax near 'test' at line 1",
        verification_result=None
    )
    
    result = analyzer.analyze(sql_finding)
    print(f"   Confidence Score: {result['confidence_score']:.2f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Needs AI: {result['needs_ai']}")
    print(f"   Matches: {result['match_count']}")
    
    assert result['confidence_score'] >= 0.5, "Should have medium-high confidence for SQL error"
    # Medium confidence bile AI skip edebilir (triage mode ile)
    print("   ✓ SQL Injection deterministik analiz başarılı")
    
    # Test 2: XSS - Medium Confidence (AI önerilir)
    print("\n2. Testing XSS (Medium Confidence)")
    xss_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.XSS_REFLECTED,
        url="https://example.com/search?q=<script>alert(1)</script>",
        heuristic_score=0.60,
        evidence="Reflected in response: <script>alert(1)</script>",
        verification_result=None
    )
    
    result = analyzer.analyze(xss_finding)
    print(f"   Confidence Score: {result['confidence_score']:.2f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Needs AI: {result['needs_ai']}")
    print(f"   Matches: {result['match_count']}")
    
    assert result['match_count'] > 0, "Should find XSS patterns"
    print("   ✓ XSS deterministik analiz başarılı")
    
    # Test 3: LFI - Very High Confidence (AI gerektirmez)
    print("\n3. Testing LFI (Very High Confidence)")
    lfi_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.LFI,
        url="https://example.com/file?path=../../etc/passwd",
        heuristic_score=0.85,
        evidence="root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin",
        verification_result=None
    )
    
    result = analyzer.analyze(lfi_finding)
    print(f"   Confidence Score: {result['confidence_score']:.2f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Needs AI: {result['needs_ai']}")
    print(f"   Matches: {result['match_count']}")
    
    assert result['confidence_score'] >= 0.6, "Should have high confidence for /etc/passwd"
    # /etc/passwd görünce confidence yüksek olmalı
    print("   ✓ LFI deterministik analiz başarılı")
    
    # Test 4: Command Injection - High Confidence
    print("\n4. Testing Command Injection (High Confidence)")
    cmd_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.COMMAND_INJECTION,
        url="https://example.com/ping?host=8.8.8.8",
        heuristic_score=0.70,
        evidence="uid=33(www-data) gid=33(www-data) groups=33(www-data)",
        verification_result=None
    )
    
    result = analyzer.analyze(cmd_finding)
    print(f"   Confidence Score: {result['confidence_score']:.2f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Needs AI: {result['needs_ai']}")
    print(f"   Matches: {result['match_count']}")
    
    assert result['confidence_score'] >= 0.5, "Should have medium-high confidence for uid/gid output"
    print("   ✓ Command Injection deterministik analiz başarılı")
    
    # Test 5: Low Confidence - AI gerekli
    print("\n5. Testing Low Confidence (AI Required)")
    low_conf_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.SQLI_BLIND,
        url="https://example.com/api/user?id=1",
        heuristic_score=0.35,
        evidence="Response looks slightly different",
        verification_result=None
    )
    
    result = analyzer.analyze(low_conf_finding)
    print(f"   Confidence Score: {result['confidence_score']:.2f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Needs AI: {result['needs_ai']}")
    print(f"   Matches: {result['match_count']}")
    
    assert result['needs_ai'], "Should need AI for low confidence"
    print("   ✓ Low confidence doğru tespit edildi")
    
    # Test 6: Response Data Analysis
    print("\n6. Testing Response Data Analysis")
    response_data = {
        'status_code': 500,
        'content_length': 15000,
        'response_time': 0.5
    }
    
    finding_with_response = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.SQLI_ERROR,
        url="https://example.com/api/search",
        heuristic_score=0.65,
        evidence="Database error detected",
        verification_result=None
    )
    
    result = analyzer.analyze(finding_with_response, response_data)
    print(f"   Confidence Score: {result['confidence_score']:.2f}")
    print(f"   Confidence Level: {result['confidence_level']}")
    print(f"   Needs AI: {result['needs_ai']}")
    print(f"   Matches: {result['match_count']}")
    
    assert result['match_count'] > 0, "Should find response-based matches"
    print("   ✓ Response data analizi başarılı")
    
    # Test 7: İstatistikleri kontrol et
    print("\n7. Testing Statistics")
    stats = analyzer.get_statistics()
    print(f"   Total analyzed: {stats['total_analyzed']}")
    print(f"   AI avoided: {stats['ai_avoided']}")
    print(f"   AI avoidance rate: {stats['ai_avoidance_rate']:.1%}")
    print(f"   High confidence rate: {stats['high_confidence_rate']:.1%}")
    
    assert stats['total_analyzed'] == 6, "Should have analyzed 6 findings"
    assert stats['ai_avoided'] >= 2, "Should have avoided AI for some findings"
    assert stats['ai_avoidance_rate'] > 0, "Should have non-zero AI avoidance"
    
    print(f"\n   ✓ İstatistikler doğru")
    print(f"   💰 AI quota %{stats['ai_avoidance_rate']*100:.0f} korundu!")
    
    print("\n" + "="*60)
    print("✅ PHASE 5.3: DETERMINISTIC FIRST STRATEGY - ALL TESTS PASSED!")
    print("="*60)
    
    # Final summary
    print(f"\nDeterministic Analysis Summary:")
    print(f"  Total Findings: {stats['total_analyzed']}")
    print(f"  High Confidence: {int(stats['high_confidence_rate'] * stats['total_analyzed'])}")
    print(f"  Medium Confidence: {int(stats['medium_confidence_rate'] * stats['total_analyzed'])}")
    print(f"  Low Confidence: {int(stats['low_confidence_rate'] * stats['total_analyzed'])}")
    print(f"  AI Avoided: {stats['ai_avoided']}")
    print(f"  AI Avoidance Rate: {stats['ai_avoidance_rate']:.1%}")
    print(f"\n🎯 Deterministik analiz ile AI kullanımı minimize edildi!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_deterministic_analyzer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)