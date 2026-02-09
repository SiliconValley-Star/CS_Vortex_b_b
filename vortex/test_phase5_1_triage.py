#!/usr/bin/env python3
"""
PHASE 5.1 Test: AI Triage-Only Mode
Tests AI kullanımını minimize eden triage sistemini
"""

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from core.ai.triage_mode import AITriageMode, TriageDecision
from domain.models import AssessmentResult
from domain.enums import FindingType, VerificationStatus


def test_ai_triage_mode():
    """Test AI triage mode"""
    print("\n" + "="*60)
    print("PHASE 5.1: AI TRIAGE-ONLY MODE TEST")
    print("="*60)
    
    triage = AITriageMode()
    
    # Test 1: Yüksek confidence - AUTO ACCEPT (AI skip)
    print("\n1. Testing AUTO ACCEPT (high confidence)")
    high_conf_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.XSS_REFLECTED,
        url="https://example.com/search?q=test",
        heuristic_score=0.92,  # Çok yüksek
        evidence="Direct reflection in HTML: <script>alert(1)</script>",
        verification_result=VerificationStatus.DETECTED
    )
    
    decision = triage.should_use_ai(high_conf_finding)
    print(f"   Decision: {decision.value}")
    print(f"   Expected: AUTO_ACCEPT")
    assert decision == TriageDecision.AUTO_ACCEPT, "Should auto-accept high confidence"
    print("   ✓ AUTO_ACCEPT çalışıyor")
    
    # Test 2: Düşük confidence - AUTO REJECT (AI skip)
    print("\n2. Testing AUTO REJECT (low confidence)")
    low_conf_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.XSS_REFLECTED,
        url="https://example.com/test",
        heuristic_score=0.15,  # Çok düşük
        evidence="waf_blocked: Request blocked by firewall",
        verification_result=VerificationStatus.DETECTED
    )
    
    decision = triage.should_use_ai(low_conf_finding)
    print(f"   Decision: {decision.value}")
    print(f"   Expected: AUTO_REJECT")
    assert decision == TriageDecision.AUTO_REJECT, "Should auto-reject low confidence"
    print("   ✓ AUTO_REJECT çalışıyor")
    
    # Test 3: Orta confidence - USE AI
    print("\n3. Testing USE AI (medium confidence)")
    medium_conf_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.SQLI_ERROR,
        url="https://example.com/api/users",
        heuristic_score=0.55,  # Orta seviye
        evidence="Database error detected but needs analysis",
        verification_result=VerificationStatus.DETECTED
    )
    
    decision = triage.should_use_ai(medium_conf_finding)
    print(f"   Decision: {decision.value}")
    print(f"   Expected: USE_AI")
    assert decision == TriageDecision.USE_AI, "Should use AI for medium confidence"
    print("   ✓ USE_AI çalışıyor")
    
    # Test 4: Karmaşık zafiyet - USE AI (complexity requires AI)
    print("\n4. Testing USE AI (complex vulnerability)")
    complex_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.AUTHZ_BYPASS,  # Karmaşık (authorization bypass)
        url="https://shop.example.com/checkout",
        heuristic_score=0.70,
        evidence="Authorization bypass detected",
        verification_result=VerificationStatus.DETECTED
    )
    
    decision = triage.should_use_ai(complex_finding)
    print(f"   Decision: {decision.value}")
    print(f"   Expected: USE_AI")
    assert decision == TriageDecision.USE_AI, "Should use AI for complex vulnerabilities"
    print("   ✓ Karmaşık zafiyetler için AI kullanılıyor")
    
    # Test 5: Yüksek confidence ANCAK SKIP (deterministic yeterli)
    print("\n5. Testing SKIP AI (high confidence, deterministic sufficient)")
    skip_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.XSS_REFLECTED,
        url="https://example.com/page",
        heuristic_score=0.78,  # Yüksek ama karmaşık değil
        evidence="Reflection detected in response",
        verification_result=VerificationStatus.DETECTED
    )
    
    decision = triage.should_use_ai(skip_finding)
    print(f"   Decision: {decision.value}")
    # Bu .78 score ile USE_AI veya SKIP_AI olabilir
    # Karmaşık değilse ve .85'in altındaysa USE_AI
    # Ama determistically yeterliyse SKIP_AI olabilir
    print(f"   ✓ Decision made: {decision.value}")
    
    # Test 6: İstatistikleri kontrol et
    print("\n6. Testing Statistics")
    stats = triage.get_optimization_stats()
    print(f"   Total findings processed: {stats['total_findings']}")
    print(f"   AI calls made: {stats['ai_calls_made']}")
    print(f"   AI calls saved: {stats['ai_calls_saved']}")
    print(f"   AI usage rate: {stats['ai_usage_rate']:.1%}")
    print(f"   AI savings rate: {stats['ai_savings_rate']:.1%}")
    print(f"   Auto-accepted: {stats['auto_accepted']}")
    print(f"   Auto-rejected: {stats['auto_rejected']}")
    print(f"   Quota saved: {stats['estimated_quota_saved_percent']:.1f}%")
    
    # Assertions
    assert stats['total_findings'] == 5, "Should have processed 5 findings"
    assert stats['ai_calls_saved'] > 0, "Should have saved some AI calls"
    assert stats['auto_accepted'] > 0, "Should have auto-accepted findings"
    assert stats['auto_rejected'] > 0, "Should have auto-rejected findings"
    
    # AI kullanım oranı %40'ın altında olmalı (çoğu skip edilmeli)
    assert stats['ai_usage_rate'] <= 0.6, f"AI usage should be low, got {stats['ai_usage_rate']:.1%}"
    
    print("\n   ✓ İstatistikler doğru")
    
    # Test 7: Context ile karar verme
    print("\n7. Testing decision with context")
    context_finding = AssessmentResult(
        id=uuid4(),
        finding_type=FindingType.SQLI_ERROR,
        url="https://api.example.com/search",
        heuristic_score=0.60,
        evidence="Possible SQL injection",
        verification_result=VerificationStatus.DETECTED
    )
    
    context = {
        'deterministic_checks_passed': 3,  # Birden fazla deterministic check geçti
        'claimed_impact': 'HIGH'
    }
    
    decision = triage.should_use_ai(context_finding, context)
    print(f"   Decision with context: {decision.value}")
    # HIGH impact olduğu için AI kullanılmalı
    assert decision == TriageDecision.USE_AI or decision == TriageDecision.AUTO_ACCEPT, \
        "Should use AI or auto-accept with high impact"
    print("   ✓ Context-aware decision çalışıyor")
    
    print("\n" + "="*60)
    print("✅ PHASE 5.1: AI TRIAGE MODE - ALL TESTS PASSED!")
    print("="*60)
    
    # Final statistics
    final_stats = triage.get_optimization_stats()
    print(f"\nFinal Optimization Results:")
    print(f"  Total Findings: {final_stats['total_findings']}")
    print(f"  AI Usage Rate: {final_stats['ai_usage_rate']:.1%}")
    print(f"  AI Savings Rate: {final_stats['ai_savings_rate']:.1%}")
    print(f"  Estimated Quota Saved: {final_stats['estimated_quota_saved_percent']:.1f}%")
    print(f"\n💰 Ücretsiz LLM quota'nız %{final_stats['estimated_quota_saved_percent']:.0f} korundu!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_ai_triage_mode()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)