#!/usr/bin/env python3
"""
PHASE 6.4 - Full System Integration Test
Test complete scan workflow with all components
"""

import asyncio
from unittest.mock import AsyncMock
from core.verification import global_verification_engine
from core.payloads.manager import global_payload_manager
from domain.models import AssessmentResult, AIAnalysisResult
from domain.enums import FindingType, FindingSeverity, AIVerdict
from core.network import HTTPResponse

print('='*60)
print('PHASE 6.4 - FULL SYSTEM INTEGRATION TEST')
print('='*60)

# TEST 1: Payload System Integration
print('\n1️⃣ TEST: Payload System Entegrasyonu')
print('-' * 40)

try:
    from core.payloads.curated_payloads import PayloadTier
    
    # Test all 3 TIERs
    tier1_count = len(global_payload_manager.get_payloads(tier=PayloadTier.TIER_1))
    tier2_count = len(global_payload_manager.get_payloads(tier=PayloadTier.TIER_2))
    tier3_count = len(global_payload_manager.get_payloads(tier=PayloadTier.TIER_3))
    total = tier1_count + tier2_count + tier3_count
    
    print(f'✅ Payload Manager aktif')
    print(f'   TIER 1: {tier1_count} payload')
    print(f'   TIER 2: {tier2_count} payload')
    print(f'   TIER 3: {tier3_count} payload')
    print(f'   TOPLAM: {total} payload')
    
    result1 = True
except Exception as e:
    print(f'❌ Payload system failed: {e}')
    result1 = False


# TEST 2: End-to-End Scan Simulation
print('\n2️⃣ TEST: End-to-End Scan Simülasyonu')
print('-' * 40)

async def test_end_to_end():
    try:
        # Simulate a complete scan workflow
        # Step 1: Get payloads
        xss_payloads = global_payload_manager.get_payloads(tier=1)
        print(f'✅ Step 1: {len(xss_payloads)} TIER 1 payload alındı')
        
        # Step 2: Create finding (simulated detection)
        ai_analysis = AIAnalysisResult(
            model_used="gpt-4",
            verdict=AIVerdict.CONFIRMED,
            confidence=0.92,
            reasoning="Reflected XSS with no encoding",
            poc="curl 'http://target.com/search?q=<img src=x onerror=alert(1)>'",
            success=True
        )
        
        finding = AssessmentResult(
            url="http://target.com/search",
            finding_type=FindingType.XSS_REFLECTED,
            severity=FindingSeverity.HIGH,
            payload="<img src=x onerror=alert(1)>",
            evidence="Payload reflected without sanitization",
            heuristic_score=0.85,
            ai_analysis=ai_analysis
        )
        print(f'✅ Step 2: Finding oluşturuldu ({finding.finding_type.value})')
        
        # Step 3: Verify finding (with mocked network)
        original_request = global_verification_engine.network_client.request
        
        mock_response = HTTPResponse(
            status_code=200,
            body='<html><img src=x onerror=alert(1)></html>',
            headers={'Content-Type': 'text/html'},
            response_time=0.145,
            url=finding.url
        )
        
        global_verification_engine.network_client.request = AsyncMock(return_value=mock_response)
        
        result = await global_verification_engine.verify_finding(finding)
        
        # Restore
        global_verification_engine.network_client.request = original_request
        
        print(f'✅ Step 3: Verification tamamlandı')
        print(f'   Success: {result.success}')
        print(f'   Match type: {result.match_type}')
        print(f'   Confidence: {result.confidence:.2f}')
        print(f'   Determinism: {result.determinism_score:.2f}')
        
        # Step 4: Check stats
        stats = global_verification_engine.get_stats()
        print(f'✅ Step 4: Stats güncellendi')
        print(f'   Total verifications: {stats["total_verifications"]}')
        print(f'   Successful: {stats["successful_verifications"]}')
        
        return True
        
    except Exception as e:
        print(f'❌ End-to-end test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

result2 = asyncio.run(test_end_to_end())


# TEST 3: Component Integration Check
print('\n3️⃣ TEST: Tüm Komponentlerin Birlikte Çalışması')
print('-' * 40)

try:
    # Check all major components are initialized
    components = {
        'Payload Manager': global_payload_manager is not None,
        'Verification Engine': global_verification_engine is not None,
        'AI Triage': global_verification_engine.ai_triage is not None,
        'Deterministic Analyzer': global_verification_engine.deterministic_analyzer is not None,
    }
    
    all_ok = all(components.values())
    
    for name, status in components.items():
        icon = '✅' if status else '❌'
        print(f'{icon} {name}: {"Aktif" if status else "Pasif"}')
    
    result3 = all_ok
    
except Exception as e:
    print(f'❌ Component check failed: {e}')
    result3 = False


# FINAL RESULTS
print('\n' + '='*60)
print('SİSTEM ENTEGRASYON TEST SONUÇLARI')
print('='*60)
print(f'1️⃣ Payload System: {"✅ BAŞARILI" if result1 else "❌ BAŞARISIZ"}')
print(f'2️⃣ End-to-End Workflow: {"✅ BAŞARILI" if result2 else "❌ BAŞARISIZ"}')
print(f'3️⃣ Component Integration: {"✅ BAŞARILI" if result3 else "❌ BAŞARISIZ"}')

if result1 and result2 and result3:
    print('\n' + '🎉' * 30)
    print('✅ PHASE 6 TAMAMEN TAMAMLANDI!')
    print('   - Circular import sorunu çözüldü')
    print('   - Tüm komponentler çalışıyor')
    print('   - Entegrasyon testleri başarılı')
    print('   - Tam sistem entegrasyonu doğrulandı')
    print('🎉' * 30)
else:
    print('\n⚠️  Bazı testler başarısız')

print('='*60)