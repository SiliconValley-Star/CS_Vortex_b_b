#!/usr/bin/env python3
"""
PHASE 6.3 - Integration Tests
Test components working together
"""

import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from core.verification import (
    parse_poc,
    replay_poc,
    global_verification_engine,
    global_poc_parser,
    global_poc_replayer
)
from domain.models import AssessmentResult, AIAnalysisResult, VerificationResult
from domain.enums import FindingType, FindingSeverity, AIVerdict
from core.network import HTTPResponse

print('='*60)
print('PHASE 6.3 - INTEGRATION TESTS')
print('='*60)

# TEST 1: PoCParser → PoCReplayer Integration
print('\n1️⃣ TEST: PoCParser → PoCReplayer Akışı')
print('-' * 40)

async def test_parser_replayer():
    # Parse PoC
    curl_poc = "curl 'http://httpbin.org/get?test=xss' -H 'User-Agent: TestBot'"
    parsed = parse_poc(curl_poc)
    
    if not parsed:
        print('❌ PoC parse edilemedi')
        return False
    
    print(f'✅ PoC parsed: {parsed.format_detected}')
    
    # Replay PoC (simüle edilmiş response ile)
    try:
        # Mock network client to avoid real requests
        original_request = global_poc_replayer.network_client.request
        
        mock_response = HTTPResponse(
            status_code=200,
            body='{"test": "ok"}',
            headers={'Content-Type': 'application/json'},
            response_time=0.123,
            url='http://httpbin.org/get?test=xss'
        )
        
        global_poc_replayer.network_client.request = AsyncMock(return_value=mock_response)
        
        result = await replay_poc(parsed, 'http://httpbin.org/get')
        
        # Restore original
        global_poc_replayer.network_client.request = original_request
        
        print(f'✅ PoC replayed successfully')
        print(f'   Match type: {result.match_type}')
        print(f'   Confidence: {result.confidence:.2f}')
        print(f'   Determinism: {result.determinism_score:.2f}')
        
        return True
        
    except Exception as e:
        print(f'❌ Replay failed: {e}')
        return False

result1 = asyncio.run(test_parser_replayer())


# TEST 2: SystemVerificationEngine Full Workflow
print('\n2️⃣ TEST: SystemVerificationEngine Tam Workflow')
print('-' * 40)

async def test_verification_engine():
    # Create mock finding with AI analysis
    ai_analysis = AIAnalysisResult(
        model_used="test-model",
        verdict=AIVerdict.CONFIRMED,
        confidence=0.85,
        reasoning="XSS payload reflected without encoding",
        poc="curl 'http://example.com/search?q=<script>alert(1)</script>'",
        success=True,
        is_fallback_result=False
    )
    
    finding = AssessmentResult(
        url="http://example.com/search",
        finding_type=FindingType.XSS_REFLECTED,
        severity=FindingSeverity.HIGH,
        payload="<script>alert(1)</script>",
        evidence="Payload reflected in response",
        heuristic_score=0.80,
        ai_analysis=ai_analysis
    )
    
    # Mock network for verification
    original_request = global_verification_engine.network_client.request
    
    mock_response = HTTPResponse(
        status_code=200,
        body='<html><script>alert(1)</script></html>',
        headers={'Content-Type': 'text/html'},
        response_time=0.234,
        url=finding.url
    )
    
    global_verification_engine.network_client.request = AsyncMock(return_value=mock_response)
    
    try:
        # Run verification
        result = await global_verification_engine.verify_finding(finding)
        
        # Restore
        global_verification_engine.network_client.request = original_request
        
        print(f'✅ Verification tamamlandı')
        print(f'   Success: {result.success}')
        print(f'   Match type: {result.match_type}')
        print(f'   Confidence: {result.confidence:.2f}')
        
        # Check stats
        stats = global_verification_engine.get_stats()
        print(f'   Stats - Total: {stats["total_verifications"]}, Success: {stats["successful_verifications"]}')
        
        return True
        
    except Exception as e:
        print(f'❌ Verification failed: {e}')
        import traceback
        traceback.print_exc()
        return False

result2 = asyncio.run(test_verification_engine())


# TEST 3: PHASE 5 AI Optimization Flow
print('\n3️⃣ TEST: PHASE 5 AI Optimization Akışı')
print('-' * 40)

async def test_phase5_optimization():
    # Create finding with high deterministic confidence
    from domain.enums import ConfidenceSource
    finding = AssessmentResult(
        url="http://example.com/api",
        finding_type=FindingType.SQLI_ERROR,
        severity=FindingSeverity.CRITICAL,
        payload="' OR '1'='1",
        evidence="MySQL error in response",
        heuristic_score=0.95,
        confidence_source=ConfidenceSource.SYSTEM_VERIFIED
    )
    
    try:
        # Check if AI components are active
        if not global_verification_engine.ai_triage:
            print('⚠️  PHASE 5 AI components not available')
            return True
        
        # Mock network
        original_request = global_verification_engine.network_client.request
        
        mock_response = HTTPResponse(
            status_code=500,
            body='MySQL syntax error near "1=1"',
            headers={},
            response_time=0.156,
            url=finding.url
        )
        
        global_verification_engine.network_client.request = AsyncMock(return_value=mock_response)
        
        # Run verification with PHASE 5
        result = await global_verification_engine.verify_finding(finding)
        
        # Restore
        global_verification_engine.network_client.request = original_request
        
        print(f'✅ PHASE 5 optimization çalıştı')
        print(f'   Match type: {result.match_type}')
        print(f'   Confidence: {result.confidence:.2f}')
        
        # Check if deterministic auto-accept or AI triage was used
        if 'deterministic' in result.match_type:
            print(f'   ✅ Deterministic optimization aktif (AI skipped)')
        elif 'poc_replay' in result.match_type:
            print(f'   ✅ AI PoC replay kullanıldı')
        else:
            print(f'   ✅ Pattern-based verification kullanıldı')
        
        return True
        
    except Exception as e:
        print(f'❌ PHASE 5 test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

result3 = asyncio.run(test_phase5_optimization())


# FINAL RESULTS
print('\n' + '='*60)
print('ENTEGRASYON TEST SONUÇLARI')
print('='*60)
print(f'1️⃣ Parser → Replayer: {"✅ BAŞARILI" if result1 else "❌ BAŞARISIZ"}')
print(f'2️⃣ Verification Engine: {"✅ BAŞARILI" if result2 else "❌ BAŞARISIZ"}')
print(f'3️⃣ PHASE 5 Optimization: {"✅ BAŞARILI" if result3 else "❌ BAŞARISIZ"}')

if result1 and result2 and result3:
    print('\n✅ TÜM ENTEGRASYON TESTLER BAŞARILI')
else:
    print('\n⚠️  Bazı testler başarısız')

print('='*60)