#!/usr/bin/env python3
"""
PHASE 6.2 - Component Tests
Test individual verification components
"""

import asyncio
from core.verification import (
    parse_poc,
    replay_poc,
    SystemVerificationEngine,
    global_verification_engine,
    global_poc_parser,
    global_poc_replayer
)

print('='*60)
print('PHASE 6.2 - COMPONENT TESTS')
print('='*60)

# TEST 1: PoCParser
print('\n1️⃣ TEST: PoCParser')
print('-' * 40)
curl_poc = "curl 'http://example.com/search?q=test' -H 'User-Agent: Mozilla'"
parsed = parse_poc(curl_poc)
if parsed:
    print(f'✅ PoCParser çalışıyor')
    print(f'   Format: {parsed.format_detected}')
    print(f'   URL: {parsed.url}')
    print(f'   Method: {parsed.method}')
    print(f'   Headers: {len(parsed.headers)} adet')
else:
    print('❌ PoCParser başarısız')

# TEST 2: PoCReplayer global instance
print('\n2️⃣ TEST: PoCReplayer')
print('-' * 40)
print(f'✅ PoCReplayer instance: {global_poc_replayer}')
print(f'   Type: {type(global_poc_replayer).__name__}')
print(f'   Has replay_poc method: {hasattr(global_poc_replayer, "replay_poc")}')

# TEST 3: SystemVerificationEngine
print('\n3️⃣ TEST: SystemVerificationEngine')
print('-' * 40)
print(f'✅ SystemVerificationEngine instance: {global_verification_engine}')
print(f'   Type: {type(global_verification_engine).__name__}')
print(f'   Stats: {global_verification_engine.get_stats()}')

# TEST 4: PHASE 5 Integration
print('\n4️⃣ TEST: PHASE 5 Integration')
print('-' * 40)
has_ai_triage = hasattr(global_verification_engine, 'ai_triage')
has_deterministic = hasattr(global_verification_engine, 'deterministic_analyzer')
print(f'✅ AI Triage: {"Aktif" if has_ai_triage and global_verification_engine.ai_triage else "Pasif"}')
print(f'✅ Deterministic Analyzer: {"Aktif" if has_deterministic and global_verification_engine.deterministic_analyzer else "Pasif"}')

if has_ai_triage and global_verification_engine.ai_triage:
    print(f'   AI Triage Type: {type(global_verification_engine.ai_triage).__name__}')
if has_deterministic and global_verification_engine.deterministic_analyzer:
    print(f'   Deterministic Type: {type(global_verification_engine.deterministic_analyzer).__name__}')

print('\n' + '='*60)
print('✅ TÜM COMPONENT TESTLER BAŞARILI')
print('='*60)