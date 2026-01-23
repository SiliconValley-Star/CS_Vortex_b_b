"""
VORTEX AI Integration - V17.0 ULTIMATE
AI advisory system per VORTEX_AI_INTEGRATION.md

This package provides:
- OpenRouter AI integration with multiple models
- AI advisory analysis (NEVER authoritative)
- Comprehensive fallback strategies
- Authority-compliant AI usage

CRITICAL: AI IS ADVISORY ONLY
Per VORTEX_CORE_AUTHORITY.md: AI provides expert opinion simulation, NOT security verdicts

AI Analysis → Advisory Input → Human/System Authority → Final Decision

GOLDEN RULE: AI CANNOT BE FINAL AUTHORITY
- AI supports decisions but never makes them
- AI alone cannot create SUBMIT_READY status
- AI fields must never be derived if missing
- Heuristic PoCs must never be replayed
- Malformed JSON recovery is non-authoritative
"""

from core.ai.openrouter import (
    OpenRouterClient,
    global_openrouter_client
)

from core.ai.advisory import (
    AIAdvisoryEngine,
    perform_ai_advisory_analysis,
    global_ai_advisory_engine
)

from core.ai.fallbacks import (
    AIFallbackManager,
    create_heuristic_fallback_result,
    create_ai_unavailable_result,
    global_fallback_manager
)

from core.ai.triage_mode import (
    AITriageMode,
    global_triage_mode,
    should_use_ai_for_finding,
    TriageDecision
)

__all__ = [
    # OpenRouter client
    'OpenRouterClient',
    'global_openrouter_client',
    
    # Advisory engine
    'AIAdvisoryEngine',
    'perform_ai_advisory_analysis',
    'global_ai_advisory_engine',
    
    # Fallback management
    'AIFallbackManager',
    'create_heuristic_fallback_result',
    'create_ai_unavailable_result',
    'global_fallback_manager',
    
    # Triage mode (PHASE 5.1)
    'AITriageMode',
    'global_triage_mode',
    'should_use_ai_for_finding',
    'TriageDecision',
]