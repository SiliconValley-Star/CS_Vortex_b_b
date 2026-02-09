"""
VORTEX Verification Module - V17.0 ULTIMATE
PoC replay and pattern-based verification system

This module provides comprehensive verification capabilities:
- PoC parsing (cURL, HTTP raw, Python requests)
- PoC replay with behavioral analysis
- Pattern-based verification
- Determinism scoring

USAGE:
    from core.verification import parse_poc, replay_poc, global_verification_engine
    
    # Parse PoC
    parsed = parse_poc(curl_command)
    
    # Replay PoC
    result = await replay_poc(parsed, original_url)
    
    # Verify finding
    verification = await global_verification_engine.verify_finding(finding)

CRITICAL: Only AI-generated PoCs should be replayed (never heuristic)
"""

from core.verification.poc_parser import (
    PoCParser,
    ParsedPoC,
    global_poc_parser,
    parse_poc
)

from core.verification.poc_replayer import (
    PoCReplayer,
    ReplayResult,
    global_poc_replayer,
    replay_poc
)

# Import from system_verification module (renamed to avoid circular import)
try:
    from core.system_verification import SystemVerificationEngine, global_verification_engine
except ImportError as e:
    # If import fails, define placeholder
    import logging
    logging.warning(f"Failed to import SystemVerificationEngine: {e}")
    SystemVerificationEngine = None
    global_verification_engine = None


__all__ = [
    # Parser
    'PoCParser',
    'ParsedPoC',
    'global_poc_parser',
    'parse_poc',
    
    # Replayer
    'PoCReplayer',
    'ReplayResult',
    'global_poc_replayer',
    'replay_poc',
    
    # Verification Engine
    'SystemVerificationEngine',
    'global_verification_engine',
]

__version__ = '17.0.0'