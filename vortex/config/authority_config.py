"""
VORTEX Authority Hierarchy Configuration - V17.0 ULTIMATE
Authority system configuration and thresholds

Per .clinerules VORTEX_CORE_AUTHORITY.md:
- Immutable authority hierarchy
- AI advisory-only limitations
- SUBMIT_READY requirements
- Authority enforcement rules

CONFIGURATION:
- Authority level priorities
- Confidence thresholds
- Evidence requirements
- Validation rules
"""

import logging
from typing import Dict, List, Any
from enum import IntEnum

logger = logging.getLogger(__name__)


class AuthorityLevel(IntEnum):
    """
    Authority hierarchy levels.
    
    Per .clinerules: IMMUTABLE HIERARCHY (NEVER violated)
    1. System Verification (Deterministic - Highest)
    2. Human Expert (Authoritative - Second)
    3. AI Analysis (Advisory ONLY - Third)
    4. Heuristic (Indicative - Lowest)
    """
    SYSTEM_VERIFICATION = 1  # Highest authority
    HUMAN_EXPERT = 2
    AI_ADVISORY = 3          # NEVER authoritative
    HEURISTIC = 4            # Lowest authority


class AuthorityConfig:
    """
    Authority system configuration.
    
    CRITICAL: These settings enforce the authority hierarchy.
    DO NOT modify without understanding implications.
    """
    
    # ============================================================================
    # IMMUTABLE AUTHORITY HIERARCHY
    # ============================================================================
    
    # Authority level names
    AUTHORITY_NAMES = {
        AuthorityLevel.SYSTEM_VERIFICATION: "System Verification",
        AuthorityLevel.HUMAN_EXPERT: "Human Expert",
        AuthorityLevel.AI_ADVISORY: "AI Advisory",
        AuthorityLevel.HEURISTIC: "Heuristic"
    }
    
    # Authority descriptions
    AUTHORITY_DESCRIPTIONS = {
        AuthorityLevel.SYSTEM_VERIFICATION: "Deterministic evidence from system verification",
        AuthorityLevel.HUMAN_EXPERT: "Authoritative analysis from security expert",
        AuthorityLevel.AI_ADVISORY: "Advisory input from AI models (NOT authoritative)",
        AuthorityLevel.HEURISTIC: "Indicative patterns from automated detection"
    }
    
    # ============================================================================
    # SUBMIT_READY REQUIREMENTS (per .clinerules)
    # ============================================================================
    
    # ALL of these MUST be true for SUBMIT_READY status
    SUBMIT_READY_REQUIREMENTS = {
        'system_verification_required': True,      # MANDATORY
        'min_system_confidence': 0.75,            # Minimum threshold
        'no_unknown_values': True,                # MANDATORY
        'deterministic_evidence': True,           # MANDATORY
        'authority_chain_valid': True,            # MANDATORY
    }
    
    # Confidence thresholds by status
    CONFIDENCE_THRESHOLDS = {
        'SUBMIT_READY': 0.75,         # High confidence required
        'SYSTEM_VERIFIED': 0.60,      # Moderate confidence
        'AI_CONFIRMED': 0.50,         # Lower confidence (advisory)
        'NEEDS_MANUAL': 0.0           # Any confidence level
    }
    
    # ============================================================================
    # AI AUTHORITY LIMITATIONS (per .clinerules)
    # ============================================================================
    
    # AI is NEVER authoritative - only advisory
    AI_LIMITATIONS = {
        'is_authoritative': False,              # NEVER True
        'can_make_final_decisions': False,      # NEVER True
        'can_bypass_system_verification': False, # NEVER True
        'can_fill_unknown_values': False,       # NEVER True
        'can_derive_missing_fields': False,     # NEVER True
        'role': 'ADVISORY_ONLY'                 # Always advisory
    }
    
    # AI advisory support thresholds
    AI_ADVISORY_THRESHOLDS = {
        'min_confidence_for_support': 0.70,     # Minimum to provide support
        'max_advisory_weight': 0.30,            # Maximum influence on decisions
        'consensus_boost': 0.10,                # Bonus for multi-model agreement
        'disagreement_penalty': 0.15            # Penalty for model disagreement
    }
    
    # ============================================================================
    # UNKNOWN VALUE HANDLING (per .clinerules)
    # ============================================================================
    
    # CRITICAL: UNKNOWN ≠ LOW ≠ FALSE ≠ 0
    UNKNOWN_VALUE_POLICY = {
        'allow_in_submit_ready': False,         # NEVER allow UNKNOWN in SUBMIT_READY
        'route_to_manual': True,                # Always route to manual review
        'never_convert_to_low': True,           # NEVER convert UNKNOWN to LOW
        'never_derive_from_other_fields': True, # NEVER derive missing fields
        'preserve_ai_uncertainty': True         # Preserve AI's uncertainty
    }
    
    # Fields that cannot be UNKNOWN for SUBMIT_READY
    REQUIRED_KNOWN_FIELDS = [
        'impact',
        'exploitability',
        'reportability',
        'severity'
    ]
    
    # ============================================================================
    # EVIDENCE DETERMINISM REQUIREMENTS
    # ============================================================================
    
    # Evidence types by determinism level
    EVIDENCE_DETERMINISM = {
        'HIGH': {
            'types': ['exact_regex', 'structural_differential', 'database_error_confirmed'],
            'min_score': 0.8,
            'required_for': ['SUBMIT_READY']
        },
        'MEDIUM': {
            'types': ['fuzzy_match', 'behavioral_differential', 'pattern_match'],
            'min_score': 0.6,
            'required_for': ['SYSTEM_VERIFIED']
        },
        'LOW': {
            'types': ['text_pattern', 'heuristic_trigger'],
            'min_score': 0.4,
            'required_for': ['AI_CONFIRMED']
        }
    }
    
    # ============================================================================
    # FASTPATH PROMOTION CRITERIA (V11.1)
    # ============================================================================
    
    # Fastpath allows qualified findings to bypass extended queues
    FASTPATH_CRITERIA = {
        'min_system_confidence': 0.85,          # Strong system evidence
        'min_evidence_determinism': 0.70,       # High quality evidence
        'min_qualifying_factors': 3,            # Multiple positive signals
        'max_blocking_factors': 0,              # Zero blocking issues
        'min_fastpath_score': 0.75              # 75% minimum overall score
    }
    
    # Fastpath bonus factors
    FASTPATH_BONUSES = {
        'high_system_confidence': 0.30,         # confidence >= 0.85
        'high_determinism': 0.25,               # determinism >= 0.8
        'no_unknown_values': 0.20,              # All fields known
        'vuln_specific_evidence': 0.15,         # Type-specific indicators
        'ai_supportive': 0.10                   # AI agreement (when available)
    }
    
    # ============================================================================
    # VULNERABILITY-SPECIFIC THRESHOLDS (V11.1)
    # ============================================================================
    
    # Different vulnerability types have different acceptance patterns
    VULNERABILITY_THRESHOLDS = {
        'sql_injection': {
            'submit_threshold': 0.70,           # Lower - high acceptance rate
            'evidence_multiplier': 1.2,         # SQL errors very deterministic
            'fastpath_eligible': True
        },
        'xss_reflected': {
            'submit_threshold': 0.72,           # Medium - context matters
            'evidence_multiplier': 1.15,        # JS execution deterministic
            'fastpath_eligible': True
        },
        'xss_stored': {
            'submit_threshold': 0.68,           # Lower - high impact
            'evidence_multiplier': 1.1,
            'fastpath_eligible': True
        },
        'ssrf': {
            'submit_threshold': 0.75,           # Standard threshold
            'evidence_multiplier': 1.05,
            'fastpath_eligible': True
        },
        'lfi': {
            'submit_threshold': 0.82,           # Higher - prone to FPs
            'evidence_multiplier': 0.95,
            'fastpath_eligible': False          # Requires careful review
        },
        'command_injection': {
            'submit_threshold': 0.78,
            'evidence_multiplier': 1.1,
            'fastpath_eligible': True
        }
    }
    
    # ============================================================================
    # AUTHORITY VIOLATION DETECTION
    # ============================================================================
    
    # Settings for detecting authority hierarchy violations
    VIOLATION_DETECTION = {
        'enabled': True,                        # Always enabled
        'log_violations': True,                 # Log all violations
        'block_violations': True,               # Block violating operations
        'alert_on_violation': True,             # Generate alerts
        'violation_severity': 'CRITICAL'        # Treat as critical
    }
    
    # Actions on authority violation
    VIOLATION_ACTIONS = {
        'revert_to_manual': True,               # Route to manual review
        'invalidate_finding': False,            # Don't auto-invalidate
        'escalate_to_human': True,              # Require human review
        'create_audit_record': True             # Audit trail
    }
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def get_authority_name(level: AuthorityLevel) -> str:
        """Get human-readable authority name."""
        return AuthorityConfig.AUTHORITY_NAMES.get(level, "Unknown")
    
    @staticmethod
    def get_authority_description(level: AuthorityLevel) -> str:
        """Get authority level description."""
        return AuthorityConfig.AUTHORITY_DESCRIPTIONS.get(level, "")
    
    @staticmethod
    def get_submit_ready_threshold() -> float:
        """Get minimum confidence for SUBMIT_READY."""
        return AuthorityConfig.SUBMIT_READY_REQUIREMENTS['min_system_confidence']
    
    @staticmethod
    def is_ai_authoritative() -> bool:
        """Check if AI can be authoritative (always False per .clinerules)."""
        return AuthorityConfig.AI_LIMITATIONS['is_authoritative']
    
    @staticmethod
    def allows_unknown_in_submit_ready() -> bool:
        """Check if UNKNOWN values allowed in SUBMIT_READY (always False)."""
        return AuthorityConfig.UNKNOWN_VALUE_POLICY['allow_in_submit_ready']
    
    @staticmethod
    def get_vulnerability_threshold(vuln_type: str) -> float:
        """Get SUBMIT_READY threshold for vulnerability type."""
        config = AuthorityConfig.VULNERABILITY_THRESHOLDS.get(vuln_type.lower())
        return config['submit_threshold'] if config else 0.75
    
    @staticmethod
    def validate_configuration() -> tuple[bool, List[str]]:
        """
        Validate authority configuration integrity.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Validate AI is never authoritative
        if AuthorityConfig.AI_LIMITATIONS['is_authoritative']:
            errors.append("CRITICAL: AI cannot be authoritative per .clinerules")
        
        # Validate UNKNOWN handling
        if AuthorityConfig.UNKNOWN_VALUE_POLICY['allow_in_submit_ready']:
            errors.append("CRITICAL: UNKNOWN values cannot be in SUBMIT_READY per .clinerules")
        
        # Validate confidence thresholds
        if AuthorityConfig.SUBMIT_READY_REQUIREMENTS['min_system_confidence'] < 0.75:
            errors.append("CRITICAL: SUBMIT_READY threshold must be >= 0.75 per .clinerules")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.error(f"Authority configuration validation FAILED: {errors}")
        else:
            logger.info("Authority configuration validated successfully")
        
        return is_valid, errors


# Validate configuration on import
_is_valid, _errors = AuthorityConfig.validate_configuration()
if not _is_valid:
    logger.critical(f"Authority configuration is INVALID: {_errors}")
    raise RuntimeError(f"Authority configuration validation failed: {_errors}")