"""
VORTEX Evidence Standards Configuration - V17.0 ULTIMATE
Evidence validation configuration and standards

Per .clinerules VORTEX_EVIDENCE_STANDARDS.md:
- Evidence hierarchy levels
- Determinism requirements
- Behavioral analysis standards
- Vulnerability-specific criteria

CONFIGURATION:
- Evidence quality thresholds
- Pattern indicators
- Causation uncertainty rules
- Validation requirements
"""

import logging
from typing import Dict, List, Any
from enum import IntEnum

logger = logging.getLogger(__name__)


class EvidenceLevel(IntEnum):
    """
    Evidence hierarchy levels.
    
    Per .clinerules VORTEX_EVIDENCE_STANDARDS.md:
    - DETERMINISTIC: Required for SUBMIT_READY
    - BEHAVIORAL: Required for SYSTEM_VERIFIED
    - PATTERN: Sufficient for AI_CONFIRMED
    """
    DETERMINISTIC = 1  # Highest quality
    BEHAVIORAL = 2
    PATTERN = 3        # Lowest quality


class EvidenceConfig:
    """
    Evidence standards configuration.
    
    CRITICAL: These standards ensure evidence quality.
    Per .clinerules: CANNOT BE LOWERED OR COMPROMISED.
    """
    
    # ============================================================================
    # EVIDENCE HIERARCHY (per .clinerules)
    # ============================================================================
    
    # Evidence level requirements
    EVIDENCE_LEVELS = {
        EvidenceLevel.DETERMINISTIC: {
            'name': 'Deterministic Evidence',
            'min_score': 0.8,
            'required_for': ['SUBMIT_READY'],
            'characteristics': [
                'Reproducible across multiple attempts',
                'Measurable behavioral changes',
                'Independent verification possible',
                'Documented state changes'
            ],
            'examples': [
                'Database error messages with SQL syntax',
                'JavaScript execution confirmation',
                'File content extraction verified',
                'Internal network access documented'
            ]
        },
        EvidenceLevel.BEHAVIORAL: {
            'name': 'Behavioral Evidence',
            'min_score': 0.6,
            'required_for': ['SYSTEM_VERIFIED'],
            'characteristics': [
                'Observable differences in responses',
                'Consistent patterns across attempts',
                'Structural changes in output',
                'Requires causation analysis'
            ],
            'examples': [
                'Response time differentials',
                'Status code changes',
                'Content length variations',
                'Header modifications'
            ]
        },
        EvidenceLevel.PATTERN: {
            'name': 'Pattern Evidence',
            'min_score': 0.4,
            'required_for': ['AI_CONFIRMED'],
            'characteristics': [
                'Text pattern matches',
                'Heuristic rule triggers',
                'Single-instance observations',
                'Needs confirmation'
            ],
            'examples': [
                'Error keywords detected',
                'Suspicious patterns found',
                'Anomalous responses',
                'Initial indicators'
            ]
        }
    }
    
    # ============================================================================
    # TEXT MATCHING LIMITATIONS (per .clinerules)
    # ============================================================================
    
    # CRITICAL: Text matching alone does NOT prove vulnerability
    TEXT_MATCHING_RULES = {
        'proves_vulnerability': False,           # NEVER proves alone
        'indicates_candidate': True,             # Can indicate potential
        'requires_verification': True,           # Always requires verification
        'max_confidence_alone': 0.50,           # Maximum confidence from text only
    }
    
    # Text patterns are indicators, not proof
    TEXT_PATTERN_LIMITATIONS = {
        'what_they_indicate': [
            'Potential vulnerability candidate',
            'Need for deeper verification',
            'Starting point for investigation'
        ],
        'what_they_do_not_indicate': [
            'Confirmed security vulnerability',
            'Backend state changes',
            'Exploitable conditions'
        ]
    }
    
    # ============================================================================
    # BEHAVIORAL DIFFERENCE ANALYSIS (per .clinerules)
    # ============================================================================
    
    # CRITICAL: Behavioral differences are INDICATIVE, not CONCLUSIVE
    BEHAVIORAL_ANALYSIS_RULES = {
        'are_conclusive': False,                 # NEVER conclusive alone
        'acknowledge_uncertainty': True,          # ALWAYS acknowledge uncertainty
        'max_automated_status': 'SYSTEM_VERIFIED', # Never auto-SUBMIT_READY
        'require_causation_analysis': True       # Human analysis needed
    }
    
    # Possible causes of behavioral differences
    BEHAVIORAL_DIFFERENCE_CAUSES = {
        'SECURITY_RELEVANT': [
            'Backend errors',
            'Logic changes',
            'Validation failures',
            'State modifications'
        ],
        'NON_SECURITY': [
            'CDN switching',
            'Load balancing',
            'Cache variations',
            'A/B testing',
            'Dynamic content',
            'Infrastructure changes'
        ],
        'UNCERTAINTY': 'System CANNOT definitively distinguish causes remotely'
    }
    
    # Behavioral indicators with uncertainty factors
    BEHAVIORAL_INDICATORS = {
        'response_time_change': {
            'threshold_seconds': 2.0,
            'uncertainty_factors': ['Could be infrastructure, not application'],
            'confidence_contribution': 0.3
        },
        'status_code_change': {
            'significant_changes': [(200, 403), (200, 500), (404, 200)],
            'uncertainty_factors': ['Could be upstream retry or rate limiting'],
            'confidence_contribution': 0.4
        },
        'content_size_change': {
            'threshold_bytes': 100,
            'uncertainty_factors': ['Could be dynamic content or caching'],
            'confidence_contribution': 0.2
        },
        'header_changes': {
            'significant_headers': ['content-type', 'server', 'x-powered-by'],
            'uncertainty_factors': ['Could be infrastructure reconfiguration'],
            'confidence_contribution': 0.3
        }
    }
    
    # Uncertainty penalty calculation
    UNCERTAINTY_PENALTY = {
        'per_factor': 0.1,                       # Penalty per uncertainty factor
        'max_penalty': 0.4,                      # Maximum total penalty
        'always_apply': True                     # Always acknowledge uncertainty
    }
    
    # ============================================================================
    # EVIDENCE QUALITY SCORING
    # ============================================================================
    
    # Scoring weights by evidence source
    EVIDENCE_SCORING_WEIGHTS = {
        'system_verification': {
            'exact_regex': 0.5,                  # Highest determinism
            'structural_differential': 0.4,      # High determinism
            'fuzzy_match': 0.3,                  # Medium determinism
            'text_pattern': 0.2                  # Lower determinism
        },
        'ai_analysis': {
            'confirmed_verdict': 0.3,            # Advisory contribution
            'likely_verdict': 0.2,               # Lesser advisory
            'uncertain_verdict': 0.1             # Minimal contribution
        },
        'heuristic': {
            'high_confidence': 0.2,              # Indicative only
            'medium_confidence': 0.1             # Lower indication
        }
    }
    
    # Evidence requirements by status
    EVIDENCE_REQUIREMENTS_BY_STATUS = {
        'SUBMIT_READY': {
            'min_determinism': 0.70,
            'min_evidence_level': EvidenceLevel.DETERMINISTIC,
            'max_unknown_values': 0,
            'min_indicators': 2
        },
        'SYSTEM_VERIFIED': {
            'min_determinism': 0.50,
            'min_evidence_level': EvidenceLevel.BEHAVIORAL,
            'max_unknown_values': 2,
            'min_indicators': 1
        },
        'AI_CONFIRMED': {
            'min_determinism': 0.30,
            'min_evidence_level': EvidenceLevel.PATTERN,
            'max_unknown_values': 5,
            'min_indicators': 1
        }
    }
    
    # ============================================================================
    # VULNERABILITY-SPECIFIC EVIDENCE CRITERIA (V11.1)
    # ============================================================================
    
    # Different vulnerability types have different evidence patterns
    VULNERABILITY_EVIDENCE_CRITERIA = {
        'sql_injection': {
            'deterministic_indicators': [
                'mysql error',
                'sql syntax',
                'database error',
                'ora-',
                'postgresql',
                'syntax error at'
            ],
            'confidence_bonus': 0.15,            # Clear errors highly deterministic
            'min_evidence_length': 50,
            'pattern_weight': 1.2
        },
        'xss_reflected': {
            'deterministic_indicators': [
                'javascript execution',
                'alert fired',
                'onerror triggered',
                'script executed',
                'dom manipulation'
            ],
            'confidence_bonus': 0.20,            # JS execution highly deterministic
            'min_evidence_length': 30,
            'pattern_weight': 1.3
        },
        'xss_stored': {
            'deterministic_indicators': [
                'persistent payload',
                'stored successfully',
                'cross-session execution',
                'reflected on other page'
            ],
            'confidence_bonus': 0.18,
            'min_evidence_length': 40,
            'pattern_weight': 1.25
        },
        'ssrf': {
            'deterministic_indicators': [
                'internal response',
                '192.168',
                '10.',
                'localhost',
                '127.0.0.1',
                'internal network'
            ],
            'confidence_bonus': 0.10,            # Internal access deterministic
            'min_evidence_length': 40,
            'pattern_weight': 1.1
        },
        'lfi': {
            'deterministic_indicators': [
                'file content',
                'etc/passwd',
                'system file',
                'root:x:0',
                'windows\\system32'
            ],
            'confidence_bonus': 0.05,            # File inclusion can be ambiguous
            'min_evidence_length': 60,
            'pattern_weight': 0.95
        },
        'command_injection': {
            'deterministic_indicators': [
                'command output',
                'shell execution',
                'uid=',
                'gid=',
                'system response'
            ],
            'confidence_bonus': 0.12,
            'min_evidence_length': 50,
            'pattern_weight': 1.15
        }
    }
    
    # ============================================================================
    # EVIDENCE VALIDATION REQUIREMENTS
    # ============================================================================
    
    # Requirements for evidence to be considered valid
    VALIDATION_REQUIREMENTS = {
        'reproducibility': {
            'min_successful_reproductions': 2,   # Must reproduce at least twice
            'max_failure_rate': 0.3,            # Allow 30% failure rate
            'consistency_threshold': 0.7         # 70% consistency required
        },
        'measurability': {
            'require_quantifiable_changes': True,
            'min_measurable_difference': 0.05,   # 5% minimum difference
            'documented_state_changes': True
        },
        'independence': {
            'third_party_verifiable': False,     # Not always possible
            'multiple_verification_methods': True,
            'documented_procedure': True
        }
    }
    
    # ============================================================================
    # CAUSATION UNCERTAINTY ACKNOWLEDGMENT (per .clinerules)
    # ============================================================================
    
    # CRITICAL: System cannot definitively determine remote causation
    CAUSATION_RULES = {
        'can_determine_definitively': False,     # NEVER claim definitive causation
        'always_acknowledge_uncertainty': True,   # ALWAYS acknowledge uncertainty
        'require_human_analysis': True,          # Human needed for causation
        'default_causation_status': 'UNKNOWN',   # Default when uncertain
        'max_automated_causation_confidence': 0.70  # Never 100% certain
    }
    
    # Causation certainty levels
    CAUSATION_CERTAINTY = {
        'DEFINITIVE': {
            'possible': False,                   # Never possible remotely
            'description': 'Definitive causation cannot be established remotely'
        },
        'HIGH_PROBABILITY': {
            'min_indicators': 3,
            'min_confidence': 0.70,
            'description': 'Multiple strong indicators suggest security cause'
        },
        'POSSIBLE': {
            'min_indicators': 2,
            'min_confidence': 0.50,
            'description': 'Some indicators suggest security relevance'
        },
        'UNCERTAIN': {
            'description': 'Insufficient evidence to determine causation'
        }
    }
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    @staticmethod
    def get_evidence_level_name(level: EvidenceLevel) -> str:
        """Get evidence level name."""
        return EvidenceConfig.EVIDENCE_LEVELS[level]['name']
    
    @staticmethod
    def get_min_score_for_level(level: EvidenceLevel) -> float:
        """Get minimum score required for evidence level."""
        return EvidenceConfig.EVIDENCE_LEVELS[level]['min_score']
    
    @staticmethod
    def get_vulnerability_indicators(vuln_type: str) -> List[str]:
        """Get deterministic indicators for vulnerability type."""
        criteria = EvidenceConfig.VULNERABILITY_EVIDENCE_CRITERIA.get(vuln_type.lower())
        return criteria['deterministic_indicators'] if criteria else []
    
    @staticmethod
    def get_vulnerability_confidence_bonus(vuln_type: str) -> float:
        """Get confidence bonus for vulnerability-specific evidence."""
        criteria = EvidenceConfig.VULNERABILITY_EVIDENCE_CRITERIA.get(vuln_type.lower())
        return criteria['confidence_bonus'] if criteria else 0.0
    
    @staticmethod
    def text_matching_proves_vulnerability() -> bool:
        """Check if text matching alone proves vulnerability (always False)."""
        return EvidenceConfig.TEXT_MATCHING_RULES['proves_vulnerability']
    
    @staticmethod
    def behavioral_differences_conclusive() -> bool:
        """Check if behavioral differences are conclusive (always False)."""
        return EvidenceConfig.BEHAVIORAL_ANALYSIS_RULES['are_conclusive']
    
    @staticmethod
    def can_determine_causation_definitively() -> bool:
        """Check if causation can be determined definitively (always False)."""
        return EvidenceConfig.CAUSATION_RULES['can_determine_definitively']
    
    @staticmethod
    def validate_configuration() -> tuple[bool, List[str]]:
        """
        Validate evidence configuration integrity.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Validate text matching cannot prove alone
        if EvidenceConfig.TEXT_MATCHING_RULES['proves_vulnerability']:
            errors.append("CRITICAL: Text matching cannot prove vulnerability per .clinerules")
        
        # Validate behavioral differences not conclusive
        if EvidenceConfig.BEHAVIORAL_ANALYSIS_RULES['are_conclusive']:
            errors.append("CRITICAL: Behavioral differences not conclusive per .clinerules")
        
        # Validate causation uncertainty
        if EvidenceConfig.CAUSATION_RULES['can_determine_definitively']:
            errors.append("CRITICAL: Cannot determine causation definitively per .clinerules")
        
        # Validate determinism thresholds
        if EvidenceConfig.EVIDENCE_REQUIREMENTS_BY_STATUS['SUBMIT_READY']['min_determinism'] < 0.70:
            errors.append("CRITICAL: SUBMIT_READY determinism must be >= 0.70 per .clinerules")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.error(f"Evidence configuration validation FAILED: {errors}")
        else:
            logger.info("Evidence configuration validated successfully")
        
        return is_valid, errors


# Validate configuration on import
_is_valid, _errors = EvidenceConfig.validate_configuration()
if not _is_valid:
    logger.critical(f"Evidence configuration is INVALID: {_errors}")
    raise RuntimeError(f"Evidence configuration validation failed: {_errors}")