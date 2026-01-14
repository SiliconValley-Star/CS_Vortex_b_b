"""
VORTEX Constants - V17.0 ULTIMATE
All system constants, thresholds, and patterns per .clinerules specifications
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


# ============================================================================
# AUTHORITY HIERARCHY CONSTANTS (VORTEX_CORE_AUTHORITY.md)
# ============================================================================

@dataclass
class AuthorityThresholds:
    """
    Authority hierarchy thresholds - IMMUTABLE per VORTEX_CORE_AUTHORITY.md
    These values define the minimum requirements for each authority level
    """
    # SUBMIT_READY requirements (ALL must be met)
    submit_ready_system_confidence: float = 0.75  # System verification minimum
    submit_ready_evidence_determinism: float = 0.70  # Evidence quality minimum
    submit_ready_ai_support_confidence: float = 0.70  # AI advisory support (if present)
    
    # SYSTEM_VERIFIED requirements
    system_verified_confidence: float = 0.60  # Behavioral evidence minimum
    
    # AI_CONFIRMED requirements (advisory only)
    ai_confirmed_confidence: float = 0.40  # Pattern evidence minimum
    
    # Fastpath promotion (V11.1 enhancement)
    fastpath_system_confidence: float = 0.85  # Strong system verification
    fastpath_evidence_determinism: float = 0.80  # High determinism
    fastpath_total_score: float = 0.75  # Overall fastpath score
    fastpath_min_qualifying_factors: int = 3  # Minimum positive signals
    
    # Very strong evidence (can proceed without AI)
    very_strong_system_confidence: float = 0.90
    
    # Authority violation response
    authority_violation_max_rate: float = 0.01  # Maximum 1% violations allowed


# Global authority thresholds instance
AUTHORITY_THRESHOLDS = AuthorityThresholds()


# ============================================================================
# EVIDENCE STANDARDS CONSTANTS (VORTEX_EVIDENCE_STANDARDS.md)
# ============================================================================

@dataclass
class EvidenceStandards:
    """
    Evidence quality standards per VORTEX_EVIDENCE_STANDARDS.md
    Different levels require different evidence quality
    """
    # Evidence level minimum scores
    deterministic_min_score: float = 0.8  # Required for SUBMIT_READY
    behavioral_min_score: float = 0.6  # Required for SYSTEM_VERIFIED
    pattern_min_score: float = 0.4  # Sufficient for AI_CONFIRMED
    
    # Match type confidence scores
    exact_regex_score: float = 0.5  # Highest determinism
    structural_differential_score: float = 0.4  # High determinism
    fuzzy_match_score: float = 0.3  # Medium determinism
    pattern_match_score: float = 0.2  # Lower determinism
    behavioral_only_score: float = 0.1  # Lowest determinism
    
    # Behavioral analysis thresholds
    behavioral_indicator_weight: float = 0.3  # Per indicator
    behavioral_max_confidence: float = 0.9  # Maximum without deterministic evidence
    uncertainty_penalty_per_factor: float = 0.1  # Penalty per uncertainty factor
    
    # Response change thresholds
    response_time_change_threshold: float = 2.0  # Seconds
    content_size_change_threshold: int = 100  # Bytes
    
    # Payload reflection bonus
    payload_reflection_bonus: float = 0.15


# Global evidence standards instance
EVIDENCE_STANDARDS = EvidenceStandards()


# Vulnerability-specific evidence requirements
# Per VORTEX_EVIDENCE_STANDARDS.md - Different vuln types have different criteria
VULN_SPECIFIC_EVIDENCE = {
    'sqli_error': {
        'deterministic_indicators': ['mysql', 'postgresql', 'sql syntax', 'database error', 'ora-'],
        'confidence_bonus': 0.15,
        'min_evidence_length': 50,
        'submit_threshold': 0.70,  # Lower - high acceptance rate
        'evidence_multiplier': 1.2
    },
    'xss_reflected': {
        'deterministic_indicators': ['javascript execution', 'alert fired', 'onerror triggered', '<script'],
        'confidence_bonus': 0.20,
        'min_evidence_length': 30,
        'submit_threshold': 0.72,  # Medium - context matters
        'evidence_multiplier': 1.15
    },
    'xss_stored': {
        'deterministic_indicators': ['stored', 'persistent', 'saved'],
        'confidence_bonus': 0.18,
        'min_evidence_length': 40,
        'submit_threshold': 0.68,  # Lower - high business impact
        'evidence_multiplier': 1.1
    },
    'ssrf': {
        'deterministic_indicators': ['internal response', '192.168', '10.', 'localhost', '127.0.0.1'],
        'confidence_bonus': 0.10,
        'min_evidence_length': 40,
        'submit_threshold': 0.75,  # Standard threshold
        'evidence_multiplier': 1.05
    },
    'lfi': {
        'deterministic_indicators': ['file content', 'etc/passwd', 'system file', 'root:x:'],
        'confidence_bonus': 0.05,
        'min_evidence_length': 60,
        'submit_threshold': 0.82,  # Higher - prone to false positives
        'evidence_multiplier': 0.95
    },
}


# ============================================================================
# AI INTEGRATION CONSTANTS (VORTEX_AI_INTEGRATION.md)
# ============================================================================

@dataclass
class AIIntegrationLimits:
    """
    AI integration limits - AI is ADVISORY ONLY per VORTEX_AI_INTEGRATION.md
    """
    # AI authority limitations
    ai_is_authoritative: bool = False  # ALWAYS False
    ai_authority_level: str = "ADVISORY_ONLY"
    ai_requires_system_validation: bool = True  # ALWAYS True
    
    # AI confidence penalties
    advisory_confidence_cap: float = 0.95  # Never too confident
    malformed_recovery_confidence_penalty: float = 0.70  # Severe penalty (×0.3)
    malformed_recovery_max_confidence: float = 0.40
    fallback_confidence_penalty: float = 0.30  # Heuristic fallback penalty
    
    # Consensus building
    consensus_agreement_boost: float = 1.1  # Boost for model agreement
    consensus_disagreement_penalty: float = 0.9  # Penalty for disagreement
    hermes_weight: float = 0.7  # Uncensored analysis priority
    gemini_weight: float = 0.3  # Validation support
    
    # Field derivation prohibition
    allow_field_derivation: bool = False  # NEVER derive missing fields
    
    # PoC replay restrictions
    allow_heuristic_poc_replay: bool = False  # NEVER replay heuristic PoCs
    allow_ai_poc_replay: bool = True  # Only AI-generated PoCs
    
    # Availability thresholds
    min_ai_availability: float = 0.70  # Minimum for system health
    degraded_availability: float = 0.50  # Degraded mode threshold


# Global AI integration limits instance
AI_LIMITS = AIIntegrationLimits()


# ============================================================================
# OPERATIONAL HEALTH CONSTANTS (VORTEX_OPERATIONAL_HEALTH.md)
# ============================================================================

@dataclass
class OperationalHealthThresholds:
    """
    System health thresholds per VORTEX_OPERATIONAL_HEALTH.md
    These ensure system remains viable and effective
    """
    # Finding distribution (V11.1 adjusted for fastpath)
    max_manual_review_rate: float = 0.75  # <75% (reduced from 80%)
    min_submit_ready_rate: float = 0.03  # >3% (increased from 2%)
    target_submit_ready_rate: float = 0.05  # Target: 5-8%
    max_false_positive_rate: float = 0.15  # <15% (tightened)
    
    # Manual review efficiency
    max_avg_manual_hours: float = 48.0  # <48h average review time
    min_manual_conversion: float = 0.25  # >25% manual to submission
    max_queue_growth_rate: float = 0.15  # <15% daily growth
    
    # System performance
    min_ai_availability: float = 0.70  # >70% AI model availability
    min_verification_success: float = 0.60  # >60% PoC verification success
    min_system_accuracy: float = 0.80  # >80% system verification accuracy
    
    # Resource limits
    max_memory_mb: float = 6000.0  # <6GB memory usage
    memory_cleanup_threshold: float = 0.85  # Cleanup at 85%
    memory_emergency_threshold: float = 0.95  # Emergency at 95%
    max_error_rate: float = 0.08  # <8% error rate (tightened)
    min_throughput_rps: float = 1.0  # >1 request/second sustained
    
    # Authority compliance (NEW per .clinerules)
    max_authority_violation_rate: float = 0.01  # <1% violations
    min_evidence_determinism_avg: float = 0.70  # >0.70 average
    max_unknown_value_rate: float = 0.10  # <10% unknowns


# Global health thresholds instance
HEALTH_THRESHOLDS = OperationalHealthThresholds()


# V11.1 Success Metrics Targets
V11_1_TARGETS = {
    'submit_ready_rate': {'target': 0.05, 'min': 0.03, 'baseline': 0.025},
    'manual_queue_reduction': {'target': 0.70, 'max': 0.75, 'baseline': 0.75},
    'quality_preservation': {'target': 0.85, 'min': 0.85, 'baseline': 0.875},
    'false_positive_rate': {'target': 0.12, 'max': 0.15, 'baseline': 0.15},
    'manual_conversion_rate': {'target': 0.25, 'min': 0.25, 'baseline': 0.25},
}


# ============================================================================
# WORKFLOW & LIFECYCLE CONSTANTS (VORTEX_WORKFLOW_LIFECYCLE.md)
# ============================================================================

@dataclass
class WorkflowSLATargets:
    """
    Workflow SLA targets per VORTEX_WORKFLOW_LIFECYCLE.md
    """
    # Phase timing targets (seconds)
    detection_to_ai_analysis: int = 300  # <5 minutes
    ai_analysis_avg: int = 30  # <30 seconds average
    system_verification_avg: int = 60  # <60 seconds average
    manual_review_registration: int = 5  # <5 seconds
    total_automated_workflow: int = 600  # <10 minutes
    
    # Manual review SLAs (hours)
    critical_max_age: int = 24
    critical_escalation: int = 12
    high_max_age: int = 48
    high_escalation: int = 24
    medium_max_age: int = 72
    medium_escalation: int = 48
    low_max_age: int = 168  # 1 week
    low_escalation: int = 96
    backlog_max_age: int = 720  # 30 days
    backlog_escalation: int = 480


# Global workflow SLA targets instance
WORKFLOW_SLA = WorkflowSLATargets()


# ============================================================================
# DETECTION PATTERNS & PAYLOADS
# ============================================================================

# SQL Injection patterns
SQLI_ERROR_PATTERNS = [
    r'SQL syntax.*MySQL',
    r'Warning.*mysql_',
    r'valid MySQL result',
    r'MySqlClient\.',
    r'PostgreSQL.*ERROR',
    r'Warning.*\Wpg_',
    r'valid PostgreSQL result',
    r'Npgsql\.',
    r'Driver.*SQL',
    r'ORA-\d{5}',
    r'Oracle error',
    r'Microsoft SQL Server',
    r'ODBC SQL Server Driver',
    r'SQLServer JDBC Driver',
    r'Syntax error in string in query expression',
    r'ERROR:\s+parser:',
]

# XSS detection patterns
XSS_REFLECTION_PATTERNS = [
    r'<script[^>]*>.*</script>',
    r'onerror\s*=',
    r'onload\s*=',
    r'javascript:',
    r'<img[^>]+src',
    r'<iframe',
    r'eval\s*\(',
]

# SSRF detection patterns
SSRF_INTERNAL_INDICATORS = [
    r'192\.168\.\d+\.\d+',
    r'10\.\d+\.\d+\.\d+',
    r'172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+',
    r'localhost',
    r'127\.0\.0\.1',
    r'0\.0\.0\.0',
    r'metadata\.google\.internal',
    r'169\.254\.169\.254',  # AWS metadata
]

# LFI/Path Traversal patterns
LFI_PATTERNS = [
    r'\.\./',
    r'\.\.\\',
    r'/etc/passwd',
    r'C:\\Windows\\',
    r'/proc/self/',
    r'root:x:0:0:',
]

# Command Injection patterns
COMMAND_INJECTION_PATTERNS = [
    r'uid=\d+',
    r'gid=\d+',
    r'groups=',
    r'Linux version',
    r'Windows.*Microsoft',
]


# ============================================================================
# SECURITY & COMPLIANCE CONSTANTS
# ============================================================================

@dataclass
class SecurityLimits:
    """Security and rate limiting constants."""
    # Rate limiting
    max_requests_per_minute: int = 120
    max_requests_per_domain: int = 10
    max_concurrent_scans: int = 5
    burst_size: int = 20
    backoff_seconds: int = 60
    
    # WAF evasion
    waf_detection_threshold: int = 3  # Failed requests before evasion
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 2
    
    # Request timeouts
    default_timeout: int = 30
    verification_timeout: int = 60
    max_redirect_follow: int = 5
    
    # Payload limits
    max_payload_size: int = 10000  # 10KB
    max_evidence_size: int = 100000  # 100KB
    max_response_capture: int = 50000  # 50KB


# Global security limits instance
SECURITY_LIMITS = SecurityLimits()


# Data retention periods (days)
DATA_RETENTION = {
    'finding_data': 90,
    'evidence': 365,
    'logs': 30,
    'pii': 7,  # Minimal retention for PII
    'backup': 30,
    'audit_trail': 365,
}


# ============================================================================
# USER AGENTS & HEADERS
# ============================================================================

DEFAULT_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

COMMON_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
}

# Metrics collection intervals (seconds)
METRICS_INTERVALS = {
    'health_check': 30,
    'memory_check': 60,
    'queue_check': 120,
    'performance_check': 300,
}


# ============================================================================
# FILE PATHS & DIRECTORIES
# ============================================================================

DEFAULT_PATHS = {
    'database': 'output/database/vortex.db',
    'evidence_db': 'output/database/evidence_integrity.db',
    'authority_audit_db': 'output/database/authority_audit.db',
    'logs': 'output/logs/application.log',
    'authority_log': 'output/logs/authority.log',
    'evidence_log': 'output/logs/evidence.log',
    'health_log': 'output/logs/health.log',
    'reports': 'output/reports/',
    'submissions': 'output/submissions/',
    'evidence': 'output/evidence/',
    'backups': 'output/database/backup/',
}


# Aliases for backward compatibility
VULNERABILITY_EVIDENCE_CRITERIA = VULN_SPECIFIC_EVIDENCE
AI_INTEGRATION_LIMITS = AI_LIMITS
OPERATIONAL_HEALTH_THRESHOLDS = HEALTH_THRESHOLDS

# Export all constants
__all__ = [
    'AUTHORITY_THRESHOLDS',
    'EVIDENCE_STANDARDS',
    'VULN_SPECIFIC_EVIDENCE',
    'VULNERABILITY_EVIDENCE_CRITERIA',
    'AI_LIMITS',
    'AI_INTEGRATION_LIMITS',
    'HEALTH_THRESHOLDS',
    'OPERATIONAL_HEALTH_THRESHOLDS',
    'V11_1_TARGETS',
    'WORKFLOW_SLA',
    'SQLI_ERROR_PATTERNS',
    'XSS_REFLECTION_PATTERNS',
    'SSRF_INTERNAL_INDICATORS',
    'LFI_PATTERNS',
    'COMMAND_INJECTION_PATTERNS',
    'SECURITY_LIMITS',
    'DATA_RETENTION',
    'DEFAULT_USER_AGENTS',
    'COMMON_HEADERS',
    'LOG_LEVELS',
    'METRICS_INTERVALS',
    'DEFAULT_PATHS',
]