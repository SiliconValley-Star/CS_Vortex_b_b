"""
VORTEX Domain Enums - V17.0 ULTIMATE
Core enumeration types following .clinerules authority hierarchy and state machine
"""

from enum import Enum, IntEnum


class VerificationStatus(str, Enum):
    """
    Finding verification status - STRICT state machine enforcement
    Follows VORTEX_WORKFLOW_LIFECYCLE.md state transitions
    """
    # Initial detection
    DETECTED = "DETECTED"
    AI_ANALYSIS_PENDING = "AI_ANALYSIS_PENDING"
    
    # AI analysis results (ADVISORY ONLY - never authoritative)
    AI_CONFIRMED = "AI_CONFIRMED"
    AI_FAILED = "AI_FAILED"
    
    # System verification states (AUTHORITATIVE)
    SYSTEM_VERIFICATION_PENDING = "SYSTEM_VERIFICATION_PENDING"
    SYSTEM_VERIFIED = "SYSTEM_VERIFIED"
    SYSTEM_VERIFICATION_FAILED = "SYSTEM_VERIFICATION_FAILED"
    
    # Final states
    SUBMIT_READY = "SUBMIT_READY"  # Requires ALL authority requirements
    NEEDS_MANUAL = "NEEDS_MANUAL_REVIEW"  # Active status requiring attention
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ERROR_STATE = "ERROR_STATE"


class FindingType(str, Enum):
    """Vulnerability types with evidence-specific criteria."""
    SQLI_ERROR = "sql_injection_error_based"
    SQLI_BLIND = "sql_injection_blind"
    SQLI_TIME = "sql_injection_time_based"
    
    XSS_REFLECTED = "xss_reflected"
    XSS_STORED = "xss_stored"
    XSS_DOM = "xss_dom_based"
    
    LFI = "local_file_inclusion"
    RFI = "remote_file_inclusion"
    
    SSRF = "server_side_request_forgery"
    SSRF_BLIND = "ssrf_blind"
    
    IDOR = "insecure_direct_object_reference"
    OPEN_REDIRECT = "open_redirect"
    XXE = "xml_external_entity"
    
    AUTH_BYPASS = "authentication_bypass"
    AUTHZ_BYPASS = "authorization_bypass"
    
    COMMAND_INJECTION = "command_injection"
    CODE_INJECTION = "code_injection"
    
    SSTI = "server_side_template_injection"
    CSTI = "client_side_template_injection"
    
    CSRF = "cross_site_request_forgery"
    CORS_MISCONFIGURATION = "cors_misconfiguration"
    
    INFO_DISCLOSURE = "information_disclosure"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    
    # Generic/Unknown type
    OTHER = "other"


class FindingSeverity(str, Enum):
    """Severity levels for findings."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AIVerdict(str, Enum):
    """
    AI analysis verdict - ADVISORY ONLY
    AI verdicts are NEVER authoritative per VORTEX_CORE_AUTHORITY.md
    """
    CONFIRMED = "CONFIRMED"  # AI suggests confirmation (advisory)
    LIKELY = "LIKELY"  # AI suggests likelihood (advisory)
    FALSE_POSITIVE = "FALSE_POSITIVE"  # AI suggests FP (advisory)
    NEEDS_MANUAL = "NEEDS_MANUAL"  # AI cannot determine (advisory)


class ImpactLevel(str, Enum):
    """
    Impact assessment levels
    UNKNOWN ≠ LOW - they have different meanings per VORTEX_CORE_AUTHORITY.md
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"  # Insufficient information to determine


class ConfidenceSource(str, Enum):
    """Source of confidence scoring - determines PoC replay eligibility."""
    AI_GENERATED = "AI_GENERATED"  # AI-created PoC (replayable)
    HEURISTIC_ONLY = "HEURISTIC_ONLY"  # Pattern-based (NOT replayable)
    SYSTEM_VERIFIED = "SYSTEM_VERIFIED"  # System evidence (authoritative)
    HUMAN_EXPERT = "HUMAN_EXPERT"  # Manual analysis (authoritative)


class EvidenceLevel(str, Enum):
    """
    Evidence quality levels per VORTEX_EVIDENCE_STANDARDS.md
    Different levels have different confidence requirements
    """
    DETERMINISTIC = "DETERMINISTIC"  # ≥0.8 score, required for SUBMIT_READY
    BEHAVIORAL = "BEHAVIORAL"  # ≥0.6 score, required for SYSTEM_VERIFIED
    PATTERN = "PATTERN"  # ≥0.4 score, sufficient for AI_CONFIRMED
    INSUFFICIENT = "INSUFFICIENT"  # <0.4 score


class MatchType(str, Enum):
    """System verification match types - determinism indicators."""
    EXACT_REGEX = "exact_regex"  # Highest determinism (0.5 score)
    STRUCTURAL_DIFFERENTIAL = "structural_differential"  # High determinism (0.4)
    FUZZY_MATCH = "fuzzy_match"  # Medium determinism (0.3)
    PATTERN_MATCH = "pattern_match"  # Lower determinism (0.2)
    BEHAVIORAL_ONLY = "behavioral_only"  # Lowest determinism (0.1)


class AIAvailabilityStatus(str, Enum):
    """AI model availability status."""
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class AuthorityLevel(str, Enum):
    """
    Authority hierarchy levels - IMMUTABLE per VORTEX_CORE_AUTHORITY.md
    Order: SYSTEM_VERIFICATION > HUMAN_EXPERT > AI_ADVISORY > HEURISTIC
    """
    SYSTEM_VERIFICATION = "SYSTEM_VERIFICATION"  # Level 1 - Highest
    HUMAN_EXPERT = "HUMAN_EXPERT"  # Level 2 - Second
    AI_ADVISORY = "AI_ADVISORY"  # Level 3 - Third (NEVER authoritative)
    HEURISTIC = "HEURISTIC"  # Level 4 - Lowest
    NONE = "NONE"  # No authority


class ScanMode(str, Enum):
    """Scanning mode configuration."""
    PASSIVE = "passive"  # Read-only, no mutations
    ACTIVE = "active"  # Standard security testing
    AGGRESSIVE = "aggressive"  # Comprehensive testing


class ScanStatus(str, Enum):
    """Scan execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class QueuePriority(IntEnum):
    """Queue priority levels for workflow management."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "HEALTHY"
    ATTENTION = "ATTENTION"  # Warnings present
    DEGRADED = "DEGRADED"  # Critical issues present
    CRITICAL = "CRITICAL"  # Multiple critical issues


class ManualReviewPriority(IntEnum):
    """
    Manual review priority levels
    Lower number = higher priority (1 is highest)
    """
    CRITICAL = 1  # Immediate attention required
    HIGH = 2  # Review within 24h
    MEDIUM = 3  # Standard review
    LOW = 4  # Low priority
    BACKLOG = 5  # Can be deferred


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComplianceStatus(str, Enum):
    """Legal compliance status."""
    COMPLIANT = "COMPLIANT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    VIOLATION = "VIOLATION"
    UNKNOWN = "UNKNOWN"


class ReportFormat(str, Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    XML = "xml"