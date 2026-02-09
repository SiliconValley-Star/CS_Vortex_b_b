"""
VORTEX Compliance Domain - V17.0 ULTIMATE
Legal compliance and ethical boundary enforcement
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import re


class ComplianceViolationType(str, Enum):
    """Types of compliance violations."""
    UNAUTHORIZED_TARGET = "unauthorized_target"
    PII_DETECTED = "pii_detected"
    SCOPE_VIOLATION = "scope_violation"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    MALICIOUS_PAYLOAD = "malicious_payload"
    DATA_RETENTION_VIOLATION = "data_retention_violation"


class PIIType(str, Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"


@dataclass
class PIIPattern:
    """PII detection pattern."""
    pii_type: PIIType
    pattern: str
    description: str
    sensitivity_level: str  # LOW, MEDIUM, HIGH, CRITICAL


# PII Detection Patterns
PII_PATTERNS = [
    PIIPattern(
        pii_type=PIIType.EMAIL,
        pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        description="Email address",
        sensitivity_level="MEDIUM"
    ),
    PIIPattern(
        pii_type=PIIType.SSN,
        pattern=r'\b\d{3}-\d{2}-\d{4}\b',
        description="US Social Security Number",
        sensitivity_level="CRITICAL"
    ),
    PIIPattern(
        pii_type=PIIType.CREDIT_CARD,
        pattern=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        description="Credit card number",
        sensitivity_level="CRITICAL"
    ),
    PIIPattern(
        pii_type=PIIType.PHONE,
        pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        description="US phone number",
        sensitivity_level="MEDIUM"
    ),
    PIIPattern(
        pii_type=PIIType.IP_ADDRESS,
        pattern=r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        description="IPv4 address",
        sensitivity_level="LOW"
    ),
]


@dataclass
class PIIDetectionResult:
    """Result of PII detection scan."""
    pii_detected: bool
    pii_types: List[PIIType] = field(default_factory=list)
    matches: List[Dict[str, Any]] = field(default_factory=list)
    redacted_content: str = ""
    
    def add_match(self, pii_type: PIIType, match: str, position: int, sensitivity: str):
        """Add a PII match."""
        self.pii_detected = True
        if pii_type not in self.pii_types:
            self.pii_types.append(pii_type)
        
        self.matches.append({
            'type': pii_type.value,
            'match': match,
            'position': position,
            'sensitivity': sensitivity
        })


@dataclass
class ScopeValidationResult:
    """Result of scope validation."""
    is_authorized: bool
    reason: str
    domain: str
    matched_rule: Optional[str] = None
    
    # Validation details
    in_authorized_list: bool = False
    matches_wildcard: bool = False
    subdomain_allowed: bool = False


@dataclass
class ComplianceCheck:
    """Single compliance check result."""
    check_type: str
    passed: bool
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceReport:
    """Complete compliance validation report."""
    target_url: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Overall status
    compliant: bool = True
    violations: List[ComplianceViolationType] = field(default_factory=list)
    
    # Individual checks
    checks: List[ComplianceCheck] = field(default_factory=list)
    
    # Specific results
    scope_validation: Optional[ScopeValidationResult] = None
    pii_detection: Optional[PIIDetectionResult] = None
    
    # Actions taken
    actions_required: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    
    def add_check(self, check: ComplianceCheck):
        """Add a compliance check."""
        self.checks.append(check)
        if not check.passed and check.severity in ["WARNING", "CRITICAL"]:
            self.compliant = False
    
    def add_violation(self, violation_type: ComplianceViolationType):
        """Add a compliance violation."""
        if violation_type not in self.violations:
            self.violations.append(violation_type)
        self.compliant = False
    
    def require_action(self, action: str):
        """Add required action."""
        if action not in self.actions_required:
            self.actions_required.append(action)


@dataclass
class LegalContact:
    """Legal contact information."""
    name: str
    email: str
    organization: str
    role: str = "Security Contact"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'email': self.email,
            'organization': self.organization,
            'role': self.role
        }


@dataclass
class ResponsibleDisclosurePolicy:
    """Responsible disclosure policy details."""
    organization: str
    policy_url: str
    contact_email: str
    
    # Response expectations
    initial_response_days: int = 3
    resolution_days: int = 90
    
    # Allowed actions
    allows_automated_scanning: bool = True
    requires_notification: bool = True
    prohibits_data_exfiltration: bool = True
    
    # Safe harbor provisions
    has_safe_harbor: bool = True
    safe_harbor_conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'organization': self.organization,
            'policy_url': self.policy_url,
            'contact_email': self.contact_email,
            'initial_response_days': self.initial_response_days,
            'resolution_days': self.resolution_days,
            'allows_automated_scanning': self.allows_automated_scanning,
            'has_safe_harbor': self.has_safe_harbor
        }


@dataclass
class DataRetentionPolicy:
    """Data retention and deletion policy."""
    # Retention periods (in days)
    finding_data_retention: int = 90
    evidence_retention: int = 365
    log_retention: int = 30
    pii_retention: int = 7  # Minimal retention for PII
    
    # Auto-deletion settings
    auto_delete_enabled: bool = True
    delete_after_submission: bool = False  # Keep for tracking
    
    # Backup policy
    backup_retention: int = 30
    encrypted_backups: bool = True
    
    def is_expired(self, data_type: str, created_at: datetime) -> bool:
        """Check if data is expired based on retention policy."""
        age_days = (datetime.utcnow() - created_at).days
        
        retention_map = {
            'finding': self.finding_data_retention,
            'evidence': self.evidence_retention,
            'log': self.log_retention,
            'pii': self.pii_retention,
            'backup': self.backup_retention
        }
        
        retention_days = retention_map.get(data_type, self.finding_data_retention)
        return age_days > retention_days


@dataclass
class EthicalBoundary:
    """Ethical boundary definition."""
    boundary_type: str
    description: str
    is_hard_limit: bool = True  # Hard limits cannot be overridden
    
    # Violation handling
    violation_action: str = "BLOCK"  # BLOCK, WARN, LOG
    requires_manual_review: bool = True


# Ethical Boundaries per .clinerules
ETHICAL_BOUNDARIES = [
    EthicalBoundary(
        boundary_type="NO_MALICIOUS_PAYLOADS",
        description="No payloads designed to cause harm, damage, or data loss",
        is_hard_limit=True,
        violation_action="BLOCK"
    ),
    EthicalBoundary(
        boundary_type="NO_DATA_EXFILTRATION",
        description="No extraction or exfiltration of sensitive data",
        is_hard_limit=True,
        violation_action="BLOCK"
    ),
    EthicalBoundary(
        boundary_type="NO_DDOS",
        description="No denial of service attacks or excessive request rates",
        is_hard_limit=True,
        violation_action="BLOCK"
    ),
    EthicalBoundary(
        boundary_type="RESPECT_ROBOTS_TXT",
        description="Respect robots.txt and crawler directives",
        is_hard_limit=False,
        violation_action="WARN"
    ),
    EthicalBoundary(
        boundary_type="NO_SOCIAL_ENGINEERING",
        description="No social engineering or phishing attempts",
        is_hard_limit=True,
        violation_action="BLOCK"
    ),
]


def detect_pii(content: str, redact: bool = True) -> PIIDetectionResult:
    """
    Detect PII in content and optionally redact it
    Per VORTEX legal compliance requirements
    """
    result = PIIDetectionResult(pii_detected=False, redacted_content=content)
    
    for pii_pattern in PII_PATTERNS:
        pattern = re.compile(pii_pattern.pattern)
        matches = pattern.finditer(content)
        
        for match in matches:
            result.add_match(
                pii_type=pii_pattern.pii_type,
                match=match.group(),
                position=match.start(),
                sensitivity=pii_pattern.sensitivity_level
            )
            
            # Redact if requested
            if redact:
                redacted_value = f"[REDACTED_{pii_pattern.pii_type.value.upper()}]"
                result.redacted_content = result.redacted_content.replace(
                    match.group(), 
                    redacted_value
                )
    
    return result


def validate_ethical_boundaries(payload: str, action: str) -> List[EthicalBoundary]:
    """
    Validate payload/action against ethical boundaries
    Returns list of violated boundaries
    """
    violations = []
    
    payload_lower = payload.lower()
    
    # Check for malicious indicators
    malicious_keywords = [
        'rm -rf', 'format c:', 'drop table', 'delete from',
        'system(', 'exec(', 'eval(', 'shell_exec'
    ]
    if any(keyword in payload_lower for keyword in malicious_keywords):
        violations.append(ETHICAL_BOUNDARIES[0])  # NO_MALICIOUS_PAYLOADS
    
    # Check for data exfiltration attempts
    exfil_keywords = ['curl', 'wget', 'http://', 'https://', 'ftp://']
    if action == "execute" and any(keyword in payload_lower for keyword in exfil_keywords):
        violations.append(ETHICAL_BOUNDARIES[1])  # NO_DATA_EXFILTRATION
    
    return violations