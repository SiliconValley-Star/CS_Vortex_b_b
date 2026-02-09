"""
VORTEX Legal Compliance Configuration - V17.0 ULTIMATE
Complete legal compliance settings per .clinerules specifications
"""

from typing import List, Dict, Any, Set
from dataclasses import dataclass, field


@dataclass
class PII_PATTERNS:
    """
    PII (Personally Identifiable Information) detection patterns
    Per .clinerules: PII detection and redaction required for GDPR compliance
    """
    # Email patterns
    email: str = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Phone numbers (international formats)
    phone_us: str = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phone_intl: str = r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    
    # Social Security Numbers (US)
    ssn: str = r'\b\d{3}-\d{2}-\d{4}\b'
    
    # Credit card numbers
    credit_card: str = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    
    # IP addresses (can be PII in some contexts)
    ipv4: str = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ipv6: str = r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
    
    # Names (basic pattern - may need enhancement)
    full_name: str = r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
    
    # Addresses
    address: str = r'\b\d{1,5}\s[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b'
    
    # API keys and tokens
    api_key: str = r'(?i)(?:api[_-]?key|token|secret)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})'
    
    # Session IDs and cookies
    session_id: str = r'(?i)(?:session[_-]?id|sess|jsessionid)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})'


@dataclass
class EthicalBoundaries:
    """
    Ethical boundaries for security testing per .clinerules
    Defines what is NEVER allowed regardless of target authorization
    """
    # Prohibited actions (NEVER allowed)
    prohibited_actions: List[str] = field(default_factory=lambda: [
        'data_destruction',           # Deleting/modifying data
        'service_disruption',         # DOS/DDOS attacks
        'data_exfiltration',          # Stealing actual data
        'privilege_escalation_abuse', # Abusing escalated privileges
        'lateral_movement',           # Moving to other systems
        'persistence_mechanisms',     # Installing backdoors
        'social_engineering',         # Manipulating people
        'physical_access',           # Physical security testing
    ])
    
    # Dangerous payload types (require special authorization)
    dangerous_payloads: List[str] = field(default_factory=lambda: [
        'destructive_sql',    # DROP, DELETE, TRUNCATE
        'file_write',         # Writing files to disk
        'command_execution',  # OS command execution
        'code_injection',     # Arbitrary code execution
    ])
    
    # Prohibited targets (NEVER test without explicit authorization)
    prohibited_target_types: List[str] = field(default_factory=lambda: [
        'critical_infrastructure',  # Power, water, healthcare
        'financial_systems',        # Banking, payment processing
        'government_systems',       # Government websites/services
        'educational_systems',      # Schools, universities
        'healthcare_systems',       # Hospitals, medical records
    ])
    
    # Maximum impact thresholds
    max_requests_per_minute: int = 60      # Rate limiting to avoid DOS
    max_concurrent_scans: int = 3          # Prevent resource exhaustion
    max_payload_size: int = 10000          # 10KB max payload
    max_test_duration_hours: int = 24      # Auto-stop after 24h


@dataclass
class ScopeValidation:
    """
    Scope validation rules per .clinerules
    Ensures testing stays within authorized boundaries
    """
    # Domain validation
    require_explicit_authorization: bool = True
    allow_subdomain_enumeration: bool = False  # Requires explicit permission
    allow_ip_range_scanning: bool = False      # Requires explicit permission
    
    # Out-of-scope indicators (automatically block)
    out_of_scope_keywords: List[str] = field(default_factory=lambda: [
        'admin',
        'internal',
        'staging',
        'dev',
        'test',
        'prod',
        'backup',
        'api-internal',
        'private',
    ])
    
    # Require confirmation for sensitive paths
    sensitive_paths: List[str] = field(default_factory=lambda: [
        '/admin',
        '/administrator',
        '/wp-admin',
        '/phpmyadmin',
        '/backup',
        '/api/internal',
        '/.git',
        '/.env',
    ])


@dataclass
class ResponsibleDisclosure:
    """
    Responsible disclosure policy per .clinerules
    Defines how vulnerabilities should be reported
    """
    # Contact information
    security_contact_email: str = ""
    security_contact_url: str = ""
    
    # Disclosure timeline (days)
    initial_disclosure_delay: int = 0      # Report immediately upon confirmation
    public_disclosure_delay: int = 90      # 90 days before public disclosure
    vendor_response_timeout: int = 30      # 30 days for vendor initial response
    
    # Vulnerability severity escalation
    critical_vulnerability_escalation: int = 7   # 7 days for critical findings
    high_vulnerability_escalation: int = 14      # 14 days for high findings
    
    # Required disclosure elements
    required_disclosure_elements: List[str] = field(default_factory=lambda: [
        'vulnerability_description',
        'affected_component',
        'impact_assessment',
        'proof_of_concept',
        'remediation_guidance',
        'discovery_timeline',
    ])


@dataclass
class DataRetention:
    """
    Data retention policies per .clinerules
    GDPR and privacy compliance requirements
    """
    # Retention periods (days)
    finding_data_retention: int = 90       # Standard findings
    evidence_retention: int = 365          # Evidence with cryptographic integrity
    logs_retention: int = 30               # Application logs
    pii_retention: int = 7                 # Minimal PII retention
    backup_retention: int = 30             # Database backups
    audit_trail_retention: int = 365       # Complete audit trail
    
    # Auto-cleanup thresholds
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24       # Daily cleanup
    
    # GDPR compliance
    gdpr_compliance_mode: bool = True
    right_to_erasure_enabled: bool = True
    data_portability_enabled: bool = True


@dataclass
class LegalComplianceConfig:
    """
    Complete legal compliance configuration
    Main container for all legal/ethical settings
    """
    # PII Detection & Redaction
    pii_detection_enabled: bool = True
    pii_redaction_enabled: bool = True
    pii_patterns: PII_PATTERNS = field(default_factory=PII_PATTERNS)
    
    # Ethical Boundaries
    ethical_boundaries: EthicalBoundaries = field(default_factory=EthicalBoundaries)
    
    # Scope Validation
    scope_validation: ScopeValidation = field(default_factory=ScopeValidation)
    
    # Responsible Disclosure
    responsible_disclosure: ResponsibleDisclosure = field(default_factory=ResponsibleDisclosure)
    
    # Data Retention
    data_retention: DataRetention = field(default_factory=DataRetention)
    
    # Legal Disclaimers
    legal_disclaimer_required: bool = True
    terms_of_service_url: str = ""
    privacy_policy_url: str = ""
    
    # Authorization Tracking
    require_written_authorization: bool = True
    authorization_expiry_days: int = 90
    
    # Legal Contact
    legal_contact_email: str = ""
    legal_contact_phone: str = ""


# Global legal compliance configuration instance
LEGAL_COMPLIANCE = LegalComplianceConfig()


# Utility functions for legal compliance checks
def is_pii_present(text: str) -> bool:
    """Check if text contains PII."""
    import re
    
    if not LEGAL_COMPLIANCE.pii_detection_enabled:
        return False
    
    patterns = LEGAL_COMPLIANCE.pii_patterns
    
    # Check each PII pattern
    pii_checks = [
        re.search(patterns.email, text),
        re.search(patterns.phone_us, text),
        re.search(patterns.phone_intl, text),
        re.search(patterns.ssn, text),
        re.search(patterns.credit_card, text),
        re.search(patterns.api_key, text),
        re.search(patterns.session_id, text),
    ]
    
    return any(pii_checks)


def redact_pii(text: str) -> str:
    """Redact PII from text."""
    import re
    
    if not LEGAL_COMPLIANCE.pii_redaction_enabled:
        return text
    
    patterns = LEGAL_COMPLIANCE.pii_patterns
    
    # Redact each PII pattern
    text = re.sub(patterns.email, '[EMAIL_REDACTED]', text)
    text = re.sub(patterns.phone_us, '[PHONE_REDACTED]', text)
    text = re.sub(patterns.phone_intl, '[PHONE_REDACTED]', text)
    text = re.sub(patterns.ssn, '[SSN_REDACTED]', text)
    text = re.sub(patterns.credit_card, '[CARD_REDACTED]', text)
    text = re.sub(patterns.api_key, r'\1=[API_KEY_REDACTED]', text)
    text = re.sub(patterns.session_id, r'\1=[SESSION_REDACTED]', text)
    
    return text


def is_action_permitted(action: str) -> bool:
    """Check if action is ethically permitted."""
    boundaries = LEGAL_COMPLIANCE.ethical_boundaries
    return action not in boundaries.prohibited_actions


def is_payload_safe(payload_type: str) -> bool:
    """Check if payload type is safe to use."""
    boundaries = LEGAL_COMPLIANCE.ethical_boundaries
    return payload_type not in boundaries.dangerous_payloads


def is_in_scope(url: str, authorized_domains: List[str]) -> bool:
    """Check if URL is within authorized scope."""
    from urllib.parse import urlparse
    
    scope = LEGAL_COMPLIANCE.scope_validation
    
    # Parse URL
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # Check if domain is authorized
    domain_authorized = any(
        domain == auth_domain or domain.endswith(f'.{auth_domain}')
        for auth_domain in authorized_domains
    )
    
    if not domain_authorized:
        return False
    
    # Check out-of-scope keywords
    for keyword in scope.out_of_scope_keywords:
        if keyword in domain or keyword in path:
            return False
    
    # Check sensitive paths (require explicit confirmation)
    for sensitive_path in scope.sensitive_paths:
        if path.startswith(sensitive_path):
            # Would require explicit confirmation in interactive mode
            return False
    
    return True


__all__ = [
    'LegalComplianceConfig',
    'PII_PATTERNS',
    'EthicalBoundaries',
    'ScopeValidation',
    'ResponsibleDisclosure',
    'DataRetention',
    'LEGAL_COMPLIANCE',
    'is_pii_present',
    'redact_pii',
    'is_action_permitted',
    'is_payload_safe',
    'is_in_scope',
]