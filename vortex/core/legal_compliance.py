"""
VORTEX Legal Compliance Engine - V17.0 ULTIMATE
Legal compliance validation and monitoring

RESPONSIBILITIES:
- Target authorization validation
- Scope compliance checking
- PII detection and redaction
- Legal disclaimers
- Evidence retention policies
"""

import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class LegalGuardian:
    """
    Legal compliance guardian for target authorization and scope validation.
    Ensures all scanning activities comply with legal and ethical requirements.
    """
    
    def __init__(self, legal_config):
        self.legal_config = legal_config
        self.authorized_domains = set(legal_config.authorized_domains)
        self.pii_patterns = self._initialize_pii_patterns()
        
        # Statistics
        self.stats = {
            'targets_validated': 0,
            'targets_authorized': 0,
            'targets_rejected': 0,
            'pii_detections': 0
        }
    
    def _initialize_pii_patterns(self) -> List[re.Pattern]:
        """Initialize PII detection patterns."""
        return [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b\d{16}\b'),  # Credit card (simple)
            re.compile(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'),  # Phone
        ]
    
    async def validate_target_authorization(self, target_url: str) -> bool:
        """
        Validate if target is authorized for scanning.
        
        Args:
            target_url: Target URL to validate
            
        Returns:
            True if authorized, False otherwise
        """
        self.stats['targets_validated'] += 1
        
        try:
            parsed = urlparse(target_url)
            domain = parsed.netloc
            
            # Check if domain is in authorized list
            if not self.authorized_domains:
                logger.warning("No authorized domains configured - rejecting all targets")
                self.stats['targets_rejected'] += 1
                return False
            
            # Check exact match or subdomain match
            is_authorized = any(
                domain == auth_domain or domain.endswith('.' + auth_domain)
                for auth_domain in self.authorized_domains
            )
            
            if is_authorized:
                self.stats['targets_authorized'] += 1
                logger.info(f"Target authorized: {target_url}")
            else:
                self.stats['targets_rejected'] += 1
                logger.warning(f"Target NOT authorized: {target_url}")
            
            return is_authorized
            
        except Exception as e:
            logger.error(f"Target validation error: {e}")
            self.stats['targets_rejected'] += 1
            return False
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of PII detections
        """
        if not self.legal_config.pii_detection_enabled:
            return []
        
        detections = []
        
        for pattern in self.pii_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                detections.append({
                    'type': 'PII',
                    'value': match.group(),
                    'position': match.span(),
                    'pattern': pattern.pattern
                })
                self.stats['pii_detections'] += 1
        
        return detections
    
    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Redacted text
        """
        if not self.legal_config.pii_redaction_enabled:
            return text
        
        redacted = text
        for pattern in self.pii_patterns:
            redacted = pattern.sub('[REDACTED]', redacted)
        
        return redacted
    
    def get_stats(self) -> Dict[str, int]:
        """Get compliance statistics."""
        return self.stats.copy()


class LegalComplianceEngine:
    """
    Legal compliance engine for web interface.
    Provides compliance validation and monitoring.
    """
    
    def __init__(self):
        self.authorized_domains = set()
        self.compliance_checks_performed = 0
        self.violations_detected = 0
    
    def validate_scope(self, target: str) -> bool:
        """
        Validate if target is within authorized scope.
        
        Args:
            target: Target URL or domain
            
        Returns:
            True if in scope, False otherwise
        """
        self.compliance_checks_performed += 1
        
        try:
            parsed = urlparse(target if target.startswith('http') else f'http://{target}')
            domain = parsed.netloc
            
            if not self.authorized_domains:
                logger.warning("No authorized domains configured")
                self.violations_detected += 1
                return False
            
            is_valid = any(
                domain == auth_domain or domain.endswith('.' + auth_domain)
                for auth_domain in self.authorized_domains
            )
            
            if not is_valid:
                self.violations_detected += 1
                logger.warning(f"Scope violation: {target}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Scope validation error: {e}")
            self.violations_detected += 1
            return False
    
    def configure_authorized_domains(self, domains: List[str]):
        """Configure authorized domains."""
        self.authorized_domains = set(domains)
        logger.info(f"Configured {len(domains)} authorized domains")
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status summary."""
        return {
            'checks_performed': self.compliance_checks_performed,
            'violations_detected': self.violations_detected,
            'authorized_domains_count': len(self.authorized_domains),
            'compliance_rate': (
                1.0 - (self.violations_detected / max(self.compliance_checks_performed, 1))
                if self.compliance_checks_performed > 0 else 1.0
            )
        }