"""
VORTEX Validation Utilities - V17.0 ULTIMATE
Validation functions for URLs, domains, parameters, payloads, and scope
"""

import re
import logging
from typing import Optional, List
from urllib.parse import urlparse

from config.legal_config import is_in_scope as legal_is_in_scope

logger = logging.getLogger(__name__)


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return False


def validate_domain(domain: str) -> bool:
    """
    Validate domain format.
    
    Args:
        domain: Domain to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Basic domain validation
        domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        return bool(re.match(domain_pattern, domain))
    except Exception as e:
        logger.error(f"Domain validation error: {e}")
        return False


def validate_parameter(param_name: str) -> bool:
    """
    Validate parameter name (alphanumeric + underscore).
    
    Args:
        param_name: Parameter name to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        param_pattern = r'^[a-zA-Z0-9_\-]+$'
        return bool(re.match(param_pattern, param_name))
    except Exception as e:
        logger.error(f"Parameter validation error: {e}")
        return False


def is_safe_payload(payload: str, max_length: int = 10000) -> bool:
    """
    Check if payload is safe (not destructive).
    
    Args:
        payload: Payload to check
        max_length: Maximum allowed length
        
    Returns:
        True if safe, False otherwise
    """
    try:
        # Check length
        if len(payload) > max_length:
            logger.warning(f"Payload too long: {len(payload)} > {max_length}")
            return False
        
        # Check for destructive SQL keywords
        destructive_sql = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE']
        payload_upper = payload.upper()
        for keyword in destructive_sql:
            if keyword in payload_upper:
                logger.warning(f"Destructive SQL keyword detected: {keyword}")
                return False
        
        # Check for file write attempts
        file_write_patterns = ['INTO OUTFILE', 'INTO DUMPFILE', 'LOAD_FILE']
        for pattern in file_write_patterns:
            if pattern in payload_upper:
                logger.warning(f"File write pattern detected: {pattern}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Payload safety check error: {e}")
        return False


def check_scope(url: str, authorized_domains: List[str]) -> bool:
    """
    Check if URL is within authorized scope.
    
    Args:
        url: URL to check
        authorized_domains: List of authorized domains
        
    Returns:
        True if in scope, False otherwise
    """
    try:
        return legal_is_in_scope(url, authorized_domains)
    except Exception as e:
        logger.error(f"Scope check error: {e}")
        return False


def validate_payload_size(payload: str, max_size: int = 10000) -> bool:
    """
    Validate payload size.
    
    Args:
        payload: Payload to check
        max_size: Maximum size in bytes
        
    Returns:
        True if within limits, False otherwise
    """
    try:
        return len(payload.encode('utf-8')) <= max_size
    except Exception as e:
        logger.error(f"Payload size validation error: {e}")
        return False


__all__ = [
    'validate_url',
    'validate_domain',
    'validate_parameter',
    'is_safe_payload',
    'check_scope',
    'validate_payload_size',
]