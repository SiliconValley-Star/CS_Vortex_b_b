"""
VORTEX Parsing Utilities - V17.0 ULTIMATE
Safe parsing functions for JSON, XML, URLs, and parameters
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


def safe_json_parse(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with error handling.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return default if default is not None else {}
    except Exception as e:
        logger.error(f"Unexpected JSON parse error: {e}")
        return default if default is not None else {}


def safe_xml_parse(text: str) -> Optional[Dict]:
    """
    Safely parse XML (basic implementation).
    
    Args:
        text: XML string to parse
        
    Returns:
        Parsed XML as dict or None
    """
    try:
        # Basic XML parsing - can be enhanced with lxml if needed
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
        return {'tag': root.tag, 'text': root.text, 'attrib': root.attrib}
    except Exception as e:
        logger.error(f"XML parse error: {e}")
        return None


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text containing URLs
        
    Returns:
        List of extracted URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    try:
        return re.findall(url_pattern, text)
    except Exception as e:
        logger.error(f"URL extraction error: {e}")
        return []


def extract_parameters(url: str) -> Dict[str, List[str]]:
    """
    Extract query parameters from URL.
    
    Args:
        url: URL to parse
        
    Returns:
        Dictionary of parameters
    """
    try:
        parsed = urlparse(url)
        return parse_qs(parsed.query)
    except Exception as e:
        logger.error(f"Parameter extraction error: {e}")
        return {}


def parse_headers(headers_text: str) -> Dict[str, str]:
    """
    Parse HTTP headers from text.
    
    Args:
        headers_text: Headers as text
        
    Returns:
        Dictionary of headers
    """
    headers = {}
    try:
        for line in headers_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Header parsing error: {e}")
    return headers


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.
    
    Args:
        url: URL to parse
        
    Returns:
        Domain name or None
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception as e:
        logger.error(f"Domain extraction error: {e}")
        return None


def extract_path(url: str) -> Optional[str]:
    """
    Extract path from URL.
    
    Args:
        url: URL to parse
        
    Returns:
        Path or None
    """
    try:
        parsed = urlparse(url)
        return parsed.path
    except Exception as e:
        logger.error(f"Path extraction error: {e}")
        return None


__all__ = [
    'safe_json_parse',
    'safe_xml_parse',
    'extract_urls',
    'extract_parameters',
    'parse_headers',
    'extract_domain',
    'extract_path',
]