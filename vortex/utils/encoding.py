"""
VORTEX Encoding Utilities - V17.0 ULTIMATE
Safe encoding/decoding functions for URL, Base64, HTML, etc.
"""

import base64
import urllib.parse
import html
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def url_encode(text: str, safe: str = '') -> str:
    """
    URL encode text.
    
    Args:
        text: Text to encode
        safe: Characters that should not be encoded
        
    Returns:
        URL encoded string
    """
    try:
        return urllib.parse.quote(text, safe=safe)
    except Exception as e:
        logger.error(f"URL encode error: {e}")
        return text


def url_decode(text: str) -> str:
    """
    URL decode text.
    
    Args:
        text: Text to decode
        
    Returns:
        URL decoded string
    """
    try:
        return urllib.parse.unquote(text)
    except Exception as e:
        logger.error(f"URL decode error: {e}")
        return text


def base64_encode(text: Union[str, bytes]) -> str:
    """
    Base64 encode text or bytes.
    
    Args:
        text: Text or bytes to encode
        
    Returns:
        Base64 encoded string
    """
    try:
        if isinstance(text, str):
            text = text.encode('utf-8')
        return base64.b64encode(text).decode('utf-8')
    except Exception as e:
        logger.error(f"Base64 encode error: {e}")
        return ""


def base64_decode(text: str) -> str:
    """
    Base64 decode text.
    
    Args:
        text: Base64 encoded text
        
    Returns:
        Decoded string
    """
    try:
        return base64.b64decode(text).decode('utf-8')
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return ""


def html_encode(text: str) -> str:
    """
    HTML encode text (escape special characters).
    
    Args:
        text: Text to encode
        
    Returns:
        HTML encoded string
    """
    try:
        return html.escape(text)
    except Exception as e:
        logger.error(f"HTML encode error: {e}")
        return text


def html_decode(text: str) -> str:
    """
    HTML decode text (unescape special characters).
    
    Args:
        text: Text to decode
        
    Returns:
        HTML decoded string
    """
    try:
        return html.unescape(text)
    except Exception as e:
        logger.error(f"HTML decode error: {e}")
        return text


def safe_encode(text: str, encoding: str = 'utf-8', errors: str = 'replace') -> bytes:
    """
    Safely encode text to bytes with error handling.
    
    Args:
        text: Text to encode
        encoding: Target encoding (default: utf-8)
        errors: Error handling strategy (default: replace)
        
    Returns:
        Encoded bytes
    """
    try:
        return text.encode(encoding, errors=errors)
    except Exception as e:
        logger.error(f"Safe encode error: {e}")
        return b''


def safe_decode(data: bytes, encoding: str = 'utf-8', errors: str = 'replace') -> str:
    """
    Safely decode bytes to text with error handling.
    
    Args:
        data: Bytes to decode
        encoding: Source encoding (default: utf-8)
        errors: Error handling strategy (default: replace)
        
    Returns:
        Decoded string
    """
    try:
        return data.decode(encoding, errors=errors)
    except Exception as e:
        logger.error(f"Safe decode error: {e}")
        return ""


def hex_encode(text: Union[str, bytes]) -> str:
    """
    Hex encode text or bytes.
    
    Args:
        text: Text or bytes to encode
        
    Returns:
        Hex encoded string
    """
    try:
        if isinstance(text, str):
            text = text.encode('utf-8')
        return text.hex()
    except Exception as e:
        logger.error(f"Hex encode error: {e}")
        return ""


def hex_decode(text: str) -> str:
    """
    Hex decode text.
    
    Args:
        text: Hex encoded text
        
    Returns:
        Decoded string
    """
    try:
        return bytes.fromhex(text).decode('utf-8')
    except Exception as e:
        logger.error(f"Hex decode error: {e}")
        return ""


__all__ = [
    'url_encode',
    'url_decode',
    'base64_encode',
    'base64_decode',
    'html_encode',
    'html_decode',
    'safe_encode',
    'safe_decode',
    'hex_encode',
    'hex_decode',
]