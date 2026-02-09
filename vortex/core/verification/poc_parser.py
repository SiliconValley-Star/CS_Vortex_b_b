"""
VORTEX PoC Parser - V17.0 ULTIMATE
Parse Proof-of-Concept (PoC) from various formats

SUPPORTED FORMATS:
- cURL commands
- Raw HTTP requests
- Python requests code
- Generic text format (auto-detection)

CRITICAL: Only parse AI-generated PoCs, never heuristic-only PoCs
"""

import re
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from urllib.parse import urlparse, parse_qs, unquote

logger = logging.getLogger(__name__)


@dataclass
class ParsedPoC:
    """Parsed PoC with all components."""
    method: str  # GET, POST, etc.
    url: str
    headers: Dict[str, str]
    body: Optional[str] = None
    parameters: Dict[str, str] = None
    format_detected: str = "unknown"
    confidence: float = 0.0


class PoCParser:
    """
    Parse PoC from various formats.
    
    Supports:
    - cURL commands (most common)
    - Raw HTTP requests
    - Python requests library code
    - Generic format detection
    """
    
    def __init__(self):
        self.stats = {
            'parsed_curl': 0,
            'parsed_http': 0,
            'parsed_python': 0,
            'parse_errors': 0
        }
    
    def parse(self, poc_text: str) -> Optional[ParsedPoC]:
        """
        Auto-detect format and parse PoC.
        
        Args:
            poc_text: PoC text in any supported format
            
        Returns:
            ParsedPoC object or None if parsing failed
        """
        if not poc_text or not poc_text.strip():
            logger.error("Empty PoC text provided")
            return None
        
        poc_text = poc_text.strip()
        
        # Try formats in order of likelihood
        if 'curl' in poc_text.lower() or poc_text.startswith('curl '):
            return self.parse_curl(poc_text)
        
        elif poc_text.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'PATCH ', 'HEAD ', 'OPTIONS ')):
            return self.parse_http_raw(poc_text)
        
        elif 'requests.' in poc_text or 'import requests' in poc_text:
            return self.parse_python(poc_text)
        
        else:
            # Try generic parsing
            return self.parse_generic(poc_text)
    
    def parse_curl(self, curl_command: str) -> Optional[ParsedPoC]:
        """
        Parse cURL command to PoC.
        
        Example:
            curl -X POST https://example.com/api -H "Content-Type: application/json" -d '{"key":"value"}'
        """
        self.stats['parsed_curl'] += 1
        logger.debug(f"Parsing cURL command: {curl_command[:100]}...")
        
        try:
            # Extract URL
            url_match = re.search(r'["\']?(https?://[^\s"\']+)["\']?', curl_command)
            if not url_match:
                logger.error("No URL found in cURL command")
                return None
            
            url = url_match.group(1).strip('"\'')
            
            # Extract method (default GET)
            method = 'GET'
            method_match = re.search(r'-X\s+(\w+)', curl_command, re.IGNORECASE)
            if method_match:
                method = method_match.group(1).upper()
            
            # Extract headers
            headers = {}
            header_matches = re.finditer(r'-H\s+["\']([^:]+):\s*([^"\']+)["\']', curl_command, re.IGNORECASE)
            for match in header_matches:
                header_name = match.group(1).strip()
                header_value = match.group(2).strip()
                headers[header_name] = header_value
            
            # Extract body/data
            body = None
            
            # Try -d/--data
            data_match = re.search(r'(?:-d|--data)\s+["\'](.+?)["\']', curl_command, re.IGNORECASE)
            if data_match:
                body = data_match.group(1)
            
            # Try --data-binary
            data_binary_match = re.search(r'--data-binary\s+["\'](.+?)["\']', curl_command, re.IGNORECASE)
            if data_binary_match:
                body = data_binary_match.group(1)
            
            # Try --data-raw
            data_raw_match = re.search(r'--data-raw\s+["\'](.+?)["\']', curl_command, re.IGNORECASE)
            if data_raw_match:
                body = data_raw_match.group(1)
            
            # Extract parameters from URL
            parameters = self._extract_url_parameters(url)
            
            return ParsedPoC(
                method=method,
                url=url,
                headers=headers,
                body=body,
                parameters=parameters,
                format_detected='curl',
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"cURL parsing error: {e}")
            self.stats['parse_errors'] += 1
            return None
    
    def parse_http_raw(self, http_text: str) -> Optional[ParsedPoC]:
        """
        Parse raw HTTP request.
        
        Example:
            POST /api/endpoint HTTP/1.1
            Host: example.com
            Content-Type: application/json
            
            {"key":"value"}
        """
        self.stats['parsed_http'] += 1
        logger.debug(f"Parsing raw HTTP request: {http_text[:100]}...")
        
        try:
            lines = http_text.strip().split('\n')
            
            # Parse request line
            request_line = lines[0].strip()
            parts = request_line.split()
            
            if len(parts) < 2:
                logger.error("Invalid HTTP request line")
                return None
            
            method = parts[0].upper()
            path = parts[1]
            
            # Parse headers
            headers = {}
            body_start = len(lines)
            
            for i, line in enumerate(lines[1:], start=1):
                line = line.strip()
                
                if not line:
                    # Empty line indicates body start
                    body_start = i + 1
                    break
                
                if ':' in line:
                    header_name, header_value = line.split(':', 1)
                    headers[header_name.strip()] = header_value.strip()
            
            # Extract host
            host = headers.get('Host', '')
            
            # Construct full URL
            scheme = 'https' if '443' in host or 'https' in http_text.lower() else 'http'
            url = f"{scheme}://{host}{path}" if host else path
            
            # Parse body
            body = None
            if body_start < len(lines):
                body = '\n'.join(lines[body_start:]).strip()
            
            # Extract parameters
            parameters = self._extract_url_parameters(url)
            
            return ParsedPoC(
                method=method,
                url=url,
                headers=headers,
                body=body,
                parameters=parameters,
                format_detected='http_raw',
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"HTTP raw parsing error: {e}")
            self.stats['parse_errors'] += 1
            return None
    
    def parse_python(self, python_code: str) -> Optional[ParsedPoC]:
        """
        Parse Python requests code.
        
        Example:
            import requests
            response = requests.post('https://example.com/api', 
                                    json={'key': 'value'},
                                    headers={'Authorization': 'Bearer token'})
        """
        self.stats['parsed_python'] += 1
        logger.debug(f"Parsing Python code: {python_code[:100]}...")
        
        try:
            # Extract method
            method = 'GET'
            for http_method in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']:
                if f'requests.{http_method}(' in python_code.lower():
                    method = http_method.upper()
                    break
            
            # Extract URL
            url_match = re.search(r'["\']?(https?://[^\s"\')\],]+)["\']?', python_code)
            if not url_match:
                logger.error("No URL found in Python code")
                return None
            
            url = url_match.group(1).strip('"\'')
            
            # Extract headers
            headers = {}
            headers_match = re.search(r'headers\s*=\s*({[^}]+})', python_code, re.IGNORECASE)
            if headers_match:
                try:
                    # Try to parse as dict
                    headers_str = headers_match.group(1)
                    headers_str = headers_str.replace("'", '"')
                    headers = json.loads(headers_str)
                except Exception:
                    pass
            
            # Extract body/data
            body = None
            
            # Try json parameter
            json_match = re.search(r'json\s*=\s*({[^}]+})', python_code, re.IGNORECASE)
            if json_match:
                body = json_match.group(1)
            
            # Try data parameter
            data_match = re.search(r'data\s*=\s*["\'](.+?)["\']', python_code, re.IGNORECASE)
            if data_match:
                body = data_match.group(1)
            
            # Extract parameters
            parameters = self._extract_url_parameters(url)
            
            # Also check params parameter
            params_match = re.search(r'params\s*=\s*({[^}]+})', python_code, re.IGNORECASE)
            if params_match:
                try:
                    params_str = params_match.group(1)
                    params_str = params_str.replace("'", '"')
                    params = json.loads(params_str)
                    parameters.update(params)
                except Exception:
                    pass
            
            return ParsedPoC(
                method=method,
                url=url,
                headers=headers,
                body=body,
                parameters=parameters,
                format_detected='python',
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Python parsing error: {e}")
            self.stats['parse_errors'] += 1
            return None
    
    def parse_generic(self, text: str) -> Optional[ParsedPoC]:
        """
        Generic parsing - extract URL and basic info.
        
        Fallback parser for unrecognized formats.
        """
        logger.debug(f"Attempting generic parsing: {text[:100]}...")
        
        try:
            # Extract URL
            url_match = re.search(r'(https?://[^\s<>"\']+)', text)
            if not url_match:
                logger.error("No URL found in generic text")
                return None
            
            url = url_match.group(1)
            
            # Try to detect method
            method = 'GET'
            for http_method in ['POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS', 'GET']:
                if http_method in text.upper():
                    method = http_method
                    break
            
            # Extract parameters
            parameters = self._extract_url_parameters(url)
            
            # Basic headers
            headers = {}
            
            return ParsedPoC(
                method=method,
                url=url,
                headers=headers,
                body=None,
                parameters=parameters,
                format_detected='generic',
                confidence=0.5
            )
            
        except Exception as e:
            logger.error(f"Generic parsing error: {e}")
            self.stats['parse_errors'] += 1
            return None
    
    def _extract_url_parameters(self, url: str) -> Dict[str, str]:
        """Extract query parameters from URL."""
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            # Flatten lists to single values
            return {k: v[0] if isinstance(v, list) and v else v for k, v in params.items()}
        except Exception:
            return {}
    
    def get_stats(self) -> Dict[str, int]:
        """Get parser statistics."""
        return self.stats.copy()


# Global parser instance
global_poc_parser = PoCParser()


def parse_poc(poc_text: str) -> Optional[ParsedPoC]:
    """
    Convenience function to parse PoC.
    
    Args:
        poc_text: PoC in any supported format
        
    Returns:
        ParsedPoC object or None
    """
    return global_poc_parser.parse(poc_text)