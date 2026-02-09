"""
VORTEX TIER 3 Payloads - Aggressive/Manual Mode
SecLists integration with quality filtering

TIER 3 CHARACTERISTICS:
- Success rate: 40-60% (aggressive but targeted)
- WAF bypass: Variable
- False positive: 10-25%
- Use case: MANUAL TESTING ONLY
- Volume: High coverage with SecLists
"""

from core.payloads.curated_payloads import CuratedPayload, VulnType, PayloadTier
from typing import List
import logging

logger = logging.getLogger(__name__)


def load_tier3_from_seclists() -> List[CuratedPayload]:
    """
    Load TIER 3 payloads from SecLists with quality filtering.
    
    TIER 3 Philosophy:
    - Use SecLists but with quality checks
    - Filter out obvious noise/spam
    - Mark as manual-only
    - Maintain some quality standards
    """
    try:
        from core.payloads.seclists_loader import SecListsLoader, PayloadCategory as SecListsCategory
        
        loader = SecListsLoader()
        tier3_payloads = []
        
        # Load from SecLists categories
        categories = {
            SecListsCategory.XSS: VulnType.XSS,
            SecListsCategory.SQLI: VulnType.SQLI,
            SecListsCategory.LFI: VulnType.LFI,
            SecListsCategory.SSRF: VulnType.SSRF,
            SecListsCategory.SSTI: VulnType.SSTI,
            SecListsCategory.XXE: VulnType.XXE,
            SecListsCategory.COMMAND_INJECTION: VulnType.COMMAND_INJECTION,
        }
        
        for seclists_category, vuln_type in categories.items():
            try:
                # Get payloads from SecLists
                payloads = loader.get_payloads(seclists_category)
                
                # Convert to CuratedPayload format with TIER 3
                # Apply basic quality filtering
                for payload_str in payloads[:100]:  # Limit per category
                    # Skip empty or very short payloads
                    if not payload_str or len(payload_str) < 2:
                        continue
                    
                    # Skip obvious duplicates/noise
                    if payload_str.count(payload_str[0]) == len(payload_str):
                        continue  # All same character
                    
                    # Create TIER 3 payload with conservative estimates
                    tier3_payload = CuratedPayload(
                        content=payload_str,
                        vuln_type=vuln_type,
                        tier=PayloadTier.TIER_3,
                        success_rate=0.50,  # Conservative estimate
                        waf_bypass_prob=0.40,  # Lower for TIER 3
                        false_positive_rate=0.15,  # Higher FP acceptable
                        description=f"SecLists {seclists_category.value} payload",
                        tags=["seclists", "tier3", "manual"],
                        source="seclists"
                    )
                    tier3_payloads.append(tier3_payload)
                    
                logger.info(f"Loaded {len([p for p in tier3_payloads if p.vuln_type == vuln_type])} TIER 3 {seclists_category.value} payloads from SecLists")
                
            except Exception as e:
                logger.warning(f"Could not load TIER 3 {seclists_category.value} from SecLists: {e}")
                continue
        
        logger.info(f"Total TIER 3 payloads loaded: {len(tier3_payloads)}")
        return tier3_payloads
        
    except ImportError:
        logger.warning("SecLists loader not available, skipping TIER 3")
        return []
    except Exception as e:
        logger.error(f"Error loading TIER 3 payloads: {e}")
        return []


def get_tier3_summary(payloads: List[CuratedPayload]) -> dict:
    """Get summary statistics for TIER 3 payloads."""
    if not payloads:
        return {}
    
    by_type = {}
    for vuln_type in VulnType:
        count = len([p for p in payloads if p.vuln_type == vuln_type])
        by_type[vuln_type.value] = count
    
    return {
        'total': len(payloads),
        'by_type': by_type,
        'tier': 'tier_3',
        'mode': 'manual_only',
        'warning': 'TIER 3 payloads are for manual testing only. Not recommended for automated scans.'
    }


class Tier3PayloadLoader:
    """Wrapper class for TIER 3 payload loading."""
    
    def __init__(self):
        self._payloads = None
    
    def load_payloads(self) -> List[CuratedPayload]:
        """Load TIER 3 payloads from SecLists."""
        if self._payloads is None:
            self._payloads = load_tier3_from_seclists()
        return self._payloads
    
    def get_summary(self) -> dict:
        """Get summary of loaded payloads."""
        if self._payloads is None:
            self.load_payloads()
        return get_tier3_summary(self._payloads)
    
    @staticmethod
    def load() -> List[CuratedPayload]:
        """Static method to load TIER 3 payloads."""
        return load_tier3_from_seclists()