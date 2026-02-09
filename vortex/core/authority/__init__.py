"""
VORTEX Authority System - V17.0 ULTIMATE
Complete authority hierarchy enforcement per VORTEX_CORE_AUTHORITY.md

This package provides:
- Authority hierarchy enforcement (System > Human > AI > Heuristic)
- Authority validation for all decisions
- Compliance tracking and audit trail

IMMUTABLE AUTHORITY HIERARCHY:
1. System Verification (Deterministic) - HIGHEST AUTHORITY
2. Human Expert Analysis (Authoritative) - SECOND
3. AI Analysis (ADVISORY ONLY) - THIRD
4. Heuristic Detection (Indicative) - LOWEST

GOLDEN RULE: AI IS NEVER AUTHORITATIVE
"""

from core.authority.hierarchy import (
    AuthorityHierarchyEnforcer,
    get_authority_level,
    compare_authority_levels,
    is_authority_sufficient,
    global_authority_enforcer
)

from core.authority.validator import (
    AuthorityValidator,
    global_authority_validator
)

from core.authority.compliance import (
    AuthorityComplianceTracker,
    ComplianceAuditor,
    global_compliance_tracker,
    global_compliance_auditor
)

__all__ = [
    # Hierarchy enforcement
    'AuthorityHierarchyEnforcer',
    'get_authority_level',
    'compare_authority_levels', 
    'is_authority_sufficient',
    'global_authority_enforcer',
    
    # Validation
    'AuthorityValidator',
    'global_authority_validator',
    
    # Compliance tracking
    'AuthorityComplianceTracker',
    'ComplianceAuditor',
    'global_compliance_tracker',
    'global_compliance_auditor',
]