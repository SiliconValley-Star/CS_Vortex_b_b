"""
VORTEX Core System - V17.0 ULTIMATE
Complete production-grade core infrastructure

MODULES:
- Authority: Immutable authority hierarchy enforcement
- Evidence: Evidence standards and validation
- AI: Advisory AI integration with strict limitations
- Workflow: Complete workflow orchestration
- Health: Operational health monitoring and auto-tuning
- Database: Production database with audit trail
- Network: Resilient HTTP client
- Verification: Authoritative system verification
- Engine: Main scan engine
- Exceptions: Comprehensive exception hierarchy
"""

# Authority system
from .authority.hierarchy import (
    AuthorityHierarchyEnforcer,
    global_authority_enforcer
)
from .authority.validator import AuthorityValidator
from .authority.compliance import (
    AuthorityComplianceTracker,
    global_compliance_tracker as global_authority_tracker
)

# Evidence system
from .evidence.standards import (
    EvidenceStandardsValidator,
    global_evidence_validator
)
from .evidence.behavioral import BehavioralAnalyzer
from .evidence.determinism import DeterminismScorer

# AI integration
from .ai.openrouter import OpenRouterClient
from .ai.advisory import (
    AIAdvisoryEngine,
    global_ai_advisory_engine
)
from .ai.fallbacks import AIFallbackManager

# Workflow system
from .workflow.state_machine import (
    StateMachine,
    global_state_machine
)
from .workflow.orchestrator import (
    WorkflowOrchestrator,
    global_workflow_orchestrator
)

# Health monitoring
from .health.monitor import (
    HealthMonitor,
    check_system_health,
    global_health_monitor
)
from .health.auto_tune import (
    AutoTuningEngine,
    global_auto_tuning_engine
)

# Infrastructure
from .database import (
    DatabaseManager,
    global_database_manager
)
from .network import (
    NetworkClient,
    HTTPResponse,
    global_network_client
)
from .verification import (
    SystemVerificationEngine,
    global_verification_engine
)
from .engine import (
    VortexScanEngine,
    global_scan_engine
)
from .state import (
    StateManager,
    SystemState,
    global_state_manager
)

# Exceptions
from .exceptions import (
    VortexException,
    AuthorityViolationError,
    EvidenceValidationError,
    AIIntegrationError,
    WorkflowError,
    NetworkError,
    DatabaseError,
    VerificationError,
    ConfigurationError
)

__all__ = [
    # Authority
    'AuthorityHierarchyEnforcer',
    'AuthorityValidator',
    'AuthorityComplianceTracker',
    'global_authority_enforcer',
    'global_authority_tracker',
    
    # Evidence
    'EvidenceStandardsValidator',
    'BehavioralAnalyzer',
    'DeterminismScorer',
    'global_evidence_validator',
    
    # AI
    'OpenRouterClient',
    'AIAdvisoryEngine',
    'AIFallbackManager',
    'global_ai_advisory_engine',
    
    # Workflow
    'StateMachine',
    'WorkflowOrchestrator',
    'global_state_machine',
    'global_workflow_orchestrator',
    
    # Health
    'HealthMonitor',
    'AutoTuningEngine',
    'check_system_health',
    'global_health_monitor',
    'global_auto_tuning_engine',
    
    # Infrastructure
    'DatabaseManager',
    'NetworkClient',
    'HTTPResponse',
    'SystemVerificationEngine',
    'VortexScanEngine',
    'StateManager',
    'SystemState',
    'global_database_manager',
    'global_network_client',
    'global_verification_engine',
    'global_scan_engine',
    'global_state_manager',
    
    # Exceptions
    'VortexException',
    'AuthorityViolationError',
    'EvidenceValidationError',
    'AIIntegrationError',
    'WorkflowError',
    'NetworkError',
    'DatabaseError',
    'VerificationError',
    'ConfigurationError',
]