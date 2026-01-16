"""
VORTEX Exception Hierarchy - V17.0 ULTIMATE
Comprehensive exception system for all VORTEX operations

EXCEPTION CATEGORIES:
- Authority violations
- Evidence validation failures
- AI integration errors
- Network/HTTP errors
- Database errors
- Workflow state errors
- Configuration errors
"""

from typing import Optional, Dict, Any


class VortexException(Exception):
    """Base exception for all VORTEX errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details
        }


# ============================================================================
# AUTHORITY & COMPLIANCE EXCEPTIONS
# ============================================================================

class AuthorityViolationError(VortexException):
    """
    CRITICAL: Authority hierarchy violation detected.
    
    This exception indicates fundamental system integrity failure.
    Must be logged and investigated immediately.
    """
    pass


class UnauthorizedAuthorityLevelError(AuthorityViolationError):
    """Attempted operation exceeds authority level."""
    pass


class AuthorityBypassAttemptError(AuthorityViolationError):
    """Detected attempt to bypass authority validation."""
    pass


class ComplianceViolationError(VortexException):
    """Legal or ethical compliance violation."""
    pass


class PIIDetectedError(ComplianceViolationError):
    """Personally Identifiable Information detected in unsafe context."""
    pass


class ScopeViolationError(ComplianceViolationError):
    """Operation outside authorized scope."""
    pass


# ============================================================================
# EVIDENCE & VALIDATION EXCEPTIONS
# ============================================================================

class EvidenceValidationError(VortexException):
    """Evidence validation failure."""
    pass


class InsufficientEvidenceError(EvidenceValidationError):
    """Evidence does not meet minimum standards."""
    pass


class EvidenceDeterminismError(EvidenceValidationError):
    """Evidence determinism score below threshold."""
    pass


class UnknownValueError(EvidenceValidationError):
    """
    CRITICAL: UNKNOWN value in field requiring determination.
    
    UNKNOWN ≠ LOW ≠ FALSE ≠ 0
    Must route to manual review, never proceed.
    """
    pass


class BehavioralAnalysisError(EvidenceValidationError):
    """Behavioral analysis failed or inconclusive."""
    pass


# ============================================================================
# AI INTEGRATION EXCEPTIONS
# ============================================================================

class AIIntegrationError(VortexException):
    """AI integration failure."""
    pass


class AIUnavailableError(AIIntegrationError):
    """All AI models unavailable."""
    pass


class AIResponseInvalidError(AIIntegrationError):
    """AI response failed validation."""
    pass


class AIFieldDerivationAttemptError(AIIntegrationError):
    """
    FORBIDDEN: Attempted to derive missing AI fields.
    
    Missing fields must remain UNKNOWN, never calculated.
    """
    pass


class HeuristicPoCReplayAttemptError(AIIntegrationError):
    """
    FORBIDDEN: Attempted to replay heuristic-generated PoC.
    
    Only AI-generated PoCs from successful analysis can be replayed.
    """
    pass


class MalformedJSONRecoveryError(AIIntegrationError):
    """Malformed JSON recovery attempted without proper restrictions."""
    pass


# ============================================================================
# WORKFLOW & STATE EXCEPTIONS
# ============================================================================

class WorkflowError(VortexException):
    """Workflow execution error."""
    pass


class InvalidStateTransitionError(WorkflowError):
    """Invalid state transition attempted."""
    
    def __init__(self, current_state: str, target_state: str, reason: str = ""):
        message = f"Invalid transition: {current_state} → {target_state}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {
            'current_state': current_state,
            'target_state': target_state,
            'reason': reason
        })


class WorkflowPhaseError(WorkflowError):
    """Error in specific workflow phase."""
    
    def __init__(self, phase: str, message: str, details: Optional[Dict] = None):
        super().__init__(f"Phase '{phase}' error: {message}", details)
        self.phase = phase


class ManualReviewRequiredError(WorkflowError):
    """Finding requires manual review."""
    
    def __init__(self, reason: str, finding_id: Optional[str] = None):
        super().__init__(f"Manual review required: {reason}", {
            'finding_id': finding_id,
            'reason': reason
        })


# ============================================================================
# NETWORK & HTTP EXCEPTIONS
# ============================================================================

class NetworkError(VortexException):
    """Network/HTTP operation failure."""
    pass


class HTTPRequestError(NetworkError):
    """HTTP request failed."""
    
    def __init__(self, url: str, method: str, status_code: Optional[int] = None, 
                 message: str = "Request failed"):
        super().__init__(message, {
            'url': url,
            'method': method,
            'status_code': status_code
        })


class ConnectionTimeoutError(NetworkError):
    """Connection timeout."""
    pass


class RateLimitExceededError(NetworkError):
    """Rate limit exceeded."""
    
    def __init__(self, retry_after: Optional[int] = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(message, {'retry_after': retry_after})


class WAFDetectedError(NetworkError):
    """Web Application Firewall detected and blocking."""
    pass


# ============================================================================
# DATABASE EXCEPTIONS
# ============================================================================

class DatabaseError(VortexException):
    """Database operation failure."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Database connection failure."""
    pass


class RecordNotFoundError(DatabaseError):
    """Database record not found."""
    
    def __init__(self, record_type: str, record_id: str):
        super().__init__(f"{record_type} not found: {record_id}", {
            'record_type': record_type,
            'record_id': record_id
        })


class DatabaseIntegrityError(DatabaseError):
    """Database integrity constraint violation."""
    pass


class DatabaseMigrationError(DatabaseError):
    """Database migration failure."""
    pass


# ============================================================================
# VERIFICATION EXCEPTIONS
# ============================================================================

class VerificationError(VortexException):
    """System verification failure."""
    pass


class PoCReplayError(VerificationError):
    """PoC replay failure."""
    pass


class VerificationTimeoutError(VerificationError):
    """Verification timeout."""
    pass


class ResponseMismatchError(VerificationError):
    """Expected vs actual response mismatch."""
    pass


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(VortexException):
    """Configuration error."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration missing."""
    
    def __init__(self, config_key: str):
        super().__init__(f"Required configuration missing: {config_key}", {
            'config_key': config_key
        })


class InvalidConfigurationError(ConfigurationError):
    """Configuration value invalid."""
    
    def __init__(self, config_key: str, value: Any, reason: str):
        super().__init__(f"Invalid configuration {config_key}={value}: {reason}", {
            'config_key': config_key,
            'value': str(value),
            'reason': reason
        })


# ============================================================================
# RESOURCE EXCEPTIONS
# ============================================================================

class ResourceError(VortexException):
    """Resource management error."""
    pass


class QueueError(ResourceError):
    """Generic queue error."""
    pass


class MemoryLimitExceededError(ResourceError):
    """Memory limit exceeded."""
    
    def __init__(self, current_mb: float, limit_mb: float):
        super().__init__(f"Memory limit exceeded: {current_mb:.0f}MB > {limit_mb:.0f}MB", {
            'current_mb': current_mb,
            'limit_mb': limit_mb
        })


class QueueOverflowError(ResourceError):
    """Queue capacity exceeded."""
    
    def __init__(self, queue_name: str, size: int, limit: int):
        super().__init__(f"Queue '{queue_name}' overflow: {size} > {limit}", {
            'queue_name': queue_name,
            'size': size,
            'limit': limit
        })


class CircuitBreakerOpenError(ResourceError):
    """Circuit breaker open - service unavailable."""
    
    def __init__(self, service_name: str, retry_after: Optional[int] = None):
        message = f"Circuit breaker open for '{service_name}'"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(message, {
            'service_name': service_name,
            'retry_after': retry_after
        })


# ============================================================================
# SCANNER EXCEPTIONS
# ============================================================================

class ScannerError(VortexException):
    """Scanner execution error."""
    pass


class PayloadGenerationError(ScannerError):
    """Payload generation failure."""
    pass


class HeuristicDetectionError(ScannerError):
    """Heuristic detection failure."""
    pass


class TargetUnreachableError(ScannerError):
    """Target URL unreachable."""
    
    def __init__(self, url: str, reason: str):
        super().__init__(f"Target unreachable: {url} ({reason})", {
            'url': url,
            'reason': reason
        })


# ============================================================================
# HEALTH MONITORING EXCEPTIONS
# ============================================================================

class HealthMonitoringError(VortexException):
    """Health monitoring error."""
    pass


class ThresholdViolationError(HealthMonitoringError):
    """Operational threshold violated."""
    
    def __init__(self, metric: str, value: float, threshold: float, 
                 severity: str = "WARNING"):
        super().__init__(
            f"{severity}: {metric} = {value:.2f} violates threshold {threshold:.2f}",
            {
                'metric': metric,
                'value': value,
                'threshold': threshold,
                'severity': severity
            }
        )


class AutoTuningError(HealthMonitoringError):
    """Auto-tuning operation error."""
    pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_exception_category(exc: Exception) -> str:
    """Get exception category for logging/metrics."""
    
    if isinstance(exc, AuthorityViolationError):
        return "AUTHORITY_VIOLATION"
    elif isinstance(exc, ComplianceViolationError):
        return "COMPLIANCE_VIOLATION"
    elif isinstance(exc, EvidenceValidationError):
        return "EVIDENCE_VALIDATION"
    elif isinstance(exc, AIIntegrationError):
        return "AI_INTEGRATION"
    elif isinstance(exc, WorkflowError):
        return "WORKFLOW"
    elif isinstance(exc, NetworkError):
        return "NETWORK"
    elif isinstance(exc, DatabaseError):
        return "DATABASE"
    elif isinstance(exc, VerificationError):
        return "VERIFICATION"
    elif isinstance(exc, ConfigurationError):
        return "CONFIGURATION"
    elif isinstance(exc, ResourceError):
        return "RESOURCE"
    elif isinstance(exc, ScannerError):
        return "SCANNER"
    elif isinstance(exc, HealthMonitoringError):
        return "HEALTH_MONITORING"
    elif isinstance(exc, VortexException):
        return "VORTEX_GENERAL"
    else:
        return "UNKNOWN"


def is_recoverable_error(exc: Exception) -> bool:
    """Check if error is potentially recoverable."""
    
    # Never recoverable - fundamental violations
    if isinstance(exc, (AuthorityViolationError, ComplianceViolationError)):
        return False
    
    # Potentially recoverable - transient failures
    if isinstance(exc, (NetworkError, AIUnavailableError, CircuitBreakerOpenError)):
        return True
    
    # Context-dependent
    if isinstance(exc, (DatabaseError, VerificationError)):
        return True
    
    # Default: not recoverable
    return False


def should_retry(exc: Exception) -> bool:
    """Check if operation should be retried."""
    
    # Transient network errors
    if isinstance(exc, (ConnectionTimeoutError, HTTPRequestError)):
        return True
    
    # Rate limiting - should retry after delay
    if isinstance(exc, RateLimitExceededError):
        return True
    
    # AI unavailable - can retry
    if isinstance(exc, AIUnavailableError):
        return True
    
    # Circuit breaker - should not retry immediately
    if isinstance(exc, CircuitBreakerOpenError):
        return False
    
    # Authority/compliance violations - never retry
    if isinstance(exc, (AuthorityViolationError, ComplianceViolationError)):
        return False
    
    # Default: no retry
    return False