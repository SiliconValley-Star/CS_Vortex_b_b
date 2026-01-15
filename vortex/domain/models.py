"""
VORTEX Domain Models - V17.0 ULTIMATE
Core data structures following .clinerules authority hierarchy and evidence standards
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from .enums import (
    VerificationStatus,
    FindingType,
    FindingSeverity,
    AIVerdict,
    ImpactLevel,
    ConfidenceSource,
    EvidenceLevel,
    MatchType,
    AIAvailabilityStatus,
    AuthorityLevel,
    HealthStatus,
    ManualReviewPriority,
    ComplianceStatus
)


@dataclass
class VerificationResult:
    """
    System verification result - THE AUTHORITATIVE EVIDENCE SOURCE
    Per VORTEX_CORE_AUTHORITY.md: System verification is Level 1 authority
    """
    success: bool
    confidence: float  # 0.0-1.0
    match_type: MatchType
    
    # Evidence details
    matched_pattern: Optional[str] = None
    response_status: Optional[int] = None
    response_time: Optional[float] = None
    response_body_sample: Optional[str] = None
    
    # Behavioral indicators
    status_code_change: bool = False
    response_time_change: Optional[float] = None
    content_size_change: Optional[int] = None
    
    # Metadata
    verified_at: datetime = field(default_factory=datetime.utcnow)
    verification_method: str = ""
    error: Optional[str] = None
    
    # Determinism scoring
    determinism_score: float = 0.0
    
    def is_deterministic(self) -> bool:
        """Check if evidence meets deterministic standards."""
        return self.determinism_score >= 0.7


@dataclass
class AIAnalysisResult:
    """
    AI analysis result - ADVISORY ONLY, NEVER AUTHORITATIVE
    Per VORTEX_CORE_AUTHORITY.md: AI is Level 3 authority (advisory only)
    Per VORTEX_AI_INTEGRATION.md: AI provides expert opinion, not verdicts
    """
    model_used: str
    verdict: AIVerdict
    confidence: float  # 0.0-1.0, but ADVISORY confidence only
    reasoning: str
    
    # Optional fields - NEVER derived if missing (per VORTEX_AI_INTEGRATION.md)
    exploitability: Optional[float] = None  # None if AI didn't provide
    impact: str = "UNKNOWN"  # UNKNOWN if AI didn't provide (not LOW!)
    reportability: Optional[float] = None  # None if AI didn't provide
    
    # PoC information
    poc: Optional[str] = None
    poc_steps: Optional[List[str]] = None
    
    # Status and metadata
    success: bool = True
    is_fallback_result: bool = False
    fallback_reason: Optional[str] = None
    availability_status: AIAvailabilityStatus = AIAvailabilityStatus.AVAILABLE
    
    # Authority enforcement fields (NEW per .clinerules)
    authority_level: AuthorityLevel = AuthorityLevel.AI_ADVISORY
    is_authoritative: bool = False  # ALWAYS False for AI
    requires_system_validation: bool = True  # ALWAYS True for AI
    
    # Cross-validation (for multi-model consensus)
    cross_validation: Optional[Dict[str, Any]] = None
    
    # Error handling
    error_message: Optional[str] = None
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_advisory_only(self) -> bool:
        """Confirm AI result is advisory only (always True)."""
        return True  # AI is NEVER authoritative


@dataclass
class BehaviorAssessment:
    """
    Behavioral evidence assessment with uncertainty acknowledgment
    Per VORTEX_EVIDENCE_STANDARDS.md: Behavioral differences are INDICATIVE, not CONCLUSIVE
    """
    indicators: List[str]
    uncertainty_factors: List[str]
    confidence: float  # With uncertainty penalty applied
    causation_determination: str = "UNKNOWN - requires expert analysis"
    max_automated_status: VerificationStatus = VerificationStatus.SYSTEM_VERIFIED
    payload_reflected: bool = False
    
    # Detailed metrics
    response_time_diff: Optional[float] = None
    status_code_original: Optional[int] = None
    status_code_replay: Optional[int] = None
    content_size_diff: Optional[int] = None
    
    assessed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FastPathAssessment:
    """
    Fastpath promotion eligibility assessment
    Per VORTEX_FASTPATH_V11.md: Enables qualified findings to bypass manual review
    """
    eligible: bool
    score: float  # 0.0-1.0, must be ≥0.75 for eligibility
    qualifying_factors: List[str]
    blocking_factors: List[str]
    
    # Requirement checks
    has_strong_system_verification: bool = False
    has_deterministic_evidence: bool = False
    has_no_unknown_values: bool = False
    has_vuln_specific_evidence: bool = False
    has_ai_support: bool = False
    
    assessed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvidenceQuality:
    """Evidence quality metrics and scoring."""
    determinism_score: float
    evidence_level: EvidenceLevel
    vuln_specific_bonus: float = 0.0
    behavioral_confidence: Optional[float] = None
    
    # Quality indicators
    has_multiple_indicators: bool = False
    has_structural_changes: bool = False
    has_error_messages: bool = False
    
    meets_submit_ready_standard: bool = False


@dataclass
class AssessmentResult:
    """
    Complete finding assessment result with full audit trail
    Central data structure for vulnerability findings
    """
    # Core identification
    id: UUID = field(default_factory=uuid4)
    url: str = ""
    finding_type: FindingType = FindingType.INFO_DISCLOSURE
    severity: FindingSeverity = FindingSeverity.INFO
    
    # Detection information
    vulnerable_parameter: Optional[str] = None
    payload: Optional[str] = None
    evidence: str = ""
    description: str = ""
    remediation: str = ""
    heuristic_score: float = 0.0
    confidence_source: ConfidenceSource = ConfidenceSource.HEURISTIC_ONLY
    
    # Status tracking
    status: VerificationStatus = VerificationStatus.DETECTED
    previous_status: Optional[VerificationStatus] = None
    
    # Analysis results (ALL following authority hierarchy)
    ai_analysis: Optional[AIAnalysisResult] = None
    verification_result: Optional[VerificationResult] = None
    behavioral_analysis: Optional[BehaviorAssessment] = None
    evidence_quality: Optional[EvidenceQuality] = None
    fastpath_assessment: Optional[FastPathAssessment] = None
    
    # Authority compliance tracking (NEW per .clinerules)
    authority_level_met: AuthorityLevel = AuthorityLevel.HEURISTIC
    authority_violations: List[str] = field(default_factory=list)
    
    # Evidence scoring
    evidence_determinism_score: float = 0.0
    vulnerability_specific_evidence_bonus: float = 0.0
    
    # Response data (for behavioral analysis)
    original_response: Optional[Dict[str, Any]] = None
    replay_response: Optional[Dict[str, Any]] = None
    
    # Workflow tracking
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    error_history: List[str] = field(default_factory=list)
    
    # Timestamps
    detected_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    
    # Legal compliance
    compliance_status: ComplianceStatus = ComplianceStatus.UNKNOWN
    pii_detected: bool = False
    scope_validated: bool = False
    
    # Additional metadata
    vulnerability_type: Optional[str] = None  # String representation for flexibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def transition_state(self, new_status: VerificationStatus, reason: str) -> None:
        """
        Transition to new state with audit trail
        Per VORTEX_WORKFLOW_LIFECYCLE.md: All transitions must be validated
        """
        self.state_history.append({
            'from': self.status,
            'to': new_status,
            'reason': reason,
            'timestamp': datetime.utcnow()
        })
        self.previous_status = self.status
        self.status = new_status
        self.last_updated = datetime.utcnow()
    
    def record_error(self, error: str) -> None:
        """Record error in history."""
        self.error_history.append(f"[{datetime.utcnow()}] {error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'url': self.url,
            'finding_type': self.finding_type.value,
            'severity': self.severity.value,
            'vulnerable_parameter': self.vulnerable_parameter,
            'payload': self.payload,
            'evidence': self.evidence,
            'heuristic_score': self.heuristic_score,
            'status': self.status.value,
            'detected_at': self.detected_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class ManualReviewSLA:
    """
    Manual review SLA tracking
    Per VORTEX_WORKFLOW_LIFECYCLE.md: NEEDS_MANUAL is ACTIVE status requiring attention
    """
    finding_id: str
    assigned_at: datetime
    priority_level: ManualReviewPriority
    max_age_hours: int
    escalation_threshold_hours: int
    retry_count: int = 0
    
    # Status tracking
    is_overdue: bool = False
    is_escalated: bool = False
    escalated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def get_age_hours(self) -> float:
        """Calculate current age in hours."""
        return (datetime.utcnow() - self.assigned_at).total_seconds() / 3600
    
    def check_overdue(self) -> bool:
        """Check if review is overdue."""
        self.is_overdue = self.get_age_hours() > self.max_age_hours
        return self.is_overdue
    
    def check_escalation(self) -> bool:
        """Check if review needs escalation."""
        if not self.is_escalated and self.get_age_hours() > self.escalation_threshold_hours:
            self.is_escalated = True
            self.escalated_at = datetime.utcnow()
        return self.is_escalated


@dataclass
class SystemHealthMetrics:
    """
    System health metrics snapshot
    Per VORTEX_OPERATIONAL_HEALTH.md: Continuous monitoring with thresholds
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Finding distribution metrics
    total_findings: int = 0
    submit_ready_rate: float = 0.0
    manual_review_rate: float = 0.0
    false_positive_rate: float = 0.0
    
    # Manual review metrics
    manual_queue_size: int = 0
    overdue_reviews: int = 0
    avg_manual_hours: float = 0.0
    
    # AI system metrics
    ai_availability: float = 0.0
    ai_success_rate: float = 0.0
    
    # System resource metrics
    memory_usage_mb: float = 0.0
    memory_zone: str = "GREEN"
    error_rate: float = 0.0
    
    # Authority compliance metrics (NEW per .clinerules)
    authority_violation_rate: float = 0.0
    evidence_determinism_avg: float = 0.0
    unknown_value_rate: float = 0.0
    total_findings_checked: int = 0


@dataclass
class HealthAssessment:
    """System health assessment result."""
    status: HealthStatus
    warnings: List[str]
    critical_issues: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class TuningRecommendation:
    """Auto-tuning recommendation with impact estimation."""
    category: str
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    description: str
    actions: List[str]
    estimated_impact: str
    auto_executable: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed: bool = False
    executed_at: Optional[datetime] = None


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    timestamp: datetime
    overall_status: str
    current_metrics: SystemHealthMetrics
    threshold_violations: List[str]
    authority_compliance: Dict[str, Any]
    tuning_recommendations: List[TuningRecommendation]
    alerts: List[Any]  # HealthAlert objects
    trend_analysis: Dict[str, Any]


@dataclass
class HealthAlert:
    """System health alert with response requirements."""
    level: str  # INFO, WARNING, CRITICAL
    message: str
    timestamp: datetime
    requires_immediate_response: bool
    suggested_actions: List[str]
    acknowledged: bool = False
    resolved: bool = False
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class WorkflowMetrics:
    """Workflow execution metrics."""
    total_workflows: int = 0
    successful_workflows: int = 0
    failed_workflows: int = 0
    avg_duration_seconds: float = 0.0
    
    # Phase-specific metrics
    ai_analysis_success_rate: float = 0.0
    system_verification_success_rate: float = 0.0
    fastpath_promotion_rate: float = 0.0
    
    # Quality metrics
    authority_compliance_rate: float = 0.0
    evidence_quality_avg: float = 0.0


@dataclass
class ScanConfiguration:
    """Scan configuration parameters."""
    targets: List[str]
    mode: str
    output_dir: Optional[str] = None
    max_concurrent: int = 10
    request_delay: float = 1.0
    timeout: int = 30
    
    # Customization
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    auth: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    # Scope control
    scope_file: Optional[str] = None
    exclude_patterns: List[str] = field(default_factory=list)
    include_vulns: Optional[List[str]] = None
    exclude_vulns: Optional[List[str]] = None
    
    # Quality control
    ai_model: Optional[str] = None
    quality_threshold: float = 0.7


@dataclass
class ScanResult:
    """Complete scan result summary."""
    scan_id: UUID
    findings: List[AssessmentResult]
    
    # Statistics
    total_requests: int = 0
    total_findings: int = 0
    submit_ready_count: int = 0
    manual_review_count: int = 0
    false_positive_count: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Health
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)