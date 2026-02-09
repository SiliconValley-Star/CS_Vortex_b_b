"""
VORTEX Test Configuration and Fixtures
Per .clinerules compliance testing requirements
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_finding_data() -> Dict[str, Any]:
    """Mock finding data for testing."""
    return {
        "url": "https://example.com/search?q=test",
        "method": "GET",
        "parameters": {"q": "test"},
        "vulnerability_type": "sql_injection",
        "severity": "HIGH",
        "heuristic_score": 0.85,
        "evidence": "MySQL error: You have an error in your SQL syntax",
        "payload": "' OR '1'='1",
        "response_status": 500,
        "response_body": "MySQL error: syntax error at line 1"
    }


@pytest.fixture
def mock_system_verification_result():
    """Mock system verification result."""
    return {
        "success": True,
        "confidence": 0.87,
        "match_type": "exact_regex",
        "matched_pattern": "MySQL error:",
        "response_time": 1.2,
        "response_status": 500,
        "behavioral_changes": [
            "Response time increased by 0.8s",
            "Status code changed: 200 → 500",
            "Content size decreased by 1500 bytes"
        ]
    }


@pytest.fixture
def mock_ai_analysis_result():
    """Mock AI analysis result."""
    return {
        "model_used": "test_model",
        "verdict": "CONFIRMED",
        "confidence": 0.84,
        "exploitability": 0.82,
        "impact": "HIGH",
        "reportability": 0.88,
        "reasoning": "Clear SQL injection with MySQL error confirmation. "
                    "Payload reflection detected. Backend error indicates successful exploitation.",
        "poc_steps": "1. Send payload\n2. Observe MySQL error\n3. Confirm injection",
        "success": True,
        "is_fallback_result": False,
        "authority_level": "ADVISORY_ONLY",
        "is_authoritative": False
    }


@pytest.fixture
def mock_heuristic_only_result():
    """Mock heuristic-only result (AI unavailable)."""
    return {
        "model_used": "heuristic_fallback",
        "availability_status": "UNAVAILABLE",
        "verdict": "NEEDS_MANUAL",
        "confidence": 0.42,  # Penalized
        "exploitability": None,
        "impact": "UNKNOWN",
        "reportability": 0.02,
        "reasoning": "Heuristic-only analysis due to AI unavailability. "
                    "Original confidence: 0.85, adjusted: 0.42",
        "success": True,
        "is_fallback_result": True,
        "fallback_reason": "All AI models unavailable"
    }


@pytest.fixture
def mock_submit_ready_finding(mock_finding_data, mock_system_verification_result, mock_ai_analysis_result):
    """Mock finding that meets all SUBMIT_READY requirements."""
    from vortex.domain.models import AssessmentResult, VerificationResult, AIAnalysisResult
    from vortex.domain.enums import VerificationStatus, FindingSeverity
    
    finding = AssessmentResult(
        url=mock_finding_data["url"],
        vulnerability_type=mock_finding_data["vulnerability_type"],
        severity=FindingSeverity.HIGH,
        heuristic_score=mock_finding_data["heuristic_score"],
        evidence=mock_finding_data["evidence"],
        payload=mock_finding_data["payload"]
    )
    
    # System verification (AUTHORITATIVE)
    finding.verification_result = VerificationResult(**mock_system_verification_result)
    
    # AI analysis (ADVISORY ONLY)
    finding.ai_analysis = AIAnalysisResult(**mock_ai_analysis_result)
    
    finding.status = VerificationStatus.SUBMIT_READY
    
    return finding


@pytest.fixture
def mock_needs_manual_finding(mock_finding_data):
    """Mock finding that requires manual review."""
    from vortex.domain.models import AssessmentResult
    from vortex.domain.enums import VerificationStatus, FindingSeverity
    
    finding = AssessmentResult(
        url=mock_finding_data["url"],
        vulnerability_type=mock_finding_data["vulnerability_type"],
        severity=FindingSeverity.MEDIUM,
        heuristic_score=0.65,
        evidence="Potential SQL injection - requires manual verification"
    )
    
    finding.status = VerificationStatus.NEEDS_MANUAL
    
    return finding


@pytest.fixture
def mock_authority_violation_finding(mock_finding_data, mock_ai_analysis_result):
    """Mock finding with SUBMIT_READY but missing system verification (violation)."""
    from vortex.domain.models import AssessmentResult, AIAnalysisResult
    from vortex.domain.enums import VerificationStatus, FindingSeverity
    
    finding = AssessmentResult(
        url=mock_finding_data["url"],
        vulnerability_type=mock_finding_data["vulnerability_type"],
        severity=FindingSeverity.HIGH,
        heuristic_score=0.88,
        evidence=mock_finding_data["evidence"]
    )
    
    # AI analysis present BUT NO system verification (VIOLATION)
    finding.ai_analysis = AIAnalysisResult(**mock_ai_analysis_result)
    finding.verification_result = None  # MISSING - AUTHORITY VIOLATION
    
    finding.status = VerificationStatus.SUBMIT_READY  # INVALID
    
    return finding


@pytest.fixture
def mock_unknown_values_finding(mock_finding_data, mock_system_verification_result):
    """Mock finding with UNKNOWN values (must route to manual)."""
    from vortex.domain.models import AssessmentResult, VerificationResult, AIAnalysisResult
    from vortex.domain.enums import VerificationStatus, FindingSeverity
    
    finding = AssessmentResult(
        url=mock_finding_data["url"],
        vulnerability_type=mock_finding_data["vulnerability_type"],
        severity=FindingSeverity.HIGH,
        heuristic_score=0.82,
        evidence=mock_finding_data["evidence"]
    )
    
    finding.verification_result = VerificationResult(**mock_system_verification_result)
    
    # AI analysis with UNKNOWN values
    finding.ai_analysis = AIAnalysisResult(
        model_used="test_model",
        verdict="LIKELY",
        confidence=0.78,
        exploitability=None,  # UNKNOWN
        impact="UNKNOWN",      # UNKNOWN
        reportability=None,    # UNKNOWN
        reasoning="Analysis complete but impact assessment uncertain",
        success=True
    )
    
    return finding


@pytest.fixture
def mock_behavioral_analysis():
    """Mock behavioral analysis with uncertainty."""
    return {
        "indicators": [
            "Response time change: 2.3s",
            "Status change: 200→500",
            "Content size change: 1500 bytes"
        ],
        "uncertainty_factors": [
            "Could be infrastructure, not application",
            "Could be upstream retry or rate limiting",
            "Could be dynamic content or caching"
        ],
        "confidence": 0.51,  # 0.9 base - 0.39 penalty
        "causation_determination": "UNKNOWN - requires expert analysis",
        "max_automated_status": "SYSTEM_VERIFIED",
        "payload_reflected": True
    }


@pytest.fixture
def mock_health_metrics():
    """Mock operational health metrics."""
    return {
        "submit_ready_rate": 0.06,
        "manual_review_rate": 0.68,
        "false_positive_rate": 0.12,
        "ai_availability": 0.85,
        "memory_usage_mb": 4500,
        "error_rate": 0.04,
        "authority_violation_rate": 0.0,
        "evidence_determinism_avg": 0.78,
        "unknown_value_rate": 0.08,
        "manual_queue_size": 15,
        "manual_sla_compliance": 0.92
    }


@pytest.fixture
def mock_authority_config():
    """Mock authority configuration."""
    return {
        "hierarchy": {
            "SYSTEM_VERIFICATION": 1,
            "HUMAN_EXPERT": 2,
            "AI_ADVISORY": 3,
            "HEURISTIC": 4
        },
        "submit_ready_requirements": {
            "system_verification_required": True,
            "min_system_confidence": 0.75,
            "no_unknown_values": True,
            "deterministic_evidence": True
        }
    }


@pytest.fixture
def mock_evidence_config():
    """Mock evidence standards configuration."""
    return {
        "evidence_levels": {
            "DETERMINISTIC": {
                "min_score": 0.8,
                "required_for": ["SUBMIT_READY"]
            },
            "BEHAVIORAL": {
                "min_score": 0.6,
                "required_for": ["SYSTEM_VERIFIED"]
            },
            "PATTERN": {
                "min_score": 0.4,
                "required_for": ["AI_CONFIRMED"]
            }
        }
    }


# Test utilities
class MockWebSocketConnection:
    """Mock WebSocket connection for testing."""
    
    def __init__(self):
        self.messages_sent = []
        self.is_connected = True
    
    async def send(self, message: str):
        """Mock send method."""
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")
        self.messages_sent.append(message)
    
    async def receive(self) -> str:
        """Mock receive method."""
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")
        return '{"type": "test", "data": {}}'
    
    async def close(self):
        """Mock close method."""
        self.is_connected = False


@pytest.fixture
def mock_websocket():
    """Mock WebSocket fixture."""
    return MockWebSocketConnection()


# Assertion helpers
def assert_authority_compliance(finding):
    """Assert finding complies with authority hierarchy."""
    if finding.status == "SUBMIT_READY":
        assert finding.verification_result is not None, "SUBMIT_READY requires system verification"
        assert finding.verification_result.success, "SUBMIT_READY requires successful verification"
        assert finding.verification_result.confidence >= 0.75, "SUBMIT_READY requires confidence ≥ 0.75"


def assert_no_unknown_values(finding):
    """Assert finding has no UNKNOWN values."""
    if finding.ai_analysis:
        assert finding.ai_analysis.impact != "UNKNOWN", "UNKNOWN impact must route to NEEDS_MANUAL"
        assert finding.ai_analysis.exploitability is not None, "None exploitability indicates UNKNOWN"
        assert finding.ai_analysis.reportability is not None, "None reportability indicates UNKNOWN"


def assert_evidence_determinism(finding, min_score: float):
    """Assert finding meets evidence determinism requirements."""
    # This would use the actual evidence validator in production
    assert hasattr(finding, 'evidence_determinism_score'), "Finding must have evidence determinism score"
    assert finding.evidence_determinism_score >= min_score, f"Evidence determinism {finding.evidence_determinism_score} below {min_score}"


# Performance testing helpers
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.utcnow()
        
        def stop(self):
            self.end_time = datetime.utcnow()
        
        @property
        def elapsed_seconds(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0.0
    
    return Timer()