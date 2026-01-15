"""
VORTEX Evidence Domain - V17.0 ULTIMATE
Evidence integrity and validation per VORTEX_EVIDENCE_STANDARDS.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
import hashlib
import json


@dataclass
class EvidencePackage:
    """
    Complete evidence package with cryptographic integrity
    Per VORTEX_EVIDENCE_STANDARDS.md: Evidence must be tamper-proof
    """
    finding_id: UUID
    evidence_type: str  # "request", "response", "screenshot", "poc"
    
    # Evidence data
    content: str
    content_type: str = "text/plain"
    encoding: str = "utf-8"
    
    # Integrity protection
    sha256_hash: str = ""
    signature: Optional[str] = None
    
    # Metadata
    collected_at: datetime = field(default_factory=datetime.utcnow)
    source: str = ""  # "heuristic", "ai", "system_verification", "manual"
    
    # Chain of custody
    custody_chain: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional context
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate hash if not provided."""
        if not self.sha256_hash:
            self.sha256_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        content_bytes = self.content.encode(self.encoding)
        return hashlib.sha256(content_bytes).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify evidence hasn't been tampered with."""
        current_hash = self._calculate_hash()
        return current_hash == self.sha256_hash
    
    def add_custody_entry(self, handler: str, action: str) -> None:
        """Add entry to chain of custody."""
        self.custody_chain.append({
            'timestamp': datetime.utcnow().isoformat(),
            'handler': handler,
            'action': action,
            'hash_at_time': self.sha256_hash
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'finding_id': str(self.finding_id),
            'evidence_type': self.evidence_type,
            'content': self.content,
            'content_type': self.content_type,
            'sha256_hash': self.sha256_hash,
            'collected_at': self.collected_at.isoformat(),
            'source': self.source,
            'custody_chain': self.custody_chain,
            'metadata': self.metadata
        }


@dataclass
class DeterministicEvidenceCriteria:
    """
    Criteria for deterministic evidence
    Per VORTEX_EVIDENCE_STANDARDS.md: Different vulnerability types have different criteria
    """
    vulnerability_type: str
    deterministic_indicators: List[str]
    confidence_bonus: float
    min_evidence_length: int
    
    # Pattern requirements
    required_patterns: List[str] = field(default_factory=list)
    optional_patterns: List[str] = field(default_factory=list)
    
    # Behavioral requirements
    requires_behavioral_changes: bool = True
    min_behavioral_indicators: int = 2
    
    def evaluate_evidence(self, evidence: str, behavioral_indicators: List[str]) -> float:
        """
        Evaluate evidence against criteria
        Returns confidence bonus (0.0-1.0)
        """
        score = 0.0
        
        # Check length requirement
        if len(evidence) < self.min_evidence_length:
            return 0.0
        
        evidence_lower = evidence.lower()
        
        # Check deterministic indicators
        indicator_matches = sum(
            1 for indicator in self.deterministic_indicators
            if indicator in evidence_lower
        )
        
        if indicator_matches >= 2:
            score = self.confidence_bonus
        elif indicator_matches == 1:
            score = self.confidence_bonus * 0.6
        
        # Check required patterns
        if self.required_patterns:
            pattern_matches = sum(
                1 for pattern in self.required_patterns
                if pattern in evidence_lower
            )
            if pattern_matches < len(self.required_patterns):
                score *= 0.5  # Penalty for missing required patterns
        
        # Check behavioral requirements
        if self.requires_behavioral_changes:
            if len(behavioral_indicators) < self.min_behavioral_indicators:
                score *= 0.7  # Penalty for insufficient behavioral evidence
        
        return min(score, 1.0)


@dataclass
class BehavioralEvidenceAnalysis:
    """
    Behavioral evidence analysis with uncertainty
    Per VORTEX_EVIDENCE_STANDARDS.md: Behavioral differences are INDICATIVE, not CONCLUSIVE
    """
    original_response: Dict[str, Any]
    modified_response: Dict[str, Any]
    payload: str
    
    # Analysis results
    indicators: List[str] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    # Specific measurements
    response_time_diff: float = 0.0
    status_code_changed: bool = False
    content_size_diff: int = 0
    payload_reflected: bool = False
    
    # Causation assessment
    causation_determination: str = "UNKNOWN - requires human expert analysis"
    max_automated_status: str = "SYSTEM_VERIFIED"
    
    def analyze(self) -> None:
        """
        Perform behavioral analysis with uncertainty acknowledgment
        Per VORTEX_EVIDENCE_STANDARDS.md rules
        """
        self.indicators.clear()
        self.uncertainty_factors.clear()
        
        # Response time analysis
        orig_time = self.original_response.get('response_time', 0.0)
        mod_time = self.modified_response.get('response_time', 0.0)
        self.response_time_diff = abs(mod_time - orig_time)
        
        if self.response_time_diff > 2.0:
            self.indicators.append(f"Response time change: {self.response_time_diff:.1f}s")
            self.uncertainty_factors.append("Could be infrastructure/load balancer, not application")
        
        # Status code analysis
        orig_status = self.original_response.get('status_code', 200)
        mod_status = self.modified_response.get('status_code', 200)
        self.status_code_changed = orig_status != mod_status
        
        if self.status_code_changed:
            self.indicators.append(f"Status change: {orig_status}→{mod_status}")
            self.uncertainty_factors.append("Could be upstream retry, rate limiting, or CDN switching")
        
        # Content size analysis
        orig_size = len(self.original_response.get('body', ''))
        mod_size = len(self.modified_response.get('body', ''))
        self.content_size_diff = abs(mod_size - orig_size)
        
        if self.content_size_diff > 100:
            self.indicators.append(f"Content size change: {self.content_size_diff} bytes")
            self.uncertainty_factors.append("Could be dynamic content, A/B testing, or cache variation")
        
        # Payload reflection analysis
        mod_body = self.modified_response.get('body', '').lower()
        self.payload_reflected = self.payload.lower() in mod_body
        
        if self.payload_reflected:
            self.indicators.append("Payload reflection detected")
            # This is more deterministic evidence
        
        # Calculate confidence with uncertainty penalty
        base_confidence = min(len(self.indicators) * 0.3, 0.9)
        uncertainty_penalty = len(self.uncertainty_factors) * 0.1
        self.confidence = max(0.0, base_confidence - uncertainty_penalty)


@dataclass
class EvidenceQualityScore:
    """
    Complete evidence quality assessment
    Per VORTEX_EVIDENCE_STANDARDS.md scoring system
    """
    # Component scores
    system_verification_score: float = 0.0
    ai_consistency_score: float = 0.0
    heuristic_consistency_score: float = 0.0
    vulnerability_specific_score: float = 0.0
    behavioral_score: float = 0.0
    
    # Final scores
    determinism_score: float = 0.0
    total_quality_score: float = 0.0
    
    # Requirements met
    meets_deterministic_standard: bool = False  # ≥0.8
    meets_behavioral_standard: bool = False  # ≥0.6
    meets_pattern_standard: bool = False  # ≥0.4
    
    def calculate_total_score(self) -> float:
        """Calculate total evidence quality score."""
        self.total_quality_score = (
            self.system_verification_score +
            self.ai_consistency_score +
            self.heuristic_consistency_score +
            self.vulnerability_specific_score +
            self.behavioral_score
        )
        return min(self.total_quality_score, 1.0)
    
    def calculate_determinism(self) -> float:
        """Calculate evidence determinism score."""
        self.determinism_score = (
            self.system_verification_score * 0.5 +
            self.vulnerability_specific_score * 0.3 +
            self.behavioral_score * 0.2
        )
        return min(self.determinism_score, 1.0)
    
    def assess_standards(self) -> None:
        """Assess which evidence standards are met."""
        self.calculate_determinism()
        
        self.meets_deterministic_standard = self.determinism_score >= 0.8
        self.meets_behavioral_standard = self.determinism_score >= 0.6
        self.meets_pattern_standard = self.determinism_score >= 0.4


# Evidence criteria by vulnerability type
# Per VORTEX_EVIDENCE_STANDARDS.md vulnerability-specific criteria
VULNERABILITY_EVIDENCE_CRITERIA = {
    'sql_injection': DeterministicEvidenceCriteria(
        vulnerability_type='sql_injection',
        deterministic_indicators=['mysql error', 'sql syntax', 'database error', 'ora-', 'postgresql'],
        confidence_bonus=0.15,
        min_evidence_length=50,
        required_patterns=['error', 'sql'],
        requires_behavioral_changes=True,
        min_behavioral_indicators=1
    ),
    'xss_reflected': DeterministicEvidenceCriteria(
        vulnerability_type='xss_reflected',
        deterministic_indicators=['javascript execution', 'alert fired', 'onerror triggered', '<script'],
        confidence_bonus=0.20,
        min_evidence_length=30,
        required_patterns=['script', 'javascript'],
        requires_behavioral_changes=True,
        min_behavioral_indicators=2
    ),
    'ssrf': DeterministicEvidenceCriteria(
        vulnerability_type='ssrf',
        deterministic_indicators=['internal response', '192.168', '10.', 'localhost', '127.0.0.1'],
        confidence_bonus=0.10,
        min_evidence_length=40,
        required_patterns=['internal', 'private'],
        requires_behavioral_changes=True,
        min_behavioral_indicators=2
    ),
    'lfi': DeterministicEvidenceCriteria(
        vulnerability_type='lfi',
        deterministic_indicators=['file content', 'etc/passwd', 'system file', 'root:x:'],
        confidence_bonus=0.05,
        min_evidence_length=60,
        required_patterns=['file', 'include'],
        requires_behavioral_changes=True,
        min_behavioral_indicators=2
    ),
}


def get_evidence_criteria(vulnerability_type: str) -> Optional[DeterministicEvidenceCriteria]:
    """Get evidence criteria for vulnerability type."""
    return VULNERABILITY_EVIDENCE_CRITERIA.get(vulnerability_type.lower())