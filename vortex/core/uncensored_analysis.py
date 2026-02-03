"""
VORTEX Uncensored Security Analysis - V17.0 ULTIMATE
Hermes 3 Uncensored wrapper for honest security assessment

Per .clinerules:
- Hermes 3 Uncensored for honest security analysis
- No AI censorship on security content
- Advisory-only role (never authoritative)
- Integration with OpenRouter

FEATURES:
- Uncensored security assessment
- Exploit analysis without restrictions
- Honest vulnerability classification
- Integration with AI advisory system
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from domain.models import AIAnalysisResult, AssessmentResult
from domain.enums import AIVerdict, AIAvailabilityStatus, AuthorityLevel
from config.constants import AI_LIMITS

logger = logging.getLogger(__name__)


@dataclass
class UncensoredAnalysisRequest:
    """Request for uncensored security analysis."""
    finding_data: Dict[str, Any]
    request_type: str  # 'vulnerability_assessment', 'exploit_analysis', 'poc_generation'
    context: Optional[str] = None


@dataclass
class UncensoredAnalysisResponse:
    """Response from uncensored analysis."""
    verdict: str
    confidence: float
    reasoning: str
    
    # Security-specific fields
    exploitability_assessment: Optional[str] = None
    attack_vector_analysis: Optional[str] = None
    poc_steps: Optional[str] = None
    
    # Metadata
    model_used: str = "hermes-3-uncensored"
    is_uncensored: bool = True
    success: bool = True


class UncensoredSecurityAnalyzer:
    """
    Hermes 3 Uncensored security analyzer.
    
    CRITICAL: This analyzer uses uncensored AI models for honest security
    assessment without content filtering that might block legitimate
    security research.
    
    RESPONSIBILITIES:
    - Honest vulnerability assessment
    - Exploit analysis without censorship
    - PoC generation assistance
    - Security-focused reasoning
    
    Per .clinerules: AI is ADVISORY ONLY, never authoritative
    """
    
    def __init__(self):
        self.model_name = "hermes-3-uncensored"
        self.advisory_only = True  # ALWAYS True per .clinerules
        
        # Statistics
        self.total_analyses = 0
        self.successful_analyses = 0
        
        logger.info("Uncensored Security Analyzer initialized")
    
    async def analyze_vulnerability(self, 
                                   finding: AssessmentResult,
                                   openrouter_client: Any) -> AIAnalysisResult:
        """
        Perform uncensored vulnerability analysis.
        
        Args:
            finding: Finding to analyze
            openrouter_client: OpenRouter client instance
            
        Returns:
            AI analysis result (ADVISORY ONLY)
        """
        self.total_analyses += 1
        
        try:
            # Prepare request
            request = self._prepare_analysis_request(finding)
            
            # Call OpenRouter with Hermes uncensored
            response = await openrouter_client.analyze_with_hermes_uncensored(request)
            
            # Convert to AIAnalysisResult
            result = self._convert_to_ai_result(response)
            
            # CRITICAL: Mark as advisory only per .clinerules
            result.authority_level = AuthorityLevel.AI_ADVISORY
            result.is_authoritative = False  # ALWAYS False
            result.requires_system_validation = True  # ALWAYS True
            
            self.successful_analyses += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Uncensored analysis failed: {e}", exc_info=True)
            return self._create_fallback_result(str(e))
    
    def _prepare_analysis_request(self, finding: AssessmentResult) -> Dict[str, Any]:
        """Prepare uncensored analysis request."""
        return {
            'url': finding.url,
            'finding_type': finding.finding_type.value if finding.finding_type else 'unknown',
            'severity': finding.severity.value if finding.severity else 'unknown',
            'evidence': finding.evidence,
            'vulnerable_parameter': finding.vulnerable_parameter,
            'payload': finding.payload,
            'heuristic_score': finding.heuristic_score,
            
            # Analysis instructions
            'analysis_type': 'uncensored_security_assessment',
            'request_honest_assessment': True,
            'include_exploit_analysis': True,
            'include_poc_if_possible': True
        }
    
    def _convert_to_ai_result(self, response: Dict[str, Any]) -> AIAnalysisResult:
        """
        Convert OpenRouter response to AIAnalysisResult.
        
        Per .clinerules: No field derivation - missing fields = None/UNKNOWN
        """
        # Required fields
        verdict_str = response.get('verdict', 'NEEDS_MANUAL')
        verdict = AIVerdict[verdict_str] if verdict_str in AIVerdict.__members__ else AIVerdict.NEEDS_MANUAL
        
        confidence = float(response.get('confidence', 0.0))
        reasoning = response.get('reasoning', 'No reasoning provided')
        
        # Optional fields - NEVER derive if missing (per .clinerules)
        exploitability = response.get('exploitability')  # None if not provided
        impact = response.get('impact', 'UNKNOWN')  # UNKNOWN if not provided (not LOW!)
        reportability = response.get('reportability')  # None if not provided
        
        # PoC information
        poc = response.get('poc')
        poc_steps = response.get('poc_steps')
        
        return AIAnalysisResult(
            model_used=self.model_name,
            verdict=verdict,
            confidence=min(confidence, AI_LIMITS.advisory_confidence_cap),  # Cap per .clinerules
            reasoning=reasoning,
            
            # Optional fields - NOT DERIVED
            exploitability=exploitability,
            impact=impact,
            reportability=reportability,
            
            # PoC
            poc=poc,
            poc_steps=poc_steps,
            
            # Status
            success=True,
            is_fallback_result=False,
            availability_status=AIAvailabilityStatus.AVAILABLE,
            
            # Authority marking (CRITICAL per .clinerules)
            authority_level=AuthorityLevel.AI_ADVISORY,
            is_authoritative=False,  # ALWAYS False
            requires_system_validation=True  # ALWAYS True
        )
    
    def _create_fallback_result(self, error_message: str) -> AIAnalysisResult:
        """Create fallback result when analysis fails."""
        return AIAnalysisResult(
            model_used=self.model_name,
            verdict=AIVerdict.NEEDS_MANUAL,
            confidence=0.0,
            reasoning=f"Uncensored analysis unavailable: {error_message}",
            
            # All optional fields None/UNKNOWN
            exploitability=None,
            impact="UNKNOWN",
            reportability=None,
            
            # Status
            success=False,
            is_fallback_result=True,
            availability_status=AIAvailabilityStatus.UNAVAILABLE,
            error_message=error_message,
            
            # Authority marking
            authority_level=AuthorityLevel.AI_ADVISORY,
            is_authoritative=False,
            requires_system_validation=True,
            fallback_reason="Uncensored model unavailable"
        )
    
    async def generate_poc(self, 
                          finding: AssessmentResult,
                          openrouter_client: Any) -> Optional[str]:
        """
        Generate PoC using uncensored model.
        
        Args:
            finding: Finding to generate PoC for
            openrouter_client: OpenRouter client
            
        Returns:
            PoC string or None
        """
        try:
            request = {
                'task': 'poc_generation',
                'finding': self._prepare_analysis_request(finding),
                'requirements': [
                    'Provide step-by-step PoC',
                    'Include actual payload/exploit',
                    'Explain expected behavior',
                    'Be technically accurate'
                ]
            }
            
            response = await openrouter_client.generate_poc_uncensored(request)
            
            if response and 'poc' in response:
                logger.info(f"Generated PoC for {finding.id} using uncensored model")
                return response['poc']
            
            return None
            
        except Exception as e:
            logger.error(f"PoC generation failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        success_rate = self.successful_analyses / self.total_analyses if self.total_analyses > 0 else 0.0
        
        return {
            'model_used': self.model_name,
            'total_analyses': self.total_analyses,
            'successful_analyses': self.successful_analyses,
            'success_rate': success_rate,
            'is_advisory_only': self.advisory_only,  # Always True
            'is_authoritative': False  # Always False per .clinerules
        }


# Global uncensored security analyzer instance
global_uncensored_analyzer = UncensoredSecurityAnalyzer()