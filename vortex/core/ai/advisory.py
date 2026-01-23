"""
VORTEX AI Advisory Engine - V17.0 ULTIMATE
AI advisory analysis per VORTEX_AI_INTEGRATION.md

CRITICAL: AI IS ADVISORY ONLY - NEVER AUTHORITATIVE
Per VORTEX_CORE_AUTHORITY.md: AI provides expert opinion simulation, NOT security verdicts
"""

import structlog
from typing import Dict, Optional
from datetime import datetime

from domain.enums import AIVerdict, AuthorityLevel, ConfidenceSource
from domain.models import AssessmentResult, AIAnalysisResult
from config.prompts import (
    SYSTEM_PROMPT_CORE,
    VULNERABILITY_ASSESSMENT_PROMPT,
    BEHAVIORAL_ANALYSIS_PROMPT
)
from config.constants import AI_INTEGRATION_LIMITS
from core.ai.openrouter import get_openrouter_client
from core.ai.fallbacks import create_ai_unavailable_result

logger = structlog.get_logger()


class AIAdvisoryEngine:
    """
    AI advisory analysis engine
    Per VORTEX_AI_INTEGRATION.md: AI is ADVISORY ONLY, never authoritative
    """
    
    def __init__(self):
        # Initialize OpenRouter client (lazy initialization)
        self.openrouter = get_openrouter_client()
        self.analysis_history = []
        
        # Advisory limitations per VORTEX_AI_INTEGRATION.md
        self.advisory_limits = {
            'is_authoritative': False,
            'can_create_submit_ready': False,
            'can_derive_fields': False,
            'can_replay_heuristic_poc': False,
            'malformed_recovery_authoritative': False
        }
    
    async def perform_advisory_analysis(
        self,
        finding_data: Dict,
        finding: Optional[AssessmentResult] = None
    ) -> AIAnalysisResult:
        """
        Perform AI advisory analysis on finding
        Per VORTEX_AI_INTEGRATION.md: Advisory role only
        
        Args:
            finding_data: Finding information dict
            finding: Optional AssessmentResult for context
        
        Returns: AIAnalysisResult marked as ADVISORY ONLY
        """
        logger.info(
            "Starting AI advisory analysis",
            finding_id=finding_data.get('id', 'unknown'),
            vuln_type=finding_data.get('finding_type', 'unknown')
        )
        
        try:
            # Primary model: Hermes uncensored (honest security assessment)
            hermes_result = await self._analyze_with_hermes(finding_data)
            
            # Secondary model: Gemini 2.0 (fast validation)
            gemini_result = await self._analyze_with_gemini(finding_data)
            
            # Create advisory consensus
            advisory_result = self._create_advisory_consensus(
                hermes_result,
                gemini_result,
                finding_data
            )
            
            # CRITICAL: Mark as advisory only
            advisory_result.authority_level = AuthorityLevel.AI_ADVISORY
            advisory_result.is_authoritative = False
            advisory_result.requires_system_validation = True
            
            logger.info(
                "AI advisory analysis complete",
                finding_id=finding_data.get('id', 'unknown'),
                verdict=advisory_result.verdict,
                confidence=advisory_result.confidence,
                is_fallback=advisory_result.is_fallback_result
            )
            
            self._record_analysis(advisory_result, finding_data)
            
            return advisory_result
            
        except Exception as e:
            logger.error(
                "AI advisory analysis failed",
                finding_id=finding_data.get('id', 'unknown'),
                error=str(e)
            )
            
            # Return AI unavailable result
            return create_ai_unavailable_result(finding_data)
    
    async def _analyze_with_hermes(self, finding_data: Dict) -> Optional[AIAnalysisResult]:
        """Analyze with Hermes uncensored model."""
        try:
            # Build prompt
            prompt = self._build_analysis_prompt(finding_data)
            
            # Call model
            response = await self.openrouter.call_model(
                model="nousresearch/hermes-3-llama-3.1-405b",
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT_CORE,
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse response
            data = await self.openrouter.parse_json_response(
                response['content'],
                allow_recovery=True
            )
            
            if not data:
                raise Exception("Failed to parse Hermes response")
            
            # Validate and create result
            return self._create_analysis_result(data, "hermes-3-llama-3.1-405b", finding_data)
            
        except Exception as e:
            logger.warning(f"Hermes analysis failed: {e}")
            return None
    
    async def _analyze_with_gemini(self, finding_data: Dict) -> Optional[AIAnalysisResult]:
        """Analyze with Gemini 2.0 model for validation."""
        try:
            # Build prompt
            prompt = self._build_analysis_prompt(finding_data, concise=True)
            
            # Call model
            response = await self.openrouter.call_model(
                model="google/gemini-2.0-flash-exp:free",
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT_CORE,
                max_tokens=1500,
                temperature=0.2
            )
            
            # Parse response
            data = await self.openrouter.parse_json_response(
                response['content'],
                allow_recovery=True
            )
            
            if not data:
                raise Exception("Failed to parse Gemini response")
            
            # Validate and create result
            return self._create_analysis_result(data, "gemini-2.0-flash", finding_data)
            
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            return None
    
    def _build_analysis_prompt(self, finding_data: Dict, concise: bool = False) -> str:
        """Build analysis prompt for AI model."""
        vuln_type = finding_data.get('finding_type', 'unknown')
        url = finding_data.get('url', 'unknown')
        evidence = finding_data.get('evidence', '')
        payload = finding_data.get('payload', '')
        parameter = finding_data.get('parameter', '')
        
        prompt_template = VULNERABILITY_ASSESSMENT_PROMPT
        
        prompt = f"""{prompt_template}

FINDING DETAILS:
- Vulnerability Type: {vuln_type}
- Target URL: {url}
- Parameter: {parameter}
- Test Payload: {payload}
- Evidence: {evidence[:500]}...

CRITICAL INSTRUCTIONS:
1. Provide ADVISORY analysis only - NOT authoritative verdict
2. Return JSON format with ALL required fields
3. DO NOT derive missing fields - leave as null if uncertain
4. Focus on technical accuracy over confidence

Required JSON format:
{{
    "verdict": "CONFIRMED" | "LIKELY" | "FALSE_POSITIVE" | "NEEDS_MANUAL",
    "confidence": 0.0-1.0,
    "exploitability": 0.0-1.0 or null,
    "impact": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | "UNKNOWN",
    "reportability": 0.0-1.0 or null,
    "reasoning": "detailed technical reasoning",
    "poc_steps": "optional PoC steps if applicable"
}}
"""
        
        return prompt
    
    def _create_analysis_result(
        self,
        data: Dict,
        model: str,
        finding_data: Dict
    ) -> AIAnalysisResult:
        """
        Create AIAnalysisResult from parsed data
        Per VORTEX_AI_INTEGRATION.md: NO field derivation
        """
        # CRITICAL: Check if this was recovered from malformed JSON
        is_recovered = data.get('_recovered', False)
        
        # Required fields - fail if missing
        required = ["verdict", "confidence", "reasoning"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Parse verdict
        verdict_str = data["verdict"].upper()
        try:
            verdict = AIVerdict[verdict_str]
        except KeyError:
            logger.warning(f"Invalid verdict '{verdict_str}', defaulting to NEEDS_MANUAL")
            verdict = AIVerdict.NEEDS_MANUAL
        
        # Optional fields - remain None if missing (DO NOT DERIVE)
        # Per VORTEX_AI_INTEGRATION.md: Missing fields must stay UNKNOWN
        exploitability = data.get("exploitability")  # May be None
        impact = data.get("impact", "UNKNOWN")       # UNKNOWN if missing
        reportability = data.get("reportability")    # May be None
        poc_steps = data.get("poc_steps", "")
        
        # If recovered from malformed JSON, apply severe penalties
        if is_recovered:
            confidence = min(data["confidence"] * 0.3, 0.4)  # Severe penalty
            verdict = AIVerdict.NEEDS_MANUAL  # Force manual review
            reportability = None  # Not reportable
        else:
            confidence = float(data["confidence"])
        
        result = AIAnalysisResult(
            model_used=model,
            verdict=verdict,
            confidence=confidence,
            exploitability=exploitability,      # NOT DERIVED
            impact=impact,                      # NOT DERIVED
            reportability=reportability,        # NOT DERIVED
            reasoning=data["reasoning"],
            poc=poc_steps,
            success=not is_recovered,           # Failed if recovered
            is_fallback_result=is_recovered,
            authority_level=AuthorityLevel.AI_ADVISORY,
            is_authoritative=False,
            requires_system_validation=True,
            error_message="Malformed JSON recovery - non-authoritative" if is_recovered else None
        )
        
        return result
    
    def _create_advisory_consensus(
        self,
        hermes_result: Optional[AIAnalysisResult],
        gemini_result: Optional[AIAnalysisResult],
        finding_data: Dict
    ) -> AIAnalysisResult:
        """
        Create advisory consensus from multiple models
        Per VORTEX_AI_INTEGRATION.md: Consensus is still ADVISORY ONLY
        """
        # Handle complete failures
        if not hermes_result and not gemini_result:
            return create_ai_unavailable_result(finding_data)
        
        # Single model available
        if not hermes_result:
            return self._mark_as_advisory_only(gemini_result)
        if not gemini_result:
            return self._mark_as_advisory_only(hermes_result)
        
        # Both available - create consensus
        # Hermes primary (uncensored advantage for security analysis)
        consensus = AIAnalysisResult(
            model_used="hermes_gemini_advisory_consensus",
            verdict=hermes_result.verdict,  # Hermes primary
            confidence=self._calculate_advisory_confidence(hermes_result, gemini_result),
            
            # Advisory fields - NO derivation if missing
            exploitability=hermes_result.exploitability,
            impact=hermes_result.impact if hermes_result.impact != "UNKNOWN" else "UNKNOWN",
            reportability=hermes_result.reportability,
            
            # Combined reasoning
            reasoning=(
                f"ADVISORY ANALYSIS (Multi-Model Consensus):\n\n"
                f"Hermes (Primary - Uncensored):\n{hermes_result.reasoning}\n\n"
                f"Gemini (Validation):\n{gemini_result.reasoning if gemini_result else 'N/A'}"
            ),
            poc=hermes_result.poc,
            
            # Advisory metadata
            success=True,
            is_fallback_result=False,
            authority_level=AuthorityLevel.AI_ADVISORY,
            is_authoritative=False,
            requires_system_validation=True,
            
            # Cross-validation info
            cross_validation={
                'hermes_verdict': hermes_result.verdict.value,
                'gemini_verdict': gemini_result.verdict.value if gemini_result else 'UNKNOWN',
                'consensus_agreement': hermes_result.verdict == gemini_result.verdict if gemini_result else False,
                'advisory_note': 'AI consensus provides advisory input only - not authoritative'
            }
        )
        
        return consensus
    
    def _calculate_advisory_confidence(
        self,
        hermes_result: AIAnalysisResult,
        gemini_result: Optional[AIAnalysisResult]
    ) -> float:
        """
        Calculate advisory confidence - reduced to reflect advisory nature
        Per VORTEX_AI_INTEGRATION.md: AI confidence capped for advisory role
        """
        hermes_conf = hermes_result.confidence
        gemini_conf = gemini_result.confidence if gemini_result else 0.5
        
        # Weighted average (Hermes priority for uncensored analysis)
        base_confidence = hermes_conf * 0.7 + gemini_conf * 0.3
        
        # Consensus boost/penalty
        if gemini_result and hermes_result.verdict == gemini_result.verdict:
            consensus_mult = 1.1  # Boost for agreement
        else:
            consensus_mult = 0.9  # Penalty for disagreement
        
        # Advisory confidence cap (never too confident - AI is not authoritative)
        advisory_confidence = min(base_confidence * consensus_mult, 0.95)
        
        return max(advisory_confidence, 0.0)
    
    def _mark_as_advisory_only(self, result: AIAnalysisResult) -> AIAnalysisResult:
        """
        Mark AI result as advisory only
        Reduce confidence to reflect advisory nature
        """
        result.authority_level = AuthorityLevel.AI_ADVISORY
        result.is_authoritative = False
        result.requires_system_validation = True
        
        # Reduce confidence to reflect advisory nature
        result.confidence = min(result.confidence * 0.9, 0.90)
        
        return result
    
    def should_replay_poc(self, finding: AssessmentResult) -> bool:
        """
        Determine if PoC should be replayed
        Per VORTEX_AI_INTEGRATION.md: NEVER replay heuristic PoCs
        """
        # NEVER replay heuristic-only PoCs
        if (hasattr(finding, 'confidence_source') and 
            finding.confidence_source == ConfidenceSource.HEURISTIC_ONLY):
            logger.debug(
                "Blocking heuristic PoC replay",
                finding_id=str(finding.id)
            )
            return False
        
        # Only replay AI-generated PoCs from successful analysis
        if not finding.ai_analysis:
            return False
        
        return (
            finding.ai_analysis.success and
            hasattr(finding.ai_analysis, 'poc') and
            bool(finding.ai_analysis.poc) and
            not finding.ai_analysis.is_fallback_result
        )
    
    def _record_analysis(self, result: AIAnalysisResult, finding_data: Dict) -> None:
        """Record analysis for audit."""
        record = {
            'timestamp': datetime.utcnow(),
            'finding_id': finding_data.get('id', 'unknown'),
            'model': result.model_used,
            'verdict': result.verdict.value,
            'confidence': result.confidence,
            'is_fallback': result.is_fallback_result,
            'success': result.success
        }
        
        self.analysis_history.append(record)
    
    def get_analysis_stats(self) -> Dict:
        """Get AI analysis statistics."""
        if not self.analysis_history:
            return {
                'total_analyses': 0,
                'success_rate': 1.0,
                'avg_confidence': 0.0
            }
        
        total = len(self.analysis_history)
        successes = sum(1 for a in self.analysis_history if a['success'])
        
        return {
            'total_analyses': total,
            'success_rate': successes / total,
            'avg_confidence': sum(a['confidence'] for a in self.analysis_history) / total,
            'fallback_rate': sum(1 for a in self.analysis_history if a['is_fallback']) / total,
            'verdict_distribution': self._get_verdict_distribution()
        }
    
    def _get_verdict_distribution(self) -> Dict[str, int]:
        """Get distribution of AI verdicts."""
        from collections import Counter
        verdicts = [a['verdict'] for a in self.analysis_history]
        return dict(Counter(verdicts))


async def perform_ai_advisory_analysis(
    finding_data: Dict,
    finding: Optional[AssessmentResult] = None
) -> AIAnalysisResult:
    """
    Convenience function for AI advisory analysis
    """
    engine = AIAdvisoryEngine()
    return await engine.perform_advisory_analysis(finding_data, finding)


# Global AI advisory engine
global_ai_advisory_engine = AIAdvisoryEngine()