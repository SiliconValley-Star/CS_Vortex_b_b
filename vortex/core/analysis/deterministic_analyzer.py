"""
VORTEX Deterministic Analysis Engine - PHASE 5.3
AI'ya başvurmadan önce deterministik metodlarla analiz

CRITICAL: Deterministik analiz AI'dan önce - hız + quota tasarrufu
"""

import re
import structlog
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum

from domain.models import AssessmentResult
from domain.enums import FindingType, MatchType

logger = structlog.get_logger()


class DeterministicConfidence(Enum):
    """Deterministik analiz confidence seviyeleri"""
    VERY_HIGH = "very_high"      # 0.9+  - AI gerektirmez
    HIGH = "high"                 # 0.75-0.9 - AI opsiyonel
    MEDIUM = "medium"             # 0.5-0.75 - AI önerilir
    LOW = "low"                   # 0.3-0.5 - AI gerekli
    VERY_LOW = "very_low"         # <0.3 - AI kritik


class DeterministicAnalyzer:
    """
    Deterministik Analiz Motoru
    
    Stratejiler:
    1. Regex-based pattern matching
    2. Error message analysis
    3. Structural differential analysis
    4. Behavioral pattern recognition
    5. Known signature detection
    """
    
    def __init__(self):
        # SQL Injection error patterns
        self.sql_error_patterns = [
            r"SQL syntax.*?error",
            r"mysql_fetch",
            r"ORA-\d{5}",
            r"PostgreSQL.*?ERROR",
            r"Microsoft SQL Server",
            r"ODBC.*?Driver",
            r"SQLite.*?error",
            r"Warning.*?mysql_",
            r"pg_query\(\)",
            r"Unclosed quotation mark",
        ]
        
        # XSS reflection patterns
        self.xss_reflection_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"onerror\s*=",
            r"onload\s*=",
            r"<img[^>]*src",
            r"<iframe[^>]*>",
        ]
        
        # Path traversal patterns
        self.lfi_patterns = [
            r"/etc/passwd",
            r"C:\\Windows\\",
            r"root:.*?:/bin/",
            r"Administrator:",
        ]
        
        # SSRF indicators
        self.ssrf_indicators = [
            r"169\.254\.169\.254",  # AWS metadata
            r"metadata\.google\.internal",
            r"localhost",
            r"127\.0\.0\.1",
            r"0\.0\.0\.0",
        ]
        
        # Command injection indicators
        self.command_injection_patterns = [
            r"uid=\d+\(",
            r"gid=\d+\(",
            r"root:x:0:0:",
            r"bash.*?not found",
            r"command not found",
        ]
        
        self.stats = {
            'total_analyzed': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'ai_avoided': 0
        }
    
    def analyze(
        self,
        finding: AssessmentResult,
        response_data: Optional[Dict] = None
    ) -> Dict:
        """
        Deterministik analiz yap
        
        Args:
            finding: Assessment result
            response_data: Opsiyonel response data
        
        Returns:
            {
                'confidence_score': float,
                'confidence_level': DeterministicConfidence,
                'matches': List[Dict],
                'needs_ai': bool,
                'reasoning': str
            }
        """
        self.stats['total_analyzed'] += 1
        
        logger.info(
            "Starting deterministic analysis",
            finding_type=finding.finding_type.value if finding.finding_type else 'unknown'
        )
        
        # Evidence analizi
        matches = []
        total_score = 0.0
        
        if finding.evidence:
            # 1. Regex pattern matching
            regex_matches = self._regex_analysis(finding)
            matches.extend(regex_matches)
            total_score += sum(m['score'] for m in regex_matches)
            
            # 2. Error message analysis
            error_matches = self._error_message_analysis(finding)
            matches.extend(error_matches)
            total_score += sum(m['score'] for m in error_matches)
        
        # 3. Response data analysis
        if response_data:
            response_matches = self._response_analysis(response_data, finding)
            matches.extend(response_matches)
            total_score += sum(m['score'] for m in response_matches)
        
        # 4. Structural analysis
        structural_matches = self._structural_analysis(finding)
        matches.extend(structural_matches)
        total_score += sum(m['score'] for m in structural_matches)
        
        # Normalize score
        confidence_score = min(total_score, 1.0)
        confidence_level = self._get_confidence_level(confidence_score)
        
        # AI gerekli mi?
        needs_ai = confidence_level in [
            DeterministicConfidence.LOW,
            DeterministicConfidence.VERY_LOW
        ]
        
        if not needs_ai:
            self.stats['ai_avoided'] += 1
        
        # Confidence kategorisi istatistiği
        if confidence_level in [DeterministicConfidence.VERY_HIGH, DeterministicConfidence.HIGH]:
            self.stats['high_confidence'] += 1
        elif confidence_level == DeterministicConfidence.MEDIUM:
            self.stats['medium_confidence'] += 1
        else:
            self.stats['low_confidence'] += 1
        
        result = {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level.value,
            'matches': matches,
            'needs_ai': needs_ai,
            'reasoning': self._generate_reasoning(matches, confidence_level),
            'match_count': len(matches),
            'analysis_method': 'deterministic'
        }
        
        logger.info(
            "Deterministic analysis complete",
            confidence_score=confidence_score,
            confidence_level=confidence_level.value,
            needs_ai=needs_ai,
            match_count=len(matches)
        )
        
        return result
    
    def _regex_analysis(self, finding: AssessmentResult) -> List[Dict]:
        """Regex pattern matching analizi"""
        matches = []
        evidence = finding.evidence.lower()
        
        # Finding type'a göre pattern seçimi
        patterns_to_check = []
        
        if finding.finding_type in [FindingType.SQLI_ERROR, FindingType.SQLI_BLIND, FindingType.SQLI_TIME]:
            patterns_to_check = self.sql_error_patterns
            match_type = MatchType.EXACT_REGEX
            base_score = 0.4
        
        elif finding.finding_type in [FindingType.XSS_REFLECTED, FindingType.XSS_STORED, FindingType.XSS_DOM]:
            patterns_to_check = self.xss_reflection_patterns
            match_type = MatchType.PATTERN_MATCH
            base_score = 0.35
        
        elif finding.finding_type in [FindingType.LFI, FindingType.RFI]:
            patterns_to_check = self.lfi_patterns
            match_type = MatchType.EXACT_REGEX
            base_score = 0.4
        
        elif finding.finding_type in [FindingType.SSRF, FindingType.SSRF_BLIND]:
            patterns_to_check = self.ssrf_indicators
            match_type = MatchType.PATTERN_MATCH
            base_score = 0.35
        
        elif finding.finding_type in [FindingType.COMMAND_INJECTION, FindingType.CODE_INJECTION]:
            patterns_to_check = self.command_injection_patterns
            match_type = MatchType.EXACT_REGEX
            base_score = 0.4
        
        # Pattern matching
        for pattern in patterns_to_check:
            if re.search(pattern, evidence, re.IGNORECASE):
                matches.append({
                    'type': 'regex_match',
                    'pattern': pattern,
                    'match_type': match_type.value,
                    'score': base_score,
                    'description': f"Regex pattern matched: {pattern}"
                })
        
        return matches
    
    def _error_message_analysis(self, finding: AssessmentResult) -> List[Dict]:
        """Error message analizi"""
        matches = []
        evidence = finding.evidence.lower()
        
        # Bilinen error patterns
        error_signatures = {
            'sql_error': [
                'syntax error', 'mysql error', 'ora-', 'postgresql error',
                'sqlite error', 'sql server error'
            ],
            'path_error': [
                'no such file', 'access denied', 'permission denied',
                'file not found'
            ],
            'command_error': [
                'command not found', 'bash:', 'sh:', 'cannot execute'
            ]
        }
        
        for error_type, signatures in error_signatures.items():
            for sig in signatures:
                if sig in evidence:
                    matches.append({
                        'type': 'error_signature',
                        'signature': sig,
                        'error_type': error_type,
                        'match_type': MatchType.FUZZY_MATCH.value,
                        'score': 0.3,
                        'description': f"Known error signature: {sig}"
                    })
        
        return matches
    
    def _response_analysis(
        self,
        response_data: Dict,
        finding: AssessmentResult
    ) -> List[Dict]:
        """HTTP response analizi"""
        matches = []
        
        # Status code analizi
        status_code = response_data.get('status_code', 0)
        if status_code >= 500:
            matches.append({
                'type': 'status_code',
                'value': status_code,
                'match_type': MatchType.BEHAVIORAL_ONLY.value,
                'score': 0.2,
                'description': f"Server error status code: {status_code}"
            })
        
        # Response size anomalies
        response_size = response_data.get('content_length', 0)
        if response_size > 100000:  # 100KB'dan büyük
            matches.append({
                'type': 'response_size_anomaly',
                'size': response_size,
                'match_type': MatchType.BEHAVIORAL_ONLY.value,
                'score': 0.15,
                'description': f"Unusually large response: {response_size} bytes"
            })
        
        # Response time analysis
        response_time = response_data.get('response_time', 0)
        if response_time > 10.0 and finding.finding_type == FindingType.SQLI_TIME:
            matches.append({
                'type': 'timing_anomaly',
                'time': response_time,
                'match_type': MatchType.BEHAVIORAL_ONLY.value,
                'score': 0.35,
                'description': f"Time-based delay detected: {response_time}s"
            })
        
        return matches
    
    def _structural_analysis(self, finding: AssessmentResult) -> List[Dict]:
        """Structural differential analysis"""
        matches = []
        
        # Heuristic score zaten yüksekse
        if finding.heuristic_score >= 0.7:
            matches.append({
                'type': 'high_heuristic',
                'score': 0.2,
                'match_type': MatchType.STRUCTURAL_DIFFERENTIAL.value,
                'description': f"High heuristic confidence: {finding.heuristic_score:.2f}"
            })
        
        # Multiple evidence types
        if finding.evidence and len(finding.evidence.split('\n')) > 3:
            matches.append({
                'type': 'multiple_evidences',
                'score': 0.15,
                'match_type': MatchType.STRUCTURAL_DIFFERENTIAL.value,
                'description': "Multiple evidence lines detected"
            })
        
        return matches
    
    def _get_confidence_level(self, score: float) -> DeterministicConfidence:
        """Score'u confidence level'a çevir"""
        if score >= 0.9:
            return DeterministicConfidence.VERY_HIGH
        elif score >= 0.75:
            return DeterministicConfidence.HIGH
        elif score >= 0.5:
            return DeterministicConfidence.MEDIUM
        elif score >= 0.3:
            return DeterministicConfidence.LOW
        else:
            return DeterministicConfidence.VERY_LOW
    
    def _generate_reasoning(
        self,
        matches: List[Dict],
        confidence_level: DeterministicConfidence
    ) -> str:
        """Analiz reasoning'i oluştur"""
        if not matches:
            return "No deterministic matches found - AI analysis required"
        
        reasoning_parts = [
            f"Deterministic analysis confidence: {confidence_level.value}",
            f"Total matches: {len(matches)}"
        ]
        
        # Match tipi dağılımı
        match_types = {}
        for match in matches:
            mtype = match.get('type', 'unknown')
            match_types[mtype] = match_types.get(mtype, 0) + 1
        
        reasoning_parts.append("Match distribution:")
        for mtype, count in match_types.items():
            reasoning_parts.append(f"  - {mtype}: {count}")
        
        return "\n".join(reasoning_parts)
    
    def get_statistics(self) -> Dict:
        """Analiz istatistikleri"""
        total = self.stats['total_analyzed']
        if total == 0:
            return {
                'total_analyzed': 0,
                'ai_avoidance_rate': 0.0
            }
        
        return {
            'total_analyzed': total,
            'high_confidence_rate': self.stats['high_confidence'] / total,
            'medium_confidence_rate': self.stats['medium_confidence'] / total,
            'low_confidence_rate': self.stats['low_confidence'] / total,
            'ai_avoided': self.stats['ai_avoided'],
            'ai_avoidance_rate': self.stats['ai_avoided'] / total
        }


# Global deterministic analyzer instance
global_deterministic_analyzer = DeterministicAnalyzer()


def analyze_deterministically(
    finding: AssessmentResult,
    response_data: Optional[Dict] = None
) -> Dict:
    """
    Convenience function: Deterministik analiz yap
    
    Returns:
        Analysis result dict
    """
    return global_deterministic_analyzer.analyze(finding, response_data)