"""
VORTEX AI Triage-Only Mode - PHASE 5.1
AI kullanımını minimize eden akıllı triage sistemi

CRITICAL: AI sadece gerektiğinde çağrılır - ücretsiz LLM quota'yı korur
"""

import structlog
from typing import Dict, Optional, List
from datetime import datetime
from enum import Enum

from domain.models import AssessmentResult
from domain.enums import FindingType, VerificationStatus

logger = structlog.get_logger()


class TriageDecision(Enum):
    """Triage kararı"""
    SKIP_AI = "skip_ai"              # AI'ya gönderme, deterministik yeterli
    USE_AI = "use_ai"                 # AI analizi gerekli
    AUTO_ACCEPT = "auto_accept"       # Otomatik kabul (yüksek confidence)
    AUTO_REJECT = "auto_reject"       # Otomatik red (false positive)


class AITriageMode:
    """
    AI Triage-Only Mode
    AI kullanımını minimize eden akıllı karar sistemi
    
    Stratejiler:
    1. Deterministik analiz öncelikli
    2. AI sadece belirsiz durumlar için
    3. Ücretsiz LLM quota koruması
    """
    
    def __init__(self):
        self.stats = {
            'total_findings': 0,
            'ai_calls_saved': 0,
            'ai_calls_made': 0,
            'auto_accepted': 0,
            'auto_rejected': 0
        }
        
        # AI gerektiren durumlar
        self.ai_required_conditions = {
            'low_heuristic_confidence': 0.4,      # Düşük heuristic = AI gerekli
            'high_impact_threshold': 'HIGH',       # Yüksek impact = AI doğrula
            'complex_vulnerabilities': [           # Karmaşık zafiyetler
                FindingType.AUTH_BYPASS,           # authentication_bypass
                FindingType.AUTHZ_BYPASS,          # authorization_bypass
                FindingType.IDOR,                  # insecure_direct_object_reference
                FindingType.SQLI_BLIND,            # Blind SQL injection (karmaşık)
                FindingType.SSRF_BLIND             # Blind SSRF (karmaşık)
            ]
        }
        
        # Otomatik kabul koşulları (AI gerektirmez)
        self.auto_accept_conditions = {
            'min_heuristic_score': 0.85,           # Çok yüksek heuristic
            'min_deterministic_checks': 3,          # Birden fazla deterministic kanıt
            'strong_evidence_types': [             # Güçlü kanıt tipleri
                'direct_reflection',
                'error_message_match',
                'behavioral_confirmation'
            ]
        }
        
        # Otomatik red koşulları (AI gerektirmez)
        self.auto_reject_conditions = {
            'max_heuristic_score': 0.2,            # Çok düşük heuristic
            'known_false_positive_patterns': [     # Bilinen false positive'ler
                'waf_blocked',
                'rate_limited',
                'timeout_only'
            ]
        }
    
    def should_use_ai(
        self,
        finding: AssessmentResult,
        context: Optional[Dict] = None
    ) -> TriageDecision:
        """
        Finding için AI kullanılmalı mı karar ver
        
        Returns:
            TriageDecision: AI kullanım kararı
        """
        self.stats['total_findings'] += 1
        
        logger.info(
            "AI triage decision starting",
            finding_type=finding.finding_type.value if finding.finding_type else 'unknown',
            heuristic_score=finding.heuristic_score
        )
        
        # 1. OTOMATIK KABUL kontrolü (AI gerektirmez)
        if self._should_auto_accept(finding, context):
            self.stats['auto_accepted'] += 1
            self.stats['ai_calls_saved'] += 1
            logger.info(
                "Auto-accepted (AI skipped)",
                finding_id=str(finding.id),
                reason="High confidence deterministic evidence"
            )
            return TriageDecision.AUTO_ACCEPT
        
        # 2. OTOMATIK RED kontrolü (AI gerektirmez)
        if self._should_auto_reject(finding, context):
            self.stats['auto_rejected'] += 1
            self.stats['ai_calls_saved'] += 1
            logger.info(
                "Auto-rejected (AI skipped)",
                finding_id=str(finding.id),
                reason="Known false positive pattern"
            )
            return TriageDecision.AUTO_REJECT
        
        # 3. AI GEREKLİ MI kontrolü
        if self._requires_ai_analysis(finding, context):
            self.stats['ai_calls_made'] += 1
            logger.info(
                "AI analysis required",
                finding_id=str(finding.id),
                reasons=self._get_ai_requirement_reasons(finding)
            )
            return TriageDecision.USE_AI
        
        # 4. Varsayılan: AI atla (deterministik yeterli)
        self.stats['ai_calls_saved'] += 1
        logger.info(
            "AI skipped (deterministic sufficient)",
            finding_id=str(finding.id),
            heuristic_score=finding.heuristic_score
        )
        return TriageDecision.SKIP_AI
    
    def _should_auto_accept(
        self,
        finding: AssessmentResult,
        context: Optional[Dict]
    ) -> bool:
        """
        Otomatik kabul edilmeli mi?
        Yüksek confidence + güçlü kanıt = AI gerektirmez
        """
        # Yüksek heuristic score
        if finding.heuristic_score >= self.auto_accept_conditions['min_heuristic_score']:
            return True
        
        # Birden fazla deterministik kontrol geçti
        if context and 'deterministic_checks_passed' in context:
            checks_passed = context['deterministic_checks_passed']
            if checks_passed >= self.auto_accept_conditions['min_deterministic_checks']:
                return True
        
        # Güçlü kanıt tipi mevcut
        if finding.evidence:
            evidence_lower = finding.evidence.lower()
            for strong_evidence in self.auto_accept_conditions['strong_evidence_types']:
                if strong_evidence in evidence_lower:
                    return True
        
        return False
    
    def _should_auto_reject(
        self,
        finding: AssessmentResult,
        context: Optional[Dict]
    ) -> bool:
        """
        Otomatik red edilmeli mi?
        Düşük score + bilinen FP pattern = AI gerektirmez
        """
        # Çok düşük heuristic score
        if finding.heuristic_score <= self.auto_reject_conditions['max_heuristic_score']:
            return True
        
        # Bilinen false positive pattern
        if finding.evidence:
            evidence_lower = finding.evidence.lower()
            for fp_pattern in self.auto_reject_conditions['known_false_positive_patterns']:
                if fp_pattern in evidence_lower:
                    logger.debug(
                        "Known false positive pattern detected",
                        pattern=fp_pattern,
                        finding_id=str(finding.id)
                    )
                    return True
        
        # WAF block tespit edildi
        if context and context.get('waf_detected'):
            return True
        
        return False
    
    def _requires_ai_analysis(
        self,
        finding: AssessmentResult,
        context: Optional[Dict]
    ) -> bool:
        """
        AI analizi gerekli mi?
        Belirsiz durumlar veya karmaşık zafiyetler için AI kullan
        """
        # Düşük heuristic confidence (belirsiz)
        if finding.heuristic_score < self.ai_required_conditions['low_heuristic_confidence']:
            return True
        
        # Karmaşık zafiyet tipi
        if finding.finding_type in self.ai_required_conditions['complex_vulnerabilities']:
            return True
        
        # Yüksek impact iddiası (doğrulama gerekli)
        if context and context.get('claimed_impact') == self.ai_required_conditions['high_impact_threshold']:
            return True
        
        # Orta seviye confidence (0.4-0.85 arası) - AI yardımıyla kesinleştir
        if 0.4 <= finding.heuristic_score < 0.85:
            return True
        
        return False
    
    def _get_ai_requirement_reasons(self, finding: AssessmentResult) -> List[str]:
        """AI gerekliliğinin nedenlerini listele"""
        reasons = []
        
        if finding.heuristic_score < self.ai_required_conditions['low_heuristic_confidence']:
            reasons.append(f"Low heuristic confidence: {finding.heuristic_score:.2f}")
        
        if finding.finding_type in self.ai_required_conditions['complex_vulnerabilities']:
            reasons.append(f"Complex vulnerability type: {finding.finding_type.value}")
        
        if 0.4 <= finding.heuristic_score < 0.85:
            reasons.append(f"Medium confidence needs AI confirmation: {finding.heuristic_score:.2f}")
        
        return reasons
    
    def get_optimization_stats(self) -> Dict:
        """
        AI optimizasyon istatistikleri
        
        Returns:
            İstatistik dictionary
        """
        total = self.stats['total_findings']
        if total == 0:
            return {
                'total_findings': 0,
                'ai_usage_rate': 0.0,
                'ai_savings_rate': 0.0,
                'auto_triage_rate': 0.0
            }
        
        ai_calls = self.stats['ai_calls_made']
        ai_saved = self.stats['ai_calls_saved']
        auto_handled = self.stats['auto_accepted'] + self.stats['auto_rejected']
        
        return {
            'total_findings': total,
            'ai_calls_made': ai_calls,
            'ai_calls_saved': ai_saved,
            'ai_usage_rate': ai_calls / total,
            'ai_savings_rate': ai_saved / total,
            'auto_triage_rate': auto_handled / total,
            'auto_accepted': self.stats['auto_accepted'],
            'auto_rejected': self.stats['auto_rejected'],
            'estimated_quota_saved_percent': (ai_saved / total * 100) if total > 0 else 0
        }
    
    def reset_stats(self) -> None:
        """İstatistikleri sıfırla"""
        self.stats = {
            'total_findings': 0,
            'ai_calls_saved': 0,
            'ai_calls_made': 0,
            'auto_accepted': 0,
            'auto_rejected': 0
        }
        logger.info("AI triage stats reset")


# Global triage mode instance
global_triage_mode = AITriageMode()


def should_use_ai_for_finding(
    finding: AssessmentResult,
    context: Optional[Dict] = None
) -> TriageDecision:
    """
    Convenience function: Finding için AI kullanılmalı mı?
    
    Args:
        finding: Assessment result
        context: Opsiyonel context bilgileri
    
    Returns:
        TriageDecision
    """
    return global_triage_mode.should_use_ai(finding, context)